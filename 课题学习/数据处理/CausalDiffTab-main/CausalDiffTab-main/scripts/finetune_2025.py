#!/usr/bin/env python
"""
Fine-tune a 2024 checkpoint on 2025 data.

Usage:
    python scripts/finetune_2025.py \
        --ckpt_dir ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2 \
        --data_dir data/nyc_crash_2025_v2 \
        --output_suffix _ft2025 \
        --epochs 500 --lr 1e-4 --device cuda:0
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

# Add project root to path
CDT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CDT_ROOT)
sys.path.insert(0, os.path.join(CDT_ROOT, "tabdiff"))

from utils_train import TabDiffDataset
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
import src as cdt_src


def load_checkpoint(ckpt_dir: str, device: str):
    """Load best checkpoint from directory."""
    pt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_model_") and f.endswith(".pt")]
    if pt_files:
        ckpt_name = sorted(pt_files)[0]
    else:
        pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        ckpt_name = sorted(pt_files)[-1] if pt_files else None
    
    if ckpt_name is None:
        raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")
    
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(f"[finetune] Loading checkpoint: {ckpt_name}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    return state


def build_model(data_dir: str, ckpt_dir: str, device: str):
    """Build UnifiedCtimeDiffusion with same architecture as source checkpoint."""
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        raw_config = cdt_src.pickle.load(f)
    
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    dataset = TabDiffDataset(
        os.path.basename(data_dir), data_dir, info,
        y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    
    d_numerical = dataset.d_numerical
    categories = dataset.categories
    
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (
        (categories + 1).tolist() if len(categories) > 0 else []
    )
    
    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)
    
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
        causal_weight_max=1.0,
        causal_warmup_steps=1,
    )
    diffusion.to(device)
    
    return diffusion, dataset, info, raw_config


def update_ema(ema_model, model, ema_decay):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 2024 checkpoint on 2025 data")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Source checkpoint directory (2024)")
    parser.add_argument("--data_dir", type=str, required=True, help="Target data directory (2025)")
    parser.add_argument("--output_suffix", type=str, default="_ft2025", help="Suffix for output checkpoint dir")
    parser.add_argument("--epochs", type=int, default=500, help="Fine-tune epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tune learning rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    args = parser.parse_args()
    
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    
    # Load source checkpoint state
    ckpt_state = load_checkpoint(args.ckpt_dir, device)
    
    # Build model
    diffusion, dataset, info, raw_config = build_model(args.data_dir, args.ckpt_dir, device)
    
    # Load weights from checkpoint
    diffusion._denoise_fn.load_state_dict(ckpt_state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(ckpt_state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(ckpt_state["cat_schedule"])
    print("[finetune] Loaded checkpoint weights into model")
    
    # Build EMA model
    ema_model = deepcopy(diffusion._denoise_fn)
    for param in ema_model.parameters():
        param.detach_()
    
    # Build output path
    src_name = os.path.basename(args.ckpt_dir)
    output_name = src_name + args.output_suffix
    output_dir = os.path.join(os.path.dirname(args.ckpt_dir), output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    import pickle
    with open(os.path.join(output_dir, "config.pkl"), "wb") as f:
        pickle.dump(raw_config, f)
    
    print(f"[finetune] Target data: {args.data_dir}")
    print(f"[finetune] Features: {info['n_num_features']} num, {info['n_cat_features']} cat")
    print(f"[finetune] Output dir: {output_dir}")
    print(f"[finetune] Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    
    # Build dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(diffusion._denoise_fn.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    steps = args.epochs
    check_every = max(1, steps // 10)
    
    best_loss = float('inf')
    best_epoch = 0
    
    diffusion.train()
    for epoch in range(steps):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.float().to(device)
            optimizer.zero_grad()
            dloss, closs = diffusion.mixed_loss(batch, epoch)
            loss = dloss + closs
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Update EMA
            ema_decay = min(0.9999, args.ema_decay + epoch / (2 * steps))
            update_ema(ema_model, diffusion._denoise_fn, ema_decay)
        
        avg_loss = epoch_loss / max(1, n_batches)
        
        if (epoch + 1) % check_every == 0 or epoch == 0:
            print(f"[finetune] Epoch {epoch+1}/{steps} | Loss={avg_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            # Save best
            state_dicts = {
                "denoise_fn": diffusion._denoise_fn.state_dict(),
                "ema_model": ema_model.state_dict(),
                "num_schedule": diffusion.num_schedule.state_dict(),
                "cat_schedule": diffusion.cat_schedule.state_dict(),
            }
            torch.save(state_dicts, os.path.join(output_dir, "best_model.pt"))
    
    print(f"[finetune] Best loss={best_loss:.4f} at epoch {best_epoch+1}")
    print(f"[finetune] Saved to {output_dir}/best_model.pt")
    print(f"[finetune] Done.")


if __name__ == "__main__":
    main()
