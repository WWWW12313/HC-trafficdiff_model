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

# Add project root to path
CDT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CDT_ROOT)
sys.path.insert(0, os.path.join(CDT_ROOT, "tabdiff"))

from tabdiff.datasets import TabDiffDataset
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.utils_train import Trainer, CDT_ROOT as TRAIN_ROOT
from tabdiff import src as cdt_src


def build_config(stage: int, device: str, tier: str = "quick") -> dict:
    config_path = os.path.join(CDT_ROOT, "tabdiff", "configs", "tabdiff_configs.toml")
    raw_config = cdt_src.load_config(config_path)
    t = {
        "quick": {"steps": 10, "batch_size": 256, "check_val_every": 5, "sample_size": 500, "num_timesteps": 10, "sample_batch_size": 500},
        "balanced": {"steps": 200, "batch_size": 512, "check_val_every": 50, "sample_size": 2000, "num_timesteps": 50, "sample_batch_size": 2000},
        "full": {"steps": 4000, "batch_size": 4096, "check_val_every": 500, "sample_size": None, "num_timesteps": 50, "sample_batch_size": 50000},
    }[tier]
    raw_config["train"]["main"]["steps"] = t["steps"]
    raw_config["train"]["main"]["batch_size"] = t["batch_size"]
    raw_config["train"]["main"]["check_val_every"] = t["check_val_every"]
    raw_config["diffusion_params"]["num_timesteps"] = t["num_timesteps"]
    raw_config["sample"]["batch_size"] = t["sample_batch_size"]
    if stage == 3:
        raw_config["train"]["main"]["lr"] = 0.001
        raw_config["train"]["main"]["closs_weight_schedule"] = "anneal"
        raw_config["unimodmlp_params"]["dropout"] = 0.15
    return raw_config


def load_checkpoint(ckpt_dir: str, device: str):
    """Load best EMA checkpoint from directory."""
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(best_path):
        # Try model_*.pt files
        pts = [f for f in os.listdir(ckpt_dir) if f.startswith("model_") and f.endswith(".pt")]
        if pts:
            best_path = os.path.join(ckpt_dir, sorted(pts)[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    
    print(f"[finetune] Loading checkpoint from {best_path}")
    state = torch.load(best_path, map_location=device, weights_only=False)
    return state


class FineTuneTrainer(Trainer):
    """Trainer that supports loading from checkpoint and fine-tuning with lower LR."""
    
    def __init__(self, *args, precomputed_masks=None, early_stop_patience=200, 
                 checkpoint_state=None, finetune_lr=None, finetune_epochs=None, **kwargs):
        super().__init__(*args, precomputed_masks=precomputed_masks, 
                        early_stop_patience=early_stop_patience, **kwargs)
        self.checkpoint_state = checkpoint_state
        self.finetune_lr = finetune_lr
        self.finetune_epochs = finetune_epochs
    
    def run(self, *args, **kwargs):
        # Override to load checkpoint before training
        if self.checkpoint_state is not None:
            self._load_from_checkpoint(self.checkpoint_state)
            print(f"[finetune] Loaded checkpoint, starting fine-tuning")
        
        # Override LR for fine-tuning
        if self.finetune_lr is not None:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.finetune_lr
            print(f"[finetune] Set fine-tune LR to {self.finetune_lr}")
        
        # Run training loop with limited epochs
        if self.finetune_epochs is not None:
            # The base Trainer uses steps, not epochs. We'll override the loop.
            return self._run_finetune_loop()
        else:
            return super().run(*args, **kwargs)
    
    def _load_from_checkpoint(self, state):
        """Load model weights from checkpoint state dict."""
        if "denoise_fn" in state:
            self.diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
            print("[finetune] Loaded denoise_fn from checkpoint")
        if "ema_model" in state or "ema" in state:
            ema_key = "ema_model" if "ema_model" in state else "ema"
            self.ema_model.load_state_dict(state[ema_key])
            print(f"[finetune] Loaded {ema_key} from checkpoint")
        if "num_schedule" in state:
            self.diffusion.num_schedule.load_state_dict(state["num_schedule"])
        if "cat_schedule" in state:
            self.diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    
    def _run_finetune_loop(self):
        """Simplified training loop for fine-tuning."""
        import math
        steps = self.finetune_epochs
        check_every = max(1, steps // 10)
        
        best_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(steps):
            self.diffusion._denoise_fn.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.diffusion(batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                # Update EMA
                if hasattr(self, 'ema_model') and self.ema_model is not None:
                    with torch.no_grad():
                        ema_decay = min(0.9999, 0.999 + epoch / (2 * steps))
                        for p_ema, p in zip(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()):
                            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
            
            avg_loss = epoch_loss / max(1, n_batches)
            
            if (epoch + 1) % check_every == 0 or epoch == 0:
                print(f"[finetune] Epoch {epoch+1}/{steps} | Loss={avg_loss:.4f} | LR={self.optimizer.param_groups[0]['lr']:.2e}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                # Save best
                state_dicts = {
                    "denoise_fn": self.diffusion._denoise_fn.state_dict(),
                    "ema_model": self.ema_model.state_dict() if hasattr(self, 'ema_model') else None,
                    "num_schedule": self.diffusion.num_schedule.state_dict(),
                    "cat_schedule": self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, "best_model.pt"))
        
        print(f"[finetune] Best loss={best_loss:.4f} at epoch {best_epoch+1}")
        print(f"[finetune] Saved to {self.model_save_path}/best_model.pt")


def load_precomputed_causal_masks(data_dir: str, device: str, mask_subdir: str = "causal_masks"):
    mask_dir = os.path.join(data_dir, mask_subdir)
    num_path = os.path.join(mask_dir, "num_causal_mask.npy")
    cat_path = os.path.join(mask_dir, "cat_causal_mask.npy")
    num_mask = np.load(num_path) if os.path.exists(num_path) else None
    cat_mask = np.load(cat_path) if os.path.exists(cat_path) else None
    if cat_mask is not None and cat_mask.size == 0:
        cat_mask = None
    return num_mask, cat_mask


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 2024 checkpoint on 2025 data")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Source checkpoint directory (2024)")
    parser.add_argument("--data_dir", type=str, required=True, help="Target data directory (2025)")
    parser.add_argument("--output_suffix", type=str, default="_ft2025", help="Suffix for output checkpoint dir")
    parser.add_argument("--epochs", type=int, default=500, help="Fine-tune epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tune learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default from tier)")
    parser.add_argument("--tier", type=str, default="full", choices=["quick", "balanced", "full"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lambda_causal", type=float, default=1.0)
    parser.add_argument("--no_causal_masks", action="store_true")
    parser.add_argument("--mask_subdir", type=str, default="causal_masks")
    parser.add_argument("--macro_relation_weight", type=float, default=0.0)
    args = parser.parse_args()
    
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    
    # Load source checkpoint
    ckpt_state = load_checkpoint(args.ckpt_dir, device)
    
    # Build config
    tier_cfg = {"quick": {"steps": 10, "batch_size": 256, "sample_size": 500},
                "balanced": {"steps": 200, "batch_size": 512, "sample_size": 2000},
                "full": {"steps": 4000, "batch_size": 4096, "sample_size": None}}[args.tier]
    
    raw_config = build_config(3, device, args.tier)
    if args.batch_size is not None:
        raw_config["train"]["main"]["batch_size"] = args.batch_size
        tier_cfg["batch_size"] = args.batch_size
    
    # Load target data
    info_path = os.path.join(args.data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    
    print(f"[finetune] Target data: {args.data_dir}")
    print(f"[finetune] Features: {info['n_num_features']} num, {info['n_cat_features']} cat")
    
    train_data = TabDiffDataset(
        os.path.basename(args.data_dir), args.data_dir, info,
        y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    
    sample_size = tier_cfg["sample_size"]
    if sample_size is not None and sample_size < len(train_data):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(train_data), size=sample_size, replace=False)
        train_data = torch.utils.data.Subset(train_data, indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=tier_cfg["batch_size"], shuffle=True, drop_last=True, num_workers=0
    )
    
    # Build model with same architecture as source
    n_num = info["n_num_features"]
    n_cat = info["n_cat_features"]
    num_list = info.get("num_list", [])
    cat_list = info.get("cat_list", [])
    
    num_mask, cat_mask = load_precomputed_causal_masks(args.data_dir, device, args.mask_subdir)
    if args.no_causal_masks:
        num_mask, cat_mask = None, None
    
    diffusion = UnifiedCtimeDiffusion(
        n_num_features=n_num, n_cat_features=n_cat,
        num_list=num_list, cat_list=cat_list,
        num_timesteps=raw_config["diffusion_params"]["num_timesteps"],
        model_params=raw_config["unimodmlp_params"],
        causal_weight_max=args.lambda_causal,
        num_causal_mask=num_mask, cat_causal_mask=cat_mask,
        device=device,
    )
    
    # Build output path
    src_name = os.path.basename(args.ckpt_dir)
    output_name = src_name + args.output_suffix
    output_dir = os.path.join(os.path.dirname(args.ckpt_dir), output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[finetune] Output dir: {output_dir}")
    print(f"[finetune] Epochs: {args.epochs}, LR: {args.lr}")
    
    # Build trainer
    optimizer = AdamW(diffusion._denoise_fn.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    trainer = FineTuneTrainer(
        model=diffusion, train_loader=train_loader, raw_config=raw_config,
        info=info, device=device,
        checkpoint_state=ckpt_state,
        finetune_lr=args.lr,
        finetune_epochs=args.epochs,
        model_save_path=output_dir,
        early_stop_patience=args.epochs,  # No early stopping for fine-tuning
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    
    # Run fine-tuning
    trainer.run()
    print(f"[finetune] Done. Checkpoint saved to {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
