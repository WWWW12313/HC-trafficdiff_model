"""
train_causal_v7.py

v7 conditional causal diffusion training:
- Context (absolute spatiotemporal + OSM + weather) is passed as condition.
- Generated variables are only non-context columns.
- 6 atomic casualty bins are modeled as categorical diffusion targets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tab_ddpm.modules import MLP
from tab_ddpm.modules import timestep_embedding
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


class V7Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train") -> None:
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "info.json"), "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.X_ctx = np.load(os.path.join(data_dir, f"X_ctx_{split}.npy"), allow_pickle=True).astype(np.float32)
        self.X_num = np.load(os.path.join(data_dir, f"X_num_{split}.npy"), allow_pickle=True).astype(np.float32)
        self.X_cat = np.load(os.path.join(data_dir, f"X_cat_{split}.npy"), allow_pickle=True).astype(np.int64)

        self.num_columns = self.info.get("num_columns", [])
        self.cat_columns = self.info.get("cat_columns", [])
        self.context_columns = self.info.get("context_columns", [])
        self.cat_sizes = np.array(self.info.get("cat_sizes", []), dtype=int)

        data_parts = []
        if self.X_num.shape[1] > 0:
            data_parts.append(self.X_num)
        if self.X_cat.shape[1] > 0:
            data_parts.append(self.X_cat.astype(np.float32))
        self.X = np.concatenate(data_parts, axis=1) if data_parts else np.zeros((len(self.X_ctx), 0), dtype=np.float32)

        self.num_numerical_features = int(self.X_num.shape[1])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.X_ctx[idx], dtype=torch.float32),
        )


class ConditionalMLPDiffusion(nn.Module):
    """MLP denoiser with context injection through additive embedding."""

    def __init__(self, d_in: int, d_context: int, d_layers: list[int], dropout: float = 0.0, dim_t: int = 128):
        super().__init__()
        self.dim_t = dim_t

        self.proj_x = nn.Linear(d_in, dim_t)
        self.proj_ctx = nn.Sequential(
            nn.Linear(d_context, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        self.mlp = MLP.make_baseline(
            d_in=dim_t,
            d_layers=d_layers,
            dropout=dropout,
            d_out=d_in,
        )

    def forward(self, x, timesteps, y=None):
        emb_t = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        emb = emb_t
        if y is not None:
            if y.ndim == 1:
                y = y.unsqueeze(1)
            emb = emb + self.proj_ctx(y.float())
        h = self.proj_x(x) + emb
        return self.mlp(h)


def update_ema(target_params, source_params, rate=0.999):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def train_v7(
    data_dir: str,
    output_dir: str,
    mode: str,
    steps: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = V7Dataset(data_dir, split="train")

    profiles = {
        "quick": {"steps": 1000, "batch": 1024, "lr": 1e-3, "d_layers": [256, 256]},
        "balanced": {"steps": 5000, "batch": 1024, "lr": 2e-3, "d_layers": [768, 768, 768, 768]},
        "full": {"steps": 10000, "batch": 1024, "lr": 2e-3, "d_layers": [768, 768, 768, 768]},
    }
    if mode not in profiles:
        mode = "balanced"
    p = profiles[mode]

    total_steps = int(steps or p["steps"])
    batch_size = p["batch"]
    lr = p["lr"]

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    d_in_total = train_ds.num_numerical_features + int(train_ds.cat_sizes.sum())
    d_ctx = train_ds.X_ctx.shape[1]

    model = ConditionalMLPDiffusion(
        d_in=d_in_total,
        d_context=d_ctx,
        d_layers=p["d_layers"],
        dropout=0.0,
        dim_t=128,
    ).to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=train_ds.cat_sizes,
        num_numerical_features=train_ds.num_numerical_features,
        denoise_fn=model,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        scheduler="cosine",
        moment_matching_weight=0.0,
        moment_matching_target_index=None,
    ).to(device)

    ema_model = deepcopy(model)
    for p_ in ema_model.parameters():
        p_.detach_()

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-5)

    os.makedirs(output_dir, exist_ok=True)
    run_name = f"v7_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    it = iter(loader)
    log = []
    best_loss = float("inf")
    t0 = time.time()

    for step in range(total_steps):
        try:
            x, ctx = next(it)
        except StopIteration:
            it = iter(loader)
            x, ctx = next(it)

        x = x.to(device)
        ctx = ctx.to(device)

        optimizer.zero_grad()
        lm, lg = diffusion.mixed_loss(x, {"y": ctx})
        loss = lm + lg
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()
        update_ema(ema_model.parameters(), model.parameters(), rate=0.999)

        val = float(loss.item())
        if val < best_loss:
            best_loss = val
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
            torch.save(ema_model.state_dict(), os.path.join(output_dir, "model_ema_best.pt"))

        if (step + 1) % 100 == 0:
            log.append({"step": step + 1, "loss": val, "loss_multi": float(lm.item()), "loss_gauss": float(lg.item())})
            print(f"[v7] step={step+1}/{total_steps} loss={val:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    torch.save(ema_model.state_dict(), os.path.join(output_dir, "model_ema.pt"))
    torch.save(diffusion.state_dict(), os.path.join(output_dir, "diffusion.pt"))

    meta = {
        "run_name": run_name,
        "data_dir": data_dir,
        "mode": mode,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "lr": lr,
        "d_in_total": d_in_total,
        "d_context": d_ctx,
        "num_numerical_features": train_ds.num_numerical_features,
        "cat_sizes": train_ds.cat_sizes.tolist(),
        "context_columns": train_ds.context_columns,
        "num_columns": train_ds.num_columns,
        "cat_columns": train_ds.cat_columns,
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
        "best_loss": best_loss,
    }

    with open(os.path.join(output_dir, "train_summary_v7.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame(log).to_csv(os.path.join(output_dir, "loss_v7.csv"), index=False)

    joblib.dump(
        {
            "num_numerical_features": train_ds.num_numerical_features,
            "cat_sizes": train_ds.cat_sizes,
            "context_columns": train_ds.context_columns,
            "num_columns": train_ds.num_columns,
            "cat_columns": train_ds.cat_columns,
        },
        os.path.join(output_dir, "meta_v7.pkl"),
    )

    print("=== v7 training done ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v7 conditional causal diffusion")
    parser.add_argument("--data", type=str, default="data/nyc_crash_v7")
    parser.add_argument("--output", type=str, default="exp/nyc_crash_v7/causal_m4_v7")
    parser.add_argument("--mode", type=str, default="balanced", choices=["quick", "balanced", "full"])
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    train_v7(args.data, args.output, args.mode, args.steps)


if __name__ == "__main__":
    main()
