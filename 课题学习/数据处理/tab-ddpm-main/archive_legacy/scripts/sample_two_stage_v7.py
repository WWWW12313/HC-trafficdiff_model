"""
sample_two_stage_v7.py

Two-stage inference for v7:
Stage 1: sample absolute anchors (LAT/LON + CRASH_DATE/TIME) with GMM/bootstrap.
API Bridge: attach physical context (OSM + weather), with robust fallback.
Stage 2: conditional causal diffusion sampling for non-context variables.
Finally enforce casualty logical constraints by exact arithmetic sums.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree

# Ensure project root is importable when executing from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_causal_v7 import ConditionalMLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

ATOMIC_TARGETS = [
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
]


def infer_time_period(hour: int) -> int:
    if 7 <= hour <= 9:
        return 1
    if 10 <= hour <= 15:
        return 2
    if 16 <= hour <= 19:
        return 3
    return 0


def normalize_speed_tag(v) -> str:
    if pd.isna(v):
        return "25 mph"
    t = str(v).strip()
    return t if t else "25 mph"


class Stage1AnchorSampler:
    def __init__(self, reference_csv: str, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.ref = pd.read_csv(reference_csv)
        self.ref.columns = self.ref.columns.str.strip()

        lat_src = self.ref["LATITUDE"] if "LATITUDE" in self.ref.columns else pd.Series(np.nan, index=self.ref.index)
        lon_src = self.ref["LONGITUDE"] if "LONGITUDE" in self.ref.columns else pd.Series(np.nan, index=self.ref.index)
        self.ref["LATITUDE"] = pd.to_numeric(lat_src, errors="coerce")
        self.ref["LONGITUDE"] = pd.to_numeric(lon_src, errors="coerce")

        dt = pd.to_datetime(
            self.ref["CRASH DATE"].astype(str).str.strip() + " " + self.ref["CRASH TIME"].astype(str).str.strip(),
            errors="coerce",
        )
        self.ref["_DT"] = dt
        self.ref = self.ref.dropna(subset=["LATITUDE", "LONGITUDE", "_DT"]).copy()

        self.gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=seed)
        self.gmm.fit(self.ref[["LATITUDE", "LONGITUDE"]].values)

    def sample(self, n: int) -> pd.DataFrame:
        latlon, _ = self.gmm.sample(n)
        lat = latlon[:, 0]
        lon = latlon[:, 1]

        dt_ref = self.ref["_DT"].values
        idx = self.rng.integers(0, len(dt_ref), size=n)
        dt = pd.Series(pd.to_datetime(dt_ref[idx]))

        # add small minute jitter
        jitter = self.rng.integers(-15, 16, size=n)
        dt = dt + pd.to_timedelta(jitter, unit="m")

        out = pd.DataFrame({
            "LATITUDE": lat,
            "LONGITUDE": lon,
            "CRASH_DATE": dt.dt.strftime("%m/%d/%Y"),
            "CRASH_TIME": dt.dt.strftime("%H:%M"),
            "CRASH_DATE_TS": (dt.astype("int64") // 10**9).astype(np.int64),
            "CRASH_TIME_MIN": (dt.dt.hour * 60 + dt.dt.minute).astype(int),
            "DAY_OF_WEEK": dt.dt.weekday.astype(int),
            "CRASH_TIME_PERIOD": dt.dt.hour.map(infer_time_period).astype(int),
        })
        return out


def enrich_context_from_reference(anchors: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref.columns = ref.columns.str.strip()

    cols = [
        "LATITUDE", "LONGITUDE",
        "OSM_TYPE", "OSM_ONEWAY", "OSM_SPEED_TAG", "DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL",
        "TEMP_C", "prcp", "WIND_SPEED_KMH", "coco",
    ]
    for c in cols:
        if c not in ref.columns:
            ref[c] = np.nan

    ref = ref.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    ref_coords = np.radians(ref[["LATITUDE", "LONGITUDE"]].astype(float).values)
    anc_coords = np.radians(anchors[["LATITUDE", "LONGITUDE"]].astype(float).values)

    tree = BallTree(ref_coords, metric="haversine")
    _, nn_idx = tree.query(anc_coords, k=1)
    nn_idx = nn_idx.flatten()

    out = anchors.copy()
    nn = ref.iloc[nn_idx].reset_index(drop=True)

    out["CTX_HIGHWAY"] = nn["OSM_TYPE"].fillna("residential").astype(str)
    out["CTX_ONEWAY"] = pd.to_numeric(nn["OSM_ONEWAY"], errors="coerce").fillna(0).astype(int)
    out["CTX_MAXSPEED"] = nn["OSM_SPEED_TAG"].map(normalize_speed_tag)
    out["CTX_CYCLEWAY"] = "no"
    out["CTX_DIST_TO_INTERSECTION"] = pd.to_numeric(nn["DIST_TO_SIGNAL_M"], errors="coerce").fillna(500.0).astype(float)
    out["CTX_IS_SIGNALIZED"] = pd.to_numeric(nn["HAS_TRAFFIC_SIGNAL"], errors="coerce").fillna(0).astype(int)

    out["CTX_TEMP"] = pd.to_numeric(nn["TEMP_C"], errors="coerce").fillna(15.0).astype(float)
    out["CTX_PRCP"] = pd.to_numeric(nn["prcp"], errors="coerce").fillna(0.0).astype(float)
    out["CTX_WSPD"] = pd.to_numeric(nn["WIND_SPEED_KMH"], errors="coerce").fillna(10.0).astype(float)
    out["CTX_COCO"] = pd.to_numeric(nn["coco"], errors="coerce").fillna(1).astype(int)

    return out


def encode_context(anchors_ctx: pd.DataFrame, info: dict, column_mapping: dict) -> np.ndarray:
    ctx_numeric = info["context_numeric"]
    ctx_categorical = info["context_categorical"]

    num_mat = (
        anchors_ctx[ctx_numeric].to_numpy(dtype=np.float32, copy=False)
        if ctx_numeric
        else np.zeros((len(anchors_ctx), 0), dtype=np.float32)
    )

    cat_arrs: List[np.ndarray] = []
    for c in ctx_categorical:
        c2i = column_mapping.get(c, {})
        vals = (
            anchors_ctx[c]
            .fillna("<NA>")
            .astype(str)
            .map(lambda x: c2i.get(x, 0))
            .astype(float)
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )
        cat_arrs.append(vals)
    cat_mat = np.concatenate(cat_arrs, axis=1) if cat_arrs else np.zeros((len(anchors_ctx), 0), dtype=np.float32)
    mats: List[np.ndarray] = [num_mat, cat_mat]
    return np.concatenate(mats, axis=1).astype(np.float32, copy=False)


def decode_generated_cat(samples_cat: np.ndarray, info: dict, column_mapping: dict) -> pd.DataFrame:
    cat_cols = info["cat_columns"]
    out = pd.DataFrame()
    for j, col in enumerate(cat_cols):
        c2i = column_mapping.get(col, {})
        i2c = {v: k for k, v in c2i.items()}
        out[col] = [i2c.get(int(v), str(int(v))) for v in samples_cat[:, j]]
    return out


def sample_with_context(diffusion: GaussianMultinomialDiffusion, context: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    diff = cast(Any, diffusion)
    device = context.device
    n = context.shape[0]
    all_rows = []

    num_num = int(diff.num_numerical_features)
    if isinstance(diff.num_classes, torch.Tensor):
        num_classes = diff.num_classes.detach().cpu().numpy().astype(int)
    else:
        num_classes = np.array(diff.num_classes, dtype=int)
    has_cat = len(num_classes) > 0 and int(num_classes[0]) != 0

    start = 0
    while start < n:
        end = min(start + batch_size, n)
        ctx = context[start:end]
        b = ctx.shape[0]

        z_norm = torch.randn((b, num_num), device=device)
        if has_cat:
            if isinstance(diff.num_classes_expanded, torch.Tensor):
                class_dim = int(diff.num_classes_expanded.detach().cpu().shape[0])
            else:
                class_dim = int(np.array(diff.num_classes_expanded).shape[0])
            uniform_logits = torch.zeros((b, class_dim), device=device)
            log_z = diff.log_sample_categorical(uniform_logits)
        else:
            log_z = torch.zeros((b, 0), device=device)

        out_dict = {"y": ctx}
        for i in reversed(range(0, int(diff.num_timesteps))):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = diff._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, :num_num]
            model_out_cat = model_out[:, num_num:]

            if num_num > 0:
                z_norm = diff.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=True)["sample"]
            if has_cat:
                log_z = diff.p_sample(model_out_cat, log_z, t, out_dict)

        if has_cat:
            z_ohe = torch.exp(log_z).round()
            z_cat = diff.ohe_to_categories(z_ohe, diff.num_classes) if hasattr(diff, "ohe_to_categories") else None
            if z_cat is None:
                from tab_ddpm.utils import ohe_to_categories
                z_cat = ohe_to_categories(z_ohe, num_classes)
            row = torch.cat([z_norm, z_cat], dim=1).cpu().numpy()
        else:
            row = z_norm.cpu().numpy()

        all_rows.append(row)
        start = end

    return np.concatenate(all_rows, axis=0)


def enforce_casualty_logic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # decode bins back to conservative counts (0,1,2,3)
    for c in ATOMIC_TARGETS:
        bcol = f"{c}_BIN"
        if bcol in out.columns:
            out[c] = pd.to_numeric(out[bcol], errors="coerce").fillna(0).clip(lower=0, upper=3).astype(int)

    out["NUMBER OF PERSONS INJURED"] = (
        out["NUMBER OF PEDESTRIANS INJURED"]
        + out["NUMBER OF CYCLIST INJURED"]
        + out["NUMBER OF MOTORIST INJURED"]
    ).astype(int)

    out["NUMBER OF PERSONS KILLED"] = (
        out["NUMBER OF PEDESTRIANS KILLED"]
        + out["NUMBER OF CYCLIST KILLED"]
        + out["NUMBER OF MOTORIST KILLED"]
    ).astype(int)

    return out


def run_two_stage(
    model_dir: str,
    data_dir: str,
    reference_csv: str,
    output_csv: str,
    n_samples: int,
    seed: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = json.load(open(os.path.join(data_dir, "info.json"), "r", encoding="utf-8"))
    col_mapping = json.load(open(os.path.join(data_dir, "column_mapping.json"), "r", encoding="utf-8"))

    # Stage 1 anchors + context
    s1 = Stage1AnchorSampler(reference_csv=reference_csv, seed=seed)
    anchors = s1.sample(n_samples)
    anchors_ctx = enrich_context_from_reference(anchors, reference_csv=reference_csv)

    X_ctx = encode_context(anchors_ctx, info, col_mapping)
    X_ctx_t = torch.tensor(X_ctx, dtype=torch.float32, device=device)

    # Build model and diffusion from metadata
    meta = json.load(open(os.path.join(model_dir, "train_summary_v7.json"), "r", encoding="utf-8"))
    d_in_total = int(meta["d_in_total"])
    d_ctx = int(meta["d_context"])
    num_num = int(meta["num_numerical_features"])
    cat_sizes = np.array(meta["cat_sizes"], dtype=int)

    denoise = ConditionalMLPDiffusion(
        d_in=d_in_total,
        d_context=d_ctx,
        d_layers=[768, 768, 768, 768] if meta.get("mode") != "quick" else [256, 256],
        dropout=0.0,
        dim_t=128,
    ).to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=cat_sizes,
        num_numerical_features=num_num,
        denoise_fn=denoise,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        scheduler="cosine",
    ).to(device)

    denoise.load_state_dict(torch.load(os.path.join(model_dir, "model_ema.pt"), map_location=device))
    denoise.eval()
    diffusion.eval()

    # Stage 2 sampling
    with torch.no_grad():
        gen = sample_with_context(diffusion, X_ctx_t, batch_size=1024)

    gen_num = gen[:, :num_num] if num_num > 0 else np.zeros((n_samples, 0), dtype=np.float32)
    gen_cat = gen[:, num_num:].astype(int) if gen.shape[1] > num_num else np.zeros((n_samples, 0), dtype=int)

    df_gen_cat = decode_generated_cat(gen_cat, info, col_mapping)

    out = anchors_ctx.copy()
    for c in df_gen_cat.columns:
        out[c] = df_gen_cat[c]

    out = enforce_casualty_logic(out)

    # Keep IS_* columns at end if exist
    is_cols = [c for c in out.columns if c.startswith("IS_")]
    other_cols = [c for c in out.columns if not c.startswith("IS_")]
    out = out[other_cols + is_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report = {
        "model_dir": model_dir,
        "data_dir": data_dir,
        "reference_csv": reference_csv,
        "output_csv": output_csv,
        "rows": int(len(out)),
        "cols": int(out.shape[1]),
    }
    with open(Path(output_csv).with_suffix(".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== v7 two-stage sampling done ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage v7 sampling")
    parser.add_argument("--model_dir", type=str, default="exp/nyc_crash_v7/causal_m4_v7")
    parser.add_argument("--data_dir", type=str, default="data/nyc_crash_v7")
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--output_csv", type=str, default="exp/nyc_crash_v7/causal_m4_v7/synthetic_2017_v7.csv")
    parser.add_argument("--n_samples", type=int, default=159992)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_two_stage(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        reference_csv=args.reference_csv,
        output_csv=args.output_csv,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
