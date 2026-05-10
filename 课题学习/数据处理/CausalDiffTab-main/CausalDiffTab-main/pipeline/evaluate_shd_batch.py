"""
Batch SHD evaluation for synthetic CSV files.

This script extracts a causal adjacency matrix from each synthetic CSV using
local NOTEARS linear backend, then computes SHD against a real adjacency matrix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Use local bundled NOTEARS package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "notears"))

from notears.linear import notears_linear  # type: ignore

from evaluate_joint_metrics import structural_hamming_distance


def _load_info(root: Path) -> Dict[str, Any]:
    p = root / "data" / "nyc_crash" / "info.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = float(out[c].median()) if out[c].notna().any() else 0.0
        out[c] = out[c].fillna(med)
    return out


def _encode_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        codes, _ = pd.factorize(out[c].astype(str).fillna("_nan_"), sort=True)
        out[c] = codes.astype(np.float64)
    return out


def _standardize(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = (X - mean) / std
    return np.where(np.isfinite(Z), Z, 0.0)


def _infer_model_and_nrows(file_name: str) -> Tuple[str, int]:
    base = file_name.replace(".csv", "")
    model = re.sub(r"_compare_n\d+$", "", base)
    m = re.search(r"_compare_n(\d+)$", base)
    n_rows = int(m.group(1)) if m else -1
    return model, n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch SHD evaluator for synthetic compare CSVs")
    parser.add_argument("--real_adj", type=str, default=str(ROOT / "configs" / "causal_matrix_notears_mlp.npy"))
    parser.add_argument("--synthetic_dir", type=str, default=str(ROOT / "results" / "synthetic"))
    parser.add_argument("--file_glob", type=str, default="*_compare_n*.csv")
    parser.add_argument("--max_rows", type=int, default=3000)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--w_threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    info = _load_info(ROOT)
    num_cols = list(info.get("num_col_names", []))
    cat_cols = list(info.get("cat_col_names", []))
    feat_cols = num_cols + cat_cols

    real_adj = np.load(args.real_adj)
    if real_adj.ndim != 2 or real_adj.shape[0] != real_adj.shape[1]:
        raise SystemExit(f"real adjacency must be square matrix, got {real_adj.shape}")
    d = int(real_adj.shape[0])

    syn_dir = Path(args.synthetic_dir)
    files = sorted([p for p in syn_dir.glob(args.file_glob) if p.is_file() and not p.name.startswith("_")])
    if not files:
        raise SystemExit(f"No synthetic files matched in {syn_dir} with glob={args.file_glob}")

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []

    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        cols = [c for c in feat_cols if c in df.columns]
        if len(cols) != d:
            # Keep only matching dims. If mismatch remains, skip to avoid invalid SHD.
            if len(cols) != d:
                rows.append(
                    {
                        "file": fp.name,
                        "model": _infer_model_and_nrows(fp.name)[0],
                        "n_rows": _infer_model_and_nrows(fp.name)[1],
                        "sampled_rows": 0,
                        "syn_edges": np.nan,
                        "shd": np.nan,
                        "shd_normalized": np.nan,
                        "error": f"feature_dim_mismatch cols={len(cols)} real_d={d}",
                    }
                )
                continue

        sub = df[cols].copy()
        sub = _to_numeric(sub, [c for c in num_cols if c in sub.columns])
        sub = _encode_categorical(sub, [c for c in cat_cols if c in sub.columns])

        if len(sub) > args.max_rows:
            idx = rng.choice(len(sub), size=args.max_rows, replace=False)
            sub = sub.iloc[idx].copy()

        X = _standardize(sub.values.astype(np.float64))

        try:
            W = notears_linear(
                X,
                lambda1=args.lambda1,
                loss_type="l2",
                max_iter=args.max_iter,
                w_threshold=args.w_threshold,
            )
            syn_adj = (np.abs(W) > 0).astype(np.int32)
            shd = structural_hamming_distance(real_adj, syn_adj)
            shd_norm = float(shd / max(d * (d - 1), 1))
            model, n_rows = _infer_model_and_nrows(fp.name)
            rows.append(
                {
                    "file": fp.name,
                    "model": model,
                    "n_rows": n_rows,
                    "sampled_rows": int(len(sub)),
                    "syn_edges": int(np.sum(syn_adj)),
                    "shd": int(shd),
                    "shd_normalized": round(shd_norm, 6),
                    "error": "",
                }
            )
            print(f"[ok] {fp.name}: shd={shd}, shd_norm={shd_norm:.6f}")
        except Exception as e:
            model, n_rows = _infer_model_and_nrows(fp.name)
            rows.append(
                {
                    "file": fp.name,
                    "model": model,
                    "n_rows": n_rows,
                    "sampled_rows": int(len(sub)),
                    "syn_edges": np.nan,
                    "shd": np.nan,
                    "shd_normalized": np.nan,
                    "error": str(e),
                }
            )
            print(f"[error] {fp.name}: {e}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"shd_report_{ts}.csv"
    json_path = out_dir / f"shd_report_{ts}.json"

    out_df = pd.DataFrame(rows).sort_values(["model", "n_rows", "file"])
    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "generated_utc": ts,
        "real_adj": str(args.real_adj),
        "file_glob": args.file_glob,
        "max_rows": args.max_rows,
        "lambda1": args.lambda1,
        "max_iter": args.max_iter,
        "w_threshold": args.w_threshold,
        "rows": out_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_csv = out_dir / "shd_report_latest.csv"
    latest_json = out_dir / "shd_report_latest.json"
    out_df.to_csv(latest_csv, index=False, encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[write] {csv_path}")
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()
