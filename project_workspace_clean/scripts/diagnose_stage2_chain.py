from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


DEFAULT_NUMERIC = [
    "LATITUDE",
    "LONGITUDE",
    "CRASH_TIME_SIN",
    "CRASH_TIME_COS",
    "TEMP_C",
    "prcp",
    "WIND_SPEED_KMH",
    "DIST_TO_SIGNAL_M",
    "REAL_SPEED_LIMIT",
    "INFERRED_LANES",
]

STAGE1_NUMERIC = ["LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS"]
STAGE2_NUMERIC = [
    "TEMP_C",
    "prcp",
    "WIND_SPEED_KMH",
    "DIST_TO_SIGNAL_M",
    "REAL_SPEED_LIMIT",
    "INFERRED_LANES",
]
STAGE1_CATEGORICAL = ["SEASON", "DAY_OF_WEEK", "TIME_PERIOD"]
STAGE2_CATEGORICAL = [
    "HAS_TRAFFIC_SIGNAL",
    "OSM_ONEWAY",
    "HAS_DIVIDER",
    "coco",
    "WEATHER_CONDITION",
    "OSM_TYPE",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def value_counts_js(left: pd.Series, right: pd.Series) -> float:
    left = left.astype(str).fillna("__NA__")
    right = right.astype(str).fillna("__NA__")
    labels = sorted(set(left.unique()) | set(right.unique()))
    if len(labels) < 2:
        return 0.0
    l_counts = left.value_counts(normalize=True)
    r_counts = right.value_counts(normalize=True)
    p = np.array([float(l_counts.get(x, 0.0)) for x in labels], dtype=float)
    q = np.array([float(r_counts.get(x, 0.0)) for x in labels], dtype=float)
    return float(jensenshannon(p, q, base=2.0))


def compare_numeric(syn: pd.DataFrame, real: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in syn.columns or col not in real.columns:
            continue
        s = numeric_series(syn, col)
        r = numeric_series(real, col)
        if len(s) < 2 or len(r) < 2:
            continue
        rows.append(
            {
                "column": col,
                "wasserstein": float(wasserstein_distance(s.to_numpy(), r.to_numpy())),
                "syn_mean": float(s.mean()),
                "real_mean": float(r.mean()),
                "mean_delta": float(s.mean() - r.mean()),
                "syn_std": float(s.std(ddof=0)),
                "real_std": float(r.std(ddof=0)),
                "syn_min": float(s.min()),
                "syn_max": float(s.max()),
                "real_min": float(r.min()),
                "real_max": float(r.max()),
            }
        )
    return sorted(rows, key=lambda x: x["wasserstein"], reverse=True)


def compare_categorical(syn: pd.DataFrame, real: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in syn.columns or col not in real.columns:
            continue
        js = value_counts_js(syn[col], real[col])
        syn_top = syn[col].astype(str).value_counts(normalize=True).head(5).to_dict()
        real_top = real[col].astype(str).value_counts(normalize=True).head(5).to_dict()
        rows.append({"column": col, "js": js, "syn_top": syn_top, "real_top": real_top})
    return sorted(rows, key=lambda x: x["js"], reverse=True)


def coordinate_support(syn: pd.DataFrame, real: pd.DataFrame, name: str) -> dict[str, Any]:
    cols = ["LATITUDE", "LONGITUDE"]
    if any(c not in syn.columns or c not in real.columns for c in cols):
        return {"name": name, "status": "missing coordinate columns"}
    syn_xy = syn[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    real_xy = real[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(syn_xy) == 0 or len(real_xy) == 0:
        return {"name": name, "status": "empty coordinates"}
    tree = cKDTree(real_xy)
    dists, _ = tree.query(syn_xy, k=1)
    nyc_mask = (
        (syn_xy[:, 0] >= 40.40)
        & (syn_xy[:, 0] <= 40.95)
        & (syn_xy[:, 1] >= -74.30)
        & (syn_xy[:, 1] <= -73.70)
    )
    return {
        "name": name,
        "n": int(len(syn_xy)),
        "nyc_bbox_rate": float(nyc_mask.mean()),
        "nearest_real_deg_mean": float(np.mean(dists)),
        "nearest_real_deg_p50": float(np.quantile(dists, 0.50)),
        "nearest_real_deg_p95": float(np.quantile(dists, 0.95)),
        "nearest_real_deg_p99": float(np.quantile(dists, 0.99)),
    }


def nearest_coordinate_feature_error(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    name: str,
    cols: list[str],
) -> list[dict[str, Any]]:
    coord_cols = ["LATITUDE", "LONGITUDE"]
    if any(c not in syn.columns or c not in real.columns for c in coord_cols):
        return []
    syn_coord = syn[coord_cols].apply(pd.to_numeric, errors="coerce")
    real_coord = real[coord_cols].apply(pd.to_numeric, errors="coerce")
    syn_valid = syn_coord.notna().all(axis=1)
    real_valid = real_coord.notna().all(axis=1)
    if not syn_valid.any() or not real_valid.any():
        return []
    syn_xy = syn_coord.loc[syn_valid].to_numpy(dtype=float)
    real_xy = real_coord.loc[real_valid].to_numpy(dtype=float)
    real_sub = real.loc[real_valid].reset_index(drop=True)
    syn_sub = syn.loc[syn_valid].reset_index(drop=True)
    tree = cKDTree(real_xy)
    coord_dist, idx = tree.query(syn_xy, k=1)
    rows = []
    for col in cols:
        if col not in syn_sub.columns or col not in real_sub.columns:
            continue
        syn_vals = pd.to_numeric(syn_sub[col], errors="coerce").to_numpy(dtype=float)
        real_vals = pd.to_numeric(real_sub.iloc[idx][col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(syn_vals) & np.isfinite(real_vals)
        if not valid.any():
            continue
        err = syn_vals[valid] - real_vals[valid]
        rows.append(
            {
                "name": name,
                "column": col,
                "nearest_coord_deg_mean": float(np.mean(coord_dist[valid])),
                "mae_to_nearest_real": float(np.mean(np.abs(err))),
                "mean_error": float(np.mean(err)),
                "p95_abs_error": float(np.quantile(np.abs(err), 0.95)),
            }
        )
    return sorted(rows, key=lambda x: x["mae_to_nearest_real"], reverse=True)


def rule_violations(df: pd.DataFrame, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "n_rows": int(len(df))}
    checks = {
        "negative_prcp_rate": ("prcp", lambda s: s < 0),
        "negative_wind_rate": ("WIND_SPEED_KMH", lambda s: s < 0),
        "invalid_speed_rate": ("REAL_SPEED_LIMIT", lambda s: (s < 0) | (s > 80)),
        "invalid_lanes_rate": ("INFERRED_LANES", lambda s: (s < 0) | (s > 12)),
        "negative_signal_dist_rate": ("DIST_TO_SIGNAL_M", lambda s: s < 0),
    }
    for key, (col, fn) in checks.items():
        vals = numeric_series(df, col)
        out[key] = None if len(vals) == 0 else float(fn(vals).mean())
    for col in ["HAS_DIVIDER", "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY"]:
        if col in df.columns:
            out[f"{col}_nunique"] = int(df[col].nunique(dropna=False))
            out[f"{col}_top"] = df[col].astype(str).value_counts(normalize=True).head(5).to_dict()
    return out


def mask_summary(mask_dir: Path, info_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"mask_dir": str(mask_dir)}
    info = read_json(info_path) if info_path.is_file() else {}
    num_path = mask_dir / "num_causal_mask.npy"
    cat_path = mask_dir / "cat_causal_mask.npy"
    if num_path.is_file():
        num = np.load(num_path)
        out["num_shape"] = list(num.shape)
        out["num_sum"] = float(num.sum())
        out["num_nonzero"] = int(np.count_nonzero(num))
        out["num_density"] = float(np.count_nonzero(num) / num.size)
        names = [info.get("target_col", "target")] + list(info.get("num_col_names", []))
        entries = []
        for i, j in np.argwhere(num > 0):
            entries.append({"src": names[i] if i < len(names) else str(i), "dst": names[j] if j < len(names) else str(j), "weight": float(num[i, j])})
        out["num_edges_top"] = entries[:80]
    if cat_path.is_file():
        cat = np.load(cat_path)
        out["cat_shape"] = list(cat.shape)
        out["cat_sum"] = float(cat.sum())
        out["cat_nonzero"] = int(np.count_nonzero(cat))
        out["cat_density"] = float(np.count_nonzero(cat) / cat.size)
    return out


def markdown_table(rows: list[dict[str, Any]], keys: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "(empty)\n"
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage2 Chain Causal Diagnostics\n")
    lines.append("## Key Findings\n")
    for item in report["key_findings"]:
        lines.append(f"- {item}\n")
    lines.append("\n## Final Synthetic Numeric Drift vs 2024\n")
    lines.append(markdown_table(report["final_vs_2024_numeric"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Final Synthetic Numeric Drift vs 2024 Adjusted\n")
    lines.append(f"Excluded numeric columns: `{report.get('excluded_numeric_columns', [])}`\n\n")
    lines.append(markdown_table(report["final_vs_2024_numeric_adjusted"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Final Synthetic Numeric Drift vs 2025\n")
    lines.append(markdown_table(report["final_vs_2025_numeric"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Final Synthetic Numeric Drift vs 2025 Adjusted\n")
    lines.append(f"Excluded numeric columns: `{report.get('excluded_numeric_columns', [])}`\n\n")
    lines.append(markdown_table(report["final_vs_2025_numeric_adjusted"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Stage1 Numeric Drift vs 2024\n")
    lines.append(markdown_table(report["stage1_vs_2024_numeric"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Stage2 Numeric Drift vs 2024\n")
    lines.append(markdown_table(report["stage2_vs_2024_numeric"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Stage2 Numeric Drift vs 2024 Adjusted\n")
    lines.append(f"Excluded numeric columns: `{report.get('excluded_numeric_columns', [])}`\n\n")
    lines.append(markdown_table(report["stage2_vs_2024_numeric_adjusted"], ["column", "wasserstein", "syn_mean", "real_mean", "mean_delta", "syn_min", "syn_max", "real_min", "real_max"]))
    lines.append("\n## Final Categorical Drift vs 2024\n")
    lines.append(markdown_table(report["final_vs_2024_categorical"], ["column", "js"], max_rows=25))
    lines.append("\n## Coordinate Support\n")
    lines.append(markdown_table(report["coordinate_support"], ["name", "nyc_bbox_rate", "nearest_real_deg_mean", "nearest_real_deg_p95", "nearest_real_deg_p99"]))
    lines.append("\n## Rule Violations\n")
    lines.append(markdown_table(report["rule_violations"], ["name", "negative_prcp_rate", "negative_wind_rate", "invalid_speed_rate", "invalid_lanes_rate", "negative_signal_dist_rate", "HAS_DIVIDER_nunique"]))
    lines.append("\n## Nearest-Coordinate Stage2 Consistency\n")
    lines.append(markdown_table(report["nearest_coordinate_consistency"], ["name", "column", "nearest_coord_deg_mean", "mae_to_nearest_real", "mean_error", "p95_abs_error"]))
    lines.append("\n## Mask Summary\n")
    for key, value in report["mask_summary"].items():
        lines.append(f"- `{key}`: `{value}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--exclude_numeric",
        nargs="*",
        default=["REAL_SPEED_LIMIT"],
        help="Numeric columns excluded from adjusted drift summaries.",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_json = args.out_json or root / "results" / "stage2_chain_diagnostics.json"
    out_md = args.out_md or root / "results" / "stage2_chain_diagnostics.md"

    info = read_json(root / "data" / "nyc_crash_2024" / "info.json")
    num_cols = list(info.get("num_col_names", DEFAULT_NUMERIC))
    adjusted_num_cols = [c for c in num_cols if c not in set(args.exclude_numeric)]
    cat_cols = list(info.get("cat_col_names", []))

    real24 = load_csv(root / "data" / "nyc_crash_2024" / "test.csv")
    real25 = load_csv(root / "data" / "nyc_crash_2025" / "test.csv")
    final_syn = load_csv(root / "results" / "synthetic_2024_stage2_causal_full" / "ours_stage2_causal_full.csv")
    stage1_syn = load_csv(root / "results" / "synthetic_2024_stage2_causal_full" / "_ours_stage2_causal_full_samples_stage1_chain.csv")
    stage2_syn = load_csv(root / "results" / "synthetic_2024_stage2_causal_full" / "_ours_stage2_causal_full_samples_stage2_chain.csv")

    final24_num = compare_numeric(final_syn, real24, num_cols)
    final25_num = compare_numeric(final_syn, real25, num_cols)
    final24_num_adjusted = compare_numeric(final_syn, real24, adjusted_num_cols)
    final25_num_adjusted = compare_numeric(final_syn, real25, adjusted_num_cols)
    stage1_num = compare_numeric(stage1_syn, real24, STAGE1_NUMERIC)
    stage2_num = compare_numeric(stage2_syn, real24, STAGE1_NUMERIC + STAGE2_NUMERIC)
    stage2_num_adjusted = compare_numeric(
        stage2_syn,
        real24,
        [c for c in STAGE1_NUMERIC + STAGE2_NUMERIC if c not in set(args.exclude_numeric)],
    )
    final24_cat = compare_categorical(final_syn, real24, cat_cols)
    stage2_cat = compare_categorical(stage2_syn, real24, STAGE1_CATEGORICAL + STAGE2_CATEGORICAL)

    report: dict[str, Any] = {
        "paths": {
            "root": str(root),
            "final_synthetic": "results/synthetic_2024_stage2_causal_full/ours_stage2_causal_full.csv",
            "stage1_chain": "results/synthetic_2024_stage2_causal_full/_ours_stage2_causal_full_samples_stage1_chain.csv",
            "stage2_chain": "results/synthetic_2024_stage2_causal_full/_ours_stage2_causal_full_samples_stage2_chain.csv",
        },
        "final_vs_2024_numeric": final24_num,
        "final_vs_2025_numeric": final25_num,
        "excluded_numeric_columns": list(args.exclude_numeric),
        "final_vs_2024_numeric_adjusted": final24_num_adjusted,
        "final_vs_2025_numeric_adjusted": final25_num_adjusted,
        "stage1_vs_2024_numeric": stage1_num,
        "stage2_vs_2024_numeric": stage2_num,
        "stage2_vs_2024_numeric_adjusted": stage2_num_adjusted,
        "final_vs_2024_categorical": final24_cat,
        "stage2_vs_2024_categorical": stage2_cat,
        "coordinate_support": [
            coordinate_support(stage1_syn, real24, "stage1_vs_2024"),
            coordinate_support(stage2_syn, real24, "stage2_vs_2024"),
            coordinate_support(final_syn, real24, "final_vs_2024"),
            coordinate_support(final_syn, real25, "final_vs_2025"),
        ],
        "rule_violations": [
            rule_violations(stage2_syn, "stage2_chain"),
            rule_violations(final_syn, "final_synthetic"),
            rule_violations(real24, "real_2024_test"),
            rule_violations(real25, "real_2025_test"),
        ],
        "nearest_coordinate_consistency": nearest_coordinate_feature_error(
            stage2_syn,
            real24,
            "stage2_chain_vs_real24_nearest_coord",
            ["DIST_TO_SIGNAL_M", "REAL_SPEED_LIMIT", "INFERRED_LANES", "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "OSM_TYPE"],
        ),
        "mask_summary": {
            "stage2": mask_summary(root / "data" / "nyc_stage2_2024" / "causal_masks", root / "data" / "nyc_stage2_2024" / "info.json"),
            "full": mask_summary(root / "data" / "nyc_crash_2024" / "causal_masks", root / "data" / "nyc_crash_2024" / "info.json"),
        },
    }

    top_final24 = final24_num[0] if final24_num else {}
    top_final25 = final25_num[0] if final25_num else {}
    top_final24_adjusted = final24_num_adjusted[0] if final24_num_adjusted else {}
    top_final25_adjusted = final25_num_adjusted[0] if final25_num_adjusted else {}
    top_stage2 = stage2_num[0] if stage2_num else {}
    top_stage2_adjusted = stage2_num_adjusted[0] if stage2_num_adjusted else {}
    stage1_max = max((row["wasserstein"] for row in stage1_num), default=float("nan"))
    stage2_max = max((row["wasserstein"] for row in stage2_num), default=float("nan"))
    final_cat_top = final24_cat[0] if final24_cat else {}
    has_divider = next((x for x in report["rule_violations"] if x["name"] == "final_synthetic"), {})
    real25_speed = numeric_series(real25, "REAL_SPEED_LIMIT")
    real25_speed_note = "REAL_SPEED_LIMIT in real25 is not constant."
    if len(real25_speed) > 0 and real25_speed.nunique(dropna=False) == 1:
        real25_speed_note = f"REAL_SPEED_LIMIT in real25 is constant at {float(real25_speed.iloc[0]):.6f}; transfer W-num for this field is not comparable until data is rebuilt."
    nearest_top = report["nearest_coordinate_consistency"][0] if report["nearest_coordinate_consistency"] else {}
    report["key_findings"] = [
        f"Final vs 2024 top numeric drift: {top_final24.get('column')} W={top_final24.get('wasserstein'):.6f}.",
        f"Final vs 2025 top numeric drift: {top_final25.get('column')} W={top_final25.get('wasserstein'):.6f}.",
        f"Adjusted top numeric drift after excluding {list(args.exclude_numeric)}: 2024 {top_final24_adjusted.get('column')} W={top_final24_adjusted.get('wasserstein'):.6f}; 2025 {top_final25_adjusted.get('column')} W={top_final25_adjusted.get('wasserstein'):.6f}.",
        real25_speed_note,
        f"Stage1 max numeric W vs 2024 is {stage1_max:.6f}; Stage2 max numeric W vs 2024 is {stage2_max:.6f}, top Stage2 column is {top_stage2.get('column')}.",
        f"Adjusted Stage2 top drift after excluding {list(args.exclude_numeric)}: {top_stage2_adjusted.get('column')} W={top_stage2_adjusted.get('wasserstein'):.6f}.",
        f"Nearest-coordinate Stage2 consistency top error: {nearest_top.get('column')} MAE={nearest_top.get('mae_to_nearest_real'):.6f}.",
        f"Top categorical JS vs 2024: {final_cat_top.get('column')} JS={final_cat_top.get('js'):.6f}.",
        f"HAS_DIVIDER unique count in final synthetic: {has_divider.get('HAS_DIVIDER_nunique')}.",
    ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, out_md)
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_md}")
    for item in report["key_findings"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()