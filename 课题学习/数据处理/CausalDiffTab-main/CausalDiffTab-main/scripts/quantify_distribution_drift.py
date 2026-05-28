#!/usr/bin/env python
"""
Quantify distribution drift between 2024 and 2025 NYC crash data.

Outputs:
- Per-feature Wasserstein distance (numeric) / JS divergence (categorical)
- Covariate shift summary
- Concept drift (P(Y|X) difference)
- Macro relation drift (group-wise injury rate difference)
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return pd.concat([train, test], ignore_index=True)


def compute_numeric_drift(df_a, df_b, col):
    """Compute Wasserstein-1 and KS statistic for a numeric column."""
    a = df_a[col].dropna().values
    b = df_b[col].dropna().values
    if len(a) == 0 or len(b) == 0:
        return {"wasserstein": None, "ks_statistic": None, "ks_pvalue": None}
    wass = wasserstein_distance(a, b)
    ks = ks_2samp(a, b)
    # Normalize Wasserstein by pooled std
    pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    wass_norm = wass / (pooled_std + 1e-12)
    return {
        "wasserstein": round(float(wass), 6),
        "wasserstein_normalized": round(float(wass_norm), 6),
        "ks_statistic": round(float(ks.statistic), 6),
        "ks_pvalue": round(float(ks.pvalue), 6) if not math.isnan(ks.pvalue) else None,
        "mean_a": round(float(np.mean(a)), 6),
        "mean_b": round(float(np.mean(b)), 6),
        "std_a": round(float(np.std(a)), 6),
        "std_b": round(float(np.std(b)), 6),
    }


def compute_categorical_drift(df_a, df_b, col):
    """Compute JS divergence for a categorical column."""
    cats = sorted(set(df_a[col].dropna().unique()) | set(df_b[col].dropna().unique()))
    counts_a = df_a[col].value_counts()
    counts_b = df_b[col].value_counts()
    p = np.array([counts_a.get(c, 0) for c in cats], dtype=float)
    q = np.array([counts_b.get(c, 0) for c in cats], dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    js = jensenshannon(p, q)
    if math.isnan(js):
        js = 0.0 if np.allclose(p, q) else 1.0
    return {
        "js_divergence": round(float(js), 6),
        "n_categories": len(cats),
        "categories": [str(c) for c in cats[:20]],  # limit output
    }


def compute_target_concept_drift(df_a, df_b, target_col):
    """Compute overall and group-wise concept drift for the target variable."""
    y_a = df_a[target_col].values
    y_b = df_b[target_col].values
    
    result = {
        "overall_mean_a": round(float(np.mean(y_a)), 6),
        "overall_mean_b": round(float(np.mean(y_b)), 6),
        "overall_abs_diff": round(float(abs(np.mean(y_a) - np.mean(y_b))), 6),
        "wasserstein": round(float(wasserstein_distance(y_a, y_b)), 6),
    }
    return result


def compute_macro_relation_drift(df_a, df_b, group_cols, target_col):
    """Compute group-wise injury rate drift."""
    def group_means(df):
        return df.groupby(group_cols)[target_col].agg(['mean', 'count']).reset_index()
    
    gm_a = group_means(df_a)
    gm_b = group_means(df_b)
    
    merged = gm_a.merge(gm_b, on=group_cols, how="outer", suffixes=("_a", "_b"))
    merged["mean_a"] = merged["mean_a"].fillna(0)
    merged["mean_b"] = merged["mean_b"].fillna(0)
    merged["abs_diff"] = (merged["mean_a"] - merged["mean_b"]).abs()
    
    return {
        "n_groups_a": int(gm_a.shape[0]),
        "n_groups_b": int(gm_b.shape[0]),
        "n_common_groups": int(merged.dropna(subset=["count_a", "count_b"]).shape[0]),
        "mean_abs_diff": round(float(merged["abs_diff"].mean()), 6),
        "max_abs_diff": round(float(merged["abs_diff"].max()), 6),
        "top_drift_groups": merged.nlargest(10, "abs_diff")[[*group_cols, "mean_a", "mean_b", "abs_diff"]].to_dict(orient="records"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantify distribution drift between two datasets")
    parser.add_argument("--data_a", type=str, default="data/nyc_crash_2024_v2", help="First dataset dir")
    parser.add_argument("--data_b", type=str, default="data/nyc_crash_2025_v2", help="Second dataset dir")
    parser.add_argument("--output", type=str, default="results/drift_report.json", help="Output JSON path")
    parser.add_argument("--target", type=str, default="NUMBER OF PERSONS INJURED", help="Target column")
    args = parser.parse_args()
    
    print(f"[drift] Loading {args.data_a} ...")
    df_a = load_data(args.data_a)
    print(f"[drift] Loading {args.data_b} ...")
    df_b = load_data(args.data_b)
    
    print(f"[drift] A: {len(df_a)} rows, B: {len(df_b)} rows")
    
    common_cols = [c for c in df_a.columns if c in df_b.columns and c != args.target]
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c])]
    cat_cols = [c for c in common_cols if c not in numeric_cols]
    
    report = {
        "data_a": args.data_a,
        "data_b": args.data_b,
        "n_rows_a": len(df_a),
        "n_rows_b": len(df_b),
        "target": args.target,
        "numeric_features": {},
        "categorical_features": {},
        "concept_drift": {},
        "macro_relation_drift": {},
    }
    
    # Numeric drift
    print(f"[drift] Computing numeric drift for {len(numeric_cols)} features ...")
    for col in numeric_cols:
        report["numeric_features"][col] = compute_numeric_drift(df_a, df_b, col)
    
    # Categorical drift
    print(f"[drift] Computing categorical drift for {len(cat_cols)} features ...")
    for col in cat_cols:
        report["categorical_features"][col] = compute_categorical_drift(df_a, df_b, col)
    
    # Target concept drift
    if args.target in df_a.columns and args.target in df_b.columns:
        report["concept_drift"] = compute_target_concept_drift(df_a, df_b, args.target)
    
    # Macro relation drift
    group_cols = ["SEASON", "WEATHER_CONDITION", "OSM_TYPE"]
    available_group_cols = [c for c in group_cols if c in df_a.columns and c in df_b.columns]
    if available_group_cols:
        report["macro_relation_drift"] = compute_macro_relation_drift(df_a, df_b, available_group_cols, args.target)
    
    # Summary stats
    num_wass = [v["wasserstein_normalized"] for v in report["numeric_features"].values() if v["wasserstein_normalized"] is not None]
    cat_js = [v["js_divergence"] for v in report["categorical_features"].values() if v["js_divergence"] is not None]
    
    report["summary"] = {
        "mean_numeric_wasserstein_normalized": round(float(np.mean(num_wass)), 6) if num_wass else None,
        "max_numeric_wasserstein_normalized": round(float(np.max(num_wass)), 6) if num_wass else None,
        "mean_categorical_js_divergence": round(float(np.mean(cat_js)), 6) if cat_js else None,
        "max_categorical_js_divergence": round(float(np.max(cat_js)), 6) if cat_js else None,
    }
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[drift] Report saved to {args.output}")
    print(f"[drift] Mean numeric Wass (norm): {report['summary']['mean_numeric_wasserstein_normalized']}")
    print(f"[drift] Mean categorical JS: {report['summary']['mean_categorical_js_divergence']}")
    print(f"[drift] Concept drift (target mean diff): {report['concept_drift'].get('overall_abs_diff', 'N/A')}")


if __name__ == "__main__":
    main()
