"""Post-hoc macro relation calibration for injury targets.

The calibrator learns parent-group injury deviations on real training data and
adjusts synthetic NUMBER OF PERSONS INJURED so macro relations such as
weather/road/vehicle/crash_type -> injury_outcome better match the source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_macro_relations import RelationSpec, _build_specs, _encode_parent, _load_json, _resolve_path

CDT_ROOT = Path(__file__).resolve().parent.parent


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fit_group_deviations(real_df: pd.DataFrame, specs: list[RelationSpec], target_col: str) -> tuple[float, dict[str, dict[str, float]]]:
    target = _to_numeric(real_df[target_col]).clip(lower=0)
    global_mean = float(target.mean())
    table: dict[str, dict[str, float]] = {}
    for spec in specs:
        if spec.parent_col not in real_df.columns:
            continue
        encoded, _ = _encode_parent(real_df[spec.parent_col], real_df[spec.parent_col], spec.kind)
        tmp = pd.DataFrame({"parent": encoded, "target": target}).dropna()
        if len(tmp) < 32:
            continue
        means = tmp.groupby("parent")["target"].mean()
        counts = tmp["parent"].value_counts()
        values = {}
        for group, mean_val in means.items():
            if int(counts.get(group, 0)) >= 20:
                values[str(group)] = float(mean_val - global_mean)
        if values:
            table[f"{spec.relation}:{spec.parent_col}"] = values
    return global_mean, table


def _fit_synthetic_deviations(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    specs: list[RelationSpec],
    target_col: str,
) -> tuple[float, dict[str, dict[str, float]], dict[str, pd.Series]]:
    target = _to_numeric(syn_df[target_col]).clip(lower=0)
    global_mean = float(target.mean())
    table: dict[str, dict[str, float]] = {}
    encoded_syn: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.parent_col not in syn_df.columns or spec.parent_col not in real_df.columns:
            continue
        _, syn_encoded = _encode_parent(real_df[spec.parent_col], syn_df[spec.parent_col], spec.kind)
        encoded_syn[f"{spec.relation}:{spec.parent_col}"] = syn_encoded
        tmp = pd.DataFrame({"parent": syn_encoded, "target": target}).dropna()
        if len(tmp) < 32:
            continue
        means = tmp.groupby("parent")["target"].mean()
        table[f"{spec.relation}:{spec.parent_col}"] = {str(k): float(v - global_mean) for k, v in means.items()}
    return global_mean, table, encoded_syn


def calibrate(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    groups: dict[str, Any],
    target_col: str,
    shrink: float,
    use_relations: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = [s for s in _build_specs(groups) if s.relation in use_relations]
    real_global, real_dev = _fit_group_deviations(real_df, specs, target_col)
    syn_global, syn_dev, encoded_syn = _fit_synthetic_deviations(syn_df, real_df, specs, target_col)

    y = _to_numeric(syn_df[target_col]).clip(lower=0).fillna(syn_global).to_numpy(dtype=np.float64)
    global_correction = np.full(len(syn_df), real_global - syn_global, dtype=np.float64)
    relation_correction = np.zeros(len(syn_df), dtype=np.float64)
    relation_counts = np.zeros(len(syn_df), dtype=np.float64)

    used_specs = []
    for key, real_values in real_dev.items():
        if key not in encoded_syn:
            continue
        syn_values = syn_dev.get(key, {})
        groups_encoded = encoded_syn[key].astype(str)
        delta = groups_encoded.map(lambda group: real_values.get(group, 0.0) - syn_values.get(group, 0.0)).to_numpy(dtype=np.float64)
        relation_correction += np.where(np.isfinite(delta), delta, 0.0)
        relation_counts += 1.0
        used_specs.append(key)

    relation_mean = relation_correction / np.maximum(relation_counts, 1.0)
    y_cal = y + float(shrink) * (global_correction + relation_mean)
    y_cal = np.clip(y_cal, 0.0, max(20.0, float(np.nanmax(_to_numeric(real_df[target_col]).fillna(0.0)))))

    out = syn_df.copy()
    out[target_col] = y_cal
    metadata = {
        "target_col": target_col,
        "real_global_mean": round(real_global, 6),
        "synthetic_global_mean_before": round(syn_global, 6),
        "synthetic_global_mean_after": round(float(np.mean(y_cal)), 6),
        "shrink": shrink,
        "n_specs_used": len(used_specs),
        "specs_used": used_specs,
    }
    return out, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate synthetic injury target using macro relation means")
    parser.add_argument("--real_train", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups_json", default="data/processed/column_groups.json")
    parser.add_argument("--target_col", default="NUMBER OF PERSONS INJURED")
    parser.add_argument("--shrink", type=float, default=0.8)
    parser.add_argument(
        "--relations",
        default="weather_to_injury,road_to_injury,vehicle_to_injury,crash_type_to_injury",
        help="Comma-separated relation groups to use",
    )
    args = parser.parse_args()

    real_path = _resolve_path(args.real_train)
    syn_path = _resolve_path(args.synthetic)
    out_path = _resolve_path(args.output)
    groups = _load_json(_resolve_path(args.groups_json))

    real_df = pd.read_csv(real_path, low_memory=False)
    syn_df = pd.read_csv(syn_path, low_memory=False)
    use_relations = {x.strip() for x in args.relations.split(",") if x.strip()}
    out_df, metadata = calibrate(real_df, syn_df, groups, args.target_col, args.shrink, use_relations)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    meta_path = out_path.with_suffix(".relation_calibration.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[write] {out_path}")
    print(f"[write] {meta_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()