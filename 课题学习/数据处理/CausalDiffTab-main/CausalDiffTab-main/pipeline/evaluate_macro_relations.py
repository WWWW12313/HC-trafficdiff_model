"""Evaluate macro causal relation matching for crash synthetic data.

This diagnostic focuses on paper-facing traffic mechanisms such as
weather/road/vehicle/crash_type -> injury_outcome. It compares real and
synthetic data using group-wise injury mean differences and a practical CMI
proxy for each parent feature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

CDT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RelationSpec:
    relation: str
    parent_col: str
    kind: str
    cond_cols: tuple[str, ...] = ()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path: str | None, default: Path | None = None) -> Path:
    p = Path(path) if path else default
    if p is None:
        raise ValueError("path is required")
    if not p.is_absolute():
        p = CDT_ROOT / p
    return p


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"target column missing: {target_col}")
    return _to_numeric(df[target_col]).clip(lower=0)


def _encode_parent(real: pd.Series, syn: pd.Series, kind: str, n_bins: int = 5) -> tuple[pd.Series, pd.Series]:
    if kind == "numeric_bin":
        r_num = _to_numeric(real)
        s_num = _to_numeric(syn)
        finite = r_num[np.isfinite(r_num)]
        if finite.nunique(dropna=True) <= 1:
            return pd.Series("all", index=real.index), pd.Series("all", index=syn.index)
        try:
            _, bins = pd.qcut(finite, q=min(n_bins, finite.nunique()), retbins=True, duplicates="drop")
        except ValueError:
            bins = np.linspace(float(finite.min()), float(finite.max()), min(n_bins, len(finite)) + 1)
        bins = np.unique(bins)
        if len(bins) < 2:
            return pd.Series("all", index=real.index), pd.Series("all", index=syn.index)
        bins[0] = -np.inf
        bins[-1] = np.inf
        return (
            pd.cut(r_num, bins=bins, include_lowest=True).astype(str).fillna("_nan_"),
            pd.cut(s_num, bins=bins, include_lowest=True).astype(str).fillna("_nan_"),
        )
    return real.astype(str).fillna("_nan_"), syn.astype(str).fillna("_nan_")


def _weighted_group_mean_error(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    spec: RelationSpec,
    target_col: str,
) -> dict[str, Any]:
    if spec.parent_col not in real_df.columns or spec.parent_col not in syn_df.columns:
        return {
            "relation": spec.relation,
            "parent_col": spec.parent_col,
            "kind": spec.kind,
            "status": "missing_column",
        }

    real_parent, syn_parent = _encode_parent(real_df[spec.parent_col], syn_df[spec.parent_col], spec.kind)
    real_target = _safe_target(real_df, target_col)
    syn_target = _safe_target(syn_df, target_col)

    real_tmp = pd.DataFrame({"parent": real_parent, "target": real_target}).dropna()
    syn_tmp = pd.DataFrame({"parent": syn_parent, "target": syn_target}).dropna()
    if len(real_tmp) < 32 or len(syn_tmp) < 32:
        return {
            "relation": spec.relation,
            "parent_col": spec.parent_col,
            "kind": spec.kind,
            "status": "too_few_rows",
        }

    real_mean = real_tmp.groupby("parent")["target"].mean()
    syn_mean = syn_tmp.groupby("parent")["target"].mean()
    real_freq = real_tmp["parent"].value_counts(normalize=True)
    groups = sorted(set(real_mean.index) | set(syn_mean.index))

    rows = []
    weighted_abs = 0.0
    weight_sum = 0.0
    max_abs = 0.0
    for group in groups:
        r_val = float(real_mean[group]) if group in real_mean.index else float("nan")
        s_val = float(syn_mean[group]) if group in syn_mean.index else float("nan")
        weight = float(real_freq[group]) if group in real_freq.index else 0.0
        abs_err = float("nan") if np.isnan(r_val) or np.isnan(s_val) else abs(r_val - s_val)
        if np.isfinite(abs_err):
            weighted_abs += weight * abs_err
            weight_sum += weight
            max_abs = max(max_abs, abs_err)
        rows.append({
            "group": str(group),
            "real_mean_injury": round(r_val, 6) if np.isfinite(r_val) else None,
            "syn_mean_injury": round(s_val, 6) if np.isfinite(s_val) else None,
            "real_weight": round(weight, 6),
            "abs_error": round(abs_err, 6) if np.isfinite(abs_err) else None,
        })

    return {
        "relation": spec.relation,
        "parent_col": spec.parent_col,
        "kind": spec.kind,
        "status": "ok",
        "n_groups_real": int(real_mean.shape[0]),
        "n_groups_syn": int(syn_mean.shape[0]),
        "weighted_mean_abs_error": round(float(weighted_abs / max(weight_sum, 1e-12)), 6),
        "max_abs_error": round(float(max_abs), 6),
        "group_details": rows,
    }


def _design_matrix(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[use_cols].copy(), dummy_na=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _estimate_cmi_proxy(df: pd.DataFrame, spec: RelationSpec, target_col: str, random_state: int) -> float:
    needed = [spec.parent_col, target_col] + [c for c in spec.cond_cols if c in df.columns]
    if spec.parent_col not in df.columns or target_col not in df.columns:
        return float("nan")
    sub = df[needed].copy().dropna()
    if len(sub) < 64:
        return float("nan")

    parent = sub[spec.parent_col]
    if spec.kind == "numeric_bin":
        x = _to_numeric(parent).fillna(_to_numeric(parent).median()).to_numpy(dtype=np.float64)
    else:
        x = pd.factorize(parent.astype(str), sort=True)[0].astype(np.float64)
    y = _to_numeric(sub[target_col]).fillna(0.0).to_numpy(dtype=np.float64)

    cond = _design_matrix(sub, spec.cond_cols)
    if cond.shape[1] > 0:
        model_x = RandomForestRegressor(n_estimators=80, random_state=random_state, n_jobs=-1, min_samples_leaf=10)
        model_y = RandomForestRegressor(n_estimators=80, random_state=random_state, n_jobs=-1, min_samples_leaf=10)
        z = cond.to_numpy(dtype=np.float64)
        model_x.fit(z, x)
        model_y.fit(z, y)
        x = x - model_x.predict(z)
        y = y - model_y.predict(z)
    else:
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    x = np.where(np.isfinite(x), x, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(mutual_info_regression(x.reshape(-1, 1), y, random_state=random_state)[0])


def _cmi_error(real_df: pd.DataFrame, syn_df: pd.DataFrame, spec: RelationSpec, target_col: str) -> dict[str, Any]:
    real_cmi = _estimate_cmi_proxy(real_df, spec, target_col, random_state=42)
    syn_cmi = _estimate_cmi_proxy(syn_df, spec, target_col, random_state=42)
    abs_err = float("nan") if np.isnan(real_cmi) or np.isnan(syn_cmi) else abs(real_cmi - syn_cmi)
    rel_err = float("nan") if np.isnan(abs_err) else abs_err / (abs(real_cmi) + 1e-8)
    return {
        "real_cmi": round(real_cmi, 6) if np.isfinite(real_cmi) else None,
        "syn_cmi": round(syn_cmi, 6) if np.isfinite(syn_cmi) else None,
        "cmi_abs_error": round(abs_err, 6) if np.isfinite(abs_err) else None,
        "cmi_rel_error": round(rel_err, 6) if np.isfinite(rel_err) else None,
        "cond_cols": list(spec.cond_cols),
    }


def _build_specs(groups: dict) -> list[RelationSpec]:
    specs: list[RelationSpec] = []
    weather_cond = ("SEASON", "CRASH_TIME_SIN", "CRASH_TIME_COS")
    for col, kind in [
        ("WEATHER_CONDITION", "categorical"),
        ("TEMP_C", "numeric_bin"),
        ("prcp", "numeric_bin"),
        ("WIND_SPEED_KMH", "numeric_bin"),
    ]:
        specs.append(RelationSpec("weather_to_injury", col, kind, weather_cond))

    road_cond = ("LATITUDE", "LONGITUDE")
    for col, kind in [
        ("HAS_TRAFFIC_SIGNAL", "categorical"),
        ("OSM_ONEWAY", "categorical"),
        ("OSM_TYPE", "categorical"),
        ("DIST_TO_SIGNAL_M", "numeric_bin"),
        ("INFERRED_LANES", "numeric_bin"),
    ]:
        specs.append(RelationSpec("road_to_injury", col, kind, road_cond))

    for col in groups.get("vehicle_binary", []):
        specs.append(RelationSpec("vehicle_to_injury", col, "categorical", ("TOTAL_VEHICLES",)))

    crash_cond = ("WEATHER_CONDITION", "OSM_TYPE", "TOTAL_VEHICLES", "IS_MULTI_VEHICLE")
    for col in groups.get("factor_binary", []):
        specs.append(RelationSpec("crash_type_to_injury", col, "categorical", crash_cond))
    return specs


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for relation in sorted({r["relation"] for r in rows}):
        subset = [r for r in rows if r["relation"] == relation and r.get("status") == "ok"]
        if not subset:
            continue
        mean_err = [r.get("weighted_mean_abs_error") for r in subset]
        cmi_err = [r.get("cmi_abs_error") for r in subset]
        out.append({
            "relation": relation,
            "n_specs": len(subset),
            "mean_group_mae": round(float(np.nanmean(mean_err)), 6),
            "max_group_mae": round(float(np.nanmax(mean_err)), 6),
            "mean_cmi_abs_error": round(float(np.nanmean(cmi_err)), 6),
            "max_cmi_abs_error": round(float(np.nanmax(cmi_err)), 6),
        })
    return out


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [f"# Macro Relation Report: {payload['file']}", ""]
    lines.append("## Summary")
    lines.append("| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["summary"]:
        lines.append(
            f"| {row['relation']} | {row['n_specs']} | {row['mean_group_mae']} | {row['max_group_mae']} | "
            f"{row['mean_cmi_abs_error']} | {row['max_cmi_abs_error']} |"
        )
    lines.append("")
    lines.append("## Per Feature")
    lines.append("| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload["rows"]:
        if row.get("status") != "ok":
            continue
        lines.append(
            f"| {row['relation']} | {row['parent_col']} | {row['weighted_mean_abs_error']} | "
            f"{row.get('cmi_abs_error')} | {row.get('real_cmi')} | {row.get('syn_cmi')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate macro edge-wise relation matching")
    parser.add_argument("--real", required=True, help="Real test CSV")
    parser.add_argument("--synthetic", required=True, help="Synthetic CSV")
    parser.add_argument("--groups_json", default="data/processed/column_groups.json")
    parser.add_argument("--target_col", default="NUMBER OF PERSONS INJURED")
    parser.add_argument("--output_tag", default=None)
    args = parser.parse_args()

    real_path = _resolve_path(args.real)
    syn_path = _resolve_path(args.synthetic)
    groups_path = _resolve_path(args.groups_json)

    real_df = pd.read_csv(real_path, low_memory=False)
    syn_df = pd.read_csv(syn_path, low_memory=False)
    groups = _load_json(groups_path)
    specs = _build_specs(groups)

    rows = []
    for spec in specs:
        row = _weighted_group_mean_error(real_df, syn_df, spec, args.target_col)
        if row.get("status") == "ok":
            row.update(_cmi_error(real_df, syn_df, spec, args.target_col))
        rows.append(row)

    payload = {
        "file": syn_path.name,
        "real": str(real_path),
        "synthetic": str(syn_path),
        "target_col": args.target_col,
        "rows": rows,
        "summary": _summarize(rows),
    }

    tag = args.output_tag or syn_path.stem
    out_json = CDT_ROOT / "results" / f"macro_relation_report_{tag}.json"
    out_md = CDT_ROOT / "results" / f"macro_relation_report_{tag}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    out_md.write_text(_markdown_report(payload), encoding="utf-8")

    print(f"[write] {out_json}")
    print(f"[write] {out_md}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()