from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.postprocess_samples import _causal_weather_resample  # noqa: E402
from src.road_snap import enrich_road_context  # noqa: E402


ROAD_CONTEXT_COLUMNS = {
    "DIST_TO_SIGNAL_M",
    "HAS_TRAFFIC_SIGNAL",
    "OSM_TYPE",
    "OSM_ONEWAY",
    "INFERRED_LANES",
    "REAL_SPEED_LIMIT",
    "HAS_DIVIDER",
}


def _norm_value(value) -> str:
    if pd.isna(value):
        return "__NA__"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "__NA__"
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    return text if text else "__NA__"


def _load_label_mappings(reference_csv: Path) -> dict:
    info_path = reference_csv.parent / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("cat_label_mappings", {})


def _label_value(value, col_name: str, label_mappings: dict) -> str:
    raw = _norm_value(value)
    mapping = label_mappings.get(col_name, {})
    return _norm_value(mapping.get(raw, mapping.get(str(value), value)))


def _row_key(row: pd.Series, cols: list[str], label_mappings: dict) -> tuple[str, ...]:
    return tuple(_label_value(row[col], col, label_mappings) for col in cols)


def _cell_for_row(row: pd.Series, resolution: int) -> str | None:
    lat = pd.to_numeric(row.get("LATITUDE"), errors="coerce")
    lon = pd.to_numeric(row.get("LONGITUDE"), errors="coerce")
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    try:
        return h3.latlng_to_cell(float(lat), float(lon), resolution)
    except Exception:
        return None


def _build_groups(ref: pd.DataFrame, condition_cols: list[str], label_mappings: dict) -> tuple[dict, dict, dict]:
    cell_condition: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    cell_only: dict[str, np.ndarray] = {}
    condition_only: dict[tuple[str, ...], np.ndarray] = {}

    if condition_cols:
        condition_keys = ref.apply(lambda row: _row_key(row, condition_cols, label_mappings), axis=1)
    else:
        condition_keys = pd.Series([tuple()] * len(ref), index=ref.index)

    for (cell, key), idx in ref.groupby(["ROAD_H3_CELL", condition_keys], dropna=False).groups.items():
        cell_condition[(str(cell), tuple(key))] = np.asarray(list(idx), dtype=int)
    for cell, idx in ref.groupby("ROAD_H3_CELL", dropna=False).groups.items():
        cell_only[str(cell)] = np.asarray(list(idx), dtype=int)
    for key, idx in condition_keys.groupby(condition_keys).groups.items():
        condition_only[tuple(key)] = np.asarray(list(idx), dtype=int)
    return cell_condition, cell_only, condition_only


def apply_h3_projection(
    df: pd.DataFrame,
    reference_csv: Path,
    resolution: int,
    condition_cols: list[str],
    max_ring: int,
    min_bucket_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    usecols = ["LATITUDE", "LONGITUDE"] + [col for col in condition_cols if col in df.columns]
    ref = pd.read_csv(reference_csv, usecols=lambda col: col in set(usecols), low_memory=False)
    ref = ref.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    ref["ROAD_H3_CELL"] = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(ref["LATITUDE"], ref["LONGITUDE"])
    ]

    label_mappings = _load_label_mappings(reference_csv)
    active_conditions = [col for col in condition_cols if col in df.columns and col in ref.columns]
    cell_condition, cell_only, condition_only = _build_groups(ref, active_conditions, label_mappings)
    global_pool = ref.index.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    stats = {
        "enabled": True,
        "reference_csv": str(reference_csv),
        "resolution": resolution,
        "condition_cols": active_conditions,
        "max_ring": max_ring,
        "min_bucket_size": min_bucket_size,
        "same_cell_condition": 0,
        "neighbor_cell_condition": 0,
        "same_cell": 0,
        "condition_only": 0,
        "global": 0,
        "invalid_cell": 0,
        "n_reference_cells": int(ref["ROAD_H3_CELL"].nunique()),
    }

    out = df.copy()
    original_cells: list[str | None] = []
    chosen_indices: list[int] = []

    for _, row in out.iterrows():
        cell = _cell_for_row(row, resolution)
        original_cells.append(cell)
        key = _row_key(row, active_conditions, label_mappings) if active_conditions else tuple()

        pool = cell_condition.get((cell, key)) if cell is not None else None
        source = "same_cell_condition"

        if pool is None or len(pool) < min_bucket_size:
            pool = None
            if cell is not None:
                for ring in range(1, max_ring + 1):
                    candidates: list[int] = []
                    for neighbor in h3.grid_disk(cell, ring):
                        arr = cell_condition.get((neighbor, key))
                        if arr is not None:
                            candidates.extend(arr.tolist())
                    if len(candidates) >= min_bucket_size:
                        pool = np.asarray(candidates, dtype=int)
                        source = "neighbor_cell_condition"
                        break

        if pool is None or len(pool) == 0:
            pool = cell_only.get(cell) if cell is not None else None
            source = "same_cell"
        if pool is None or len(pool) == 0:
            pool = condition_only.get(key)
            source = "condition_only"
        if pool is None or len(pool) == 0:
            pool = global_pool
            source = "global"
        if cell is None:
            stats["invalid_cell"] += 1

        chosen_indices.append(int(rng.choice(pool)))
        stats[source] += 1

    sampled = ref.iloc[chosen_indices].reset_index(drop=True)
    out["ROAD_H3_CELL"] = [cell or "__INVALID__" for cell in original_cells]
    out["ROAD_H3_ANCHOR_CELL"] = sampled["ROAD_H3_CELL"].to_numpy()
    out["LATITUDE"] = sampled["LATITUDE"].to_numpy(dtype=float)
    out["LONGITUDE"] = sampled["LONGITUDE"].to_numpy(dtype=float)
    return out, stats


def run_one(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.input_csv, low_memory=False)
    condition_cols = [col.strip() for col in args.condition_cols.split(",") if col.strip()]
    out, h3_stats = apply_h3_projection(
        df,
        reference_csv=Path(args.reference_csv),
        resolution=args.resolution,
        condition_cols=condition_cols,
        max_ring=args.max_ring,
        min_bucket_size=args.min_bucket_size,
        seed=args.seed,
    )

    if args.graphml:
        out = enrich_road_context(
            out,
            graphml_path=args.graphml,
            signals_path=args.signals if args.signals else None,
            columns=ROAD_CONTEXT_COLUMNS & set(out.columns),
            overwrite=True,
            verbose=True,
        )

    weather_stats = None
    if args.weather_reference_csv:
        weather_condition_cols = [col.strip() for col in args.weather_condition_cols.split(",") if col.strip()]
        out, weather_stats = _causal_weather_resample(
            out,
            reference_csv=args.weather_reference_csv,
            condition_cols=weather_condition_cols,
            min_bucket_size=args.weather_min_bucket_size,
            seed=args.seed,
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    report = {"input": args.input_csv, "output": str(output_csv), "rows": len(out), "h3": h3_stats, "weather": weather_stats}
    report_path = output_csv.with_suffix(".h3_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply H3 road-cell anchor projection to synthetic crash rows.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--reference_csv", default=str(ROOT / "data" / "nyc_crash_2024" / "train.csv"))
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--condition_cols", default="SEASON,TIME_PERIOD")
    parser.add_argument("--max_ring", type=int, default=2)
    parser.add_argument("--min_bucket_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graphml", default=str(ROOT / "raw_data" / "osm" / "2024" / "nyc_drive_graph.graphml"))
    parser.add_argument("--signals", default=str(ROOT / "raw_data" / "osm" / "2024" / "nyc_traffic_signals.geojson"))
    parser.add_argument("--weather_reference_csv", default=None)
    parser.add_argument("--weather_condition_cols", default="SEASON,TIME_PERIOD")
    parser.add_argument("--weather_min_bucket_size", type=int, default=30)
    args = parser.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()