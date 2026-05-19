"""
Generate sparse macro-level causal masks for the 2024 v2 schema.

The mask follows the current paper-facing mechanism:
  time -> weather
  geo -> weather (weak)
  geo -> road context
  weather / road context / vehicle / time-light proxy -> crash type
  crash type / road context / vehicle / weather -> injury outcome

It writes CausalDiffTab-ready num/cat masks directly, without applying the older
domain rule expander that would add many extra edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CDT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _feature_groups(groups: dict) -> dict[str, list[str]]:
    return {
        "geo": ["LATITUDE", "LONGITUDE"],
        "time": ["CRASH_TIME_SIN", "CRASH_TIME_COS", "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK"],
        "time_light_proxy": ["CRASH_TIME_SIN", "CRASH_TIME_COS", "IS_AM_PEAK", "IS_PM_PEAK"],
        "weather": ["TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION"],
        "road_context": ["DIST_TO_SIGNAL_M", "INFERRED_LANES", "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "OSM_TYPE"],
        "vehicle_type": list(groups.get("vehicle_binary", [])),
        "crash_type": list(groups.get("factor_binary", [])),
        "injury_outcome": list(groups.get("injury_binary", [])) + ["TOTAL_VEHICLES", "IS_MULTI_VEHICLE"],
    }


def _make_column_matrix(features: list[str], groups: dict) -> tuple[np.ndarray, list[dict]]:
    idx = {name: i for i, name in enumerate(features)}
    macro = _feature_groups(groups)
    matrix = np.zeros((len(features), len(features)), dtype=np.float32)
    records: list[dict] = []

    def add(src_group: str, dst_group: str, weight: float, note: str) -> None:
        added = 0
        for src in macro[src_group]:
            for dst in macro[dst_group]:
                if src in idx and dst in idx and src != dst:
                    matrix[idx[src], idx[dst]] = max(matrix[idx[src], idx[dst]], float(weight))
                    added += 1
        records.append({
            "source_group": src_group,
            "target_group": dst_group,
            "weight": float(weight),
            "expanded_edges": added,
            "note": note,
        })

    add("time", "weather", 0.80, "season/time-of-day strongly shape weather context")
    add("geo", "weather", 0.25, "NYC microclimate is spatial but kept weak")
    add("geo", "road_context", 1.00, "road attributes are location-determined")
    add("weather", "crash_type", 0.70, "weather changes proximate crash mechanism")
    add("road_context", "crash_type", 0.80, "road context shapes crash mechanism")
    add("vehicle_type", "crash_type", 0.80, "vehicle mix shapes crash mechanism")
    add("time_light_proxy", "crash_type", 0.45, "light condition is absent, so time acts as a weak proxy")
    add("crash_type", "injury_outcome", 1.00, "proximate crash type drives injury outcome")
    add("road_context", "injury_outcome", 0.50, "road context affects injury severity")
    add("vehicle_type", "injury_outcome", 0.50, "vehicle type affects injury severity")
    add("weather", "injury_outcome", 0.40, "weather/light context affects injury severity")

    np.fill_diagonal(matrix, 0.0)
    return matrix, records


def _build_masks(info: dict, matrix: np.ndarray, features: list[str], out_dir: Path, include_target_edges: bool) -> tuple[np.ndarray, np.ndarray]:
    num_cols = info["num_col_names"]
    cat_cols = info["cat_col_names"]
    cat_sizes = info["cat_sizes"]
    idx = {name: i for i, name in enumerate(features)}

    num_idx = [idx[name] for name in num_cols]
    cat_idx = [idx[name] for name in cat_cols]

    num_mask = np.zeros((len(num_cols) + 1, len(num_cols) + 1), dtype=np.float32)
    num_mask[1:, 1:] = matrix[np.ix_(num_idx, num_idx)]

    if include_target_edges:
        target_num_sources = ["TEMP_C", "prcp", "WIND_SPEED_KMH", "DIST_TO_SIGNAL_M", "INFERRED_LANES"]
        for name in target_num_sources:
            if name in num_cols:
                num_mask[0, num_cols.index(name) + 1] = 0.50

    expanded = [size + 1 for size in cat_sizes]
    cat_mask = np.zeros((sum(expanded), sum(expanded)), dtype=np.float32)
    offsets = []
    cursor = 0
    for size in expanded:
        offsets.append(cursor)
        cursor += size

    cat_matrix = matrix[np.ix_(cat_idx, cat_idx)]
    for i, size_i in enumerate(expanded):
        for j, size_j in enumerate(expanded):
            value = float(cat_matrix[i, j])
            if value > 0:
                oi, oj = offsets[i], offsets[j]
                cat_mask[oi:oi + size_i, oj:oj + size_j] = value

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "num_causal_mask.npy", num_mask)
    np.save(out_dir / "cat_causal_mask.npy", cat_mask)
    return num_mask, cat_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sparse macro causal masks for v2 schema")
    parser.add_argument("--dataname", default="nyc_crash_2024_v2")
    parser.add_argument("--stage1_dataname", default=None)
    parser.add_argument("--mask_subdir", default="causal_masks_macro_sparse")
    args = parser.parse_args()

    dataname = args.dataname
    stage1_name = args.stage1_dataname or (
        dataname.replace("nyc_crash", "nyc_stage1", 1) if dataname.startswith("nyc_crash_") else "nyc_stage1"
    )
    full_dir = CDT_ROOT / "data" / dataname
    stage1_dir = CDT_ROOT / "data" / stage1_name
    groups = _load_json(CDT_ROOT / "data" / "processed" / "column_groups.json")
    features = groups["continuous_cols"] + groups["categorical_cols"]
    matrix, edge_records = _make_column_matrix(features, groups)

    info_full = _load_json(full_dir / "info.json")
    full_num, full_cat = _build_masks(info_full, matrix, features, full_dir / args.mask_subdir, include_target_edges=True)

    info_stage1 = _load_json(stage1_dir / "info.json")
    stage1_num, stage1_cat = _build_masks(info_stage1, matrix, features, stage1_dir / args.mask_subdir, include_target_edges=False)

    report = {
        "dataname": dataname,
        "stage1_dataname": stage1_name,
        "mask_subdir": args.mask_subdir,
        "column_level_edges": int((matrix > 0).sum()),
        "full_num_nonzero": int((full_num > 0).sum()),
        "full_cat_nonzero": int((full_cat > 0).sum()),
        "stage1_num_nonzero": int((stage1_num > 0).sum()),
        "stage1_cat_nonzero": int((stage1_cat > 0).sum()),
        "edge_groups": edge_records,
    }
    report_path = full_dir / args.mask_subdir / "macro_sparse_mask_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[saved] full masks: {full_dir / args.mask_subdir}")
    print(f"  num={full_num.shape}, nonzero={report['full_num_nonzero']}")
    print(f"  cat={full_cat.shape}, nonzero={report['full_cat_nonzero']}")
    print(f"[saved] stage1 masks: {stage1_dir / args.mask_subdir}")
    print(f"  num={stage1_num.shape}, nonzero={report['stage1_num_nonzero']}")
    print(f"  cat={stage1_cat.shape}, nonzero={report['stage1_cat_nonzero']}")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()