"""
Hierarchical CausalDiffTab - 合成样本后处理
============================================
将扩散模型输出的 samples.csv 还原为物理值:
  1. 连续特征: continuous_scaler.pkl inverse_transform → 原始尺度
  2. 分类特征: cat.codes 索引 → 原始标签 (字母序映射)
  3. 目标列: clamp 到非负整数
  4. 物理合理性校验

Usage:
  python src/postprocess_samples.py \
    --samples_csv result/nyc_crash/stage3_full_full/500/samples.csv \
    --output_csv  result/nyc_crash/stage3_full_full/500/samples_physical.csv
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

VEHICLE_CODE_COLS = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
VEHICLE_PRIORITY = [
    "is_emergency",
    "is_taxi",
    "is_bus",
    "is_truck",
    "is_other_vehicle",
    "is_pickup",
    "is_van",
    "is_motorcycle",
    "is_bicycle",
    "is_suv",
    "is_sedan",
]
VEHICLE_CANONICAL_LABEL = {
    "is_sedan": "Sedan",
    "is_suv": "Station Wagon/Sport Utility Vehicle",
    "is_taxi": "Taxi",
    "is_truck": "Box Truck",
    "is_other_vehicle": "Other",
    "is_pickup": "Pick-up Truck",
    "is_bus": "Bus",
    "is_van": "Van",
    "is_motorcycle": "Motorcycle",
    "is_bicycle": "Bicycle",
    "is_emergency": "Ambulance",
}

WEATHER_COLUMNS = ["TEMP_C", "prcp", "WIND_SPEED_KMH", "coco", "WEATHER_CONDITION"]
INJURY_BIN_COLUMNS = [
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",
]
DEFAULT_TARGET_CONDITION_COLS = [
    "TOTAL_VEHICLES",
    "IS_MULTI_VEHICLE",
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",
]


def _normalize_condition_value(value) -> str:
    if pd.isna(value):
        return "__NA__"
    text = str(value).strip()
    return text if text else "__NA__"


def _hour_to_hhmm(hour_series: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour_series, errors="coerce").fillna(0.0).clip(0.0, 23.999)
    hours = h.to_numpy(dtype=float, copy=False)
    total_minutes = np.round(hours * 60.0).astype(int)
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return pd.Series([f"{int(a):02d}:{int(b):02d}" for a, b in zip(hh, mm)], index=hour_series.index)


def _build_date_pools(reference_raw_csv: str) -> dict:
    """Build date pools keyed by (season, day_of_week) from reference raw csv."""
    try:
        ref = pd.read_csv(reference_raw_csv, usecols=["CRASH DATE"], low_memory=False)
    except Exception:
        return {"by_key": {}, "by_season": {}, "all": []}

    dt = pd.to_datetime(ref["CRASH DATE"], errors="coerce")
    dt = dt.dropna()
    if len(dt) == 0:
        return {"by_key": {}, "by_season": {}, "all": []}

    months = dt.dt.month
    season = pd.Series(np.where(months.isin([12, 1, 2]), "winter", ""), index=dt.index)
    season = season.mask(months.isin([3, 4, 5]), "spring")
    season = season.mask(months.isin([6, 7, 8]), "summer")
    season = season.mask(months.isin([9, 10, 11]), "autumn")
    dow = dt.dt.dayofweek.astype(int)

    by_key: dict = {}
    by_season: dict = {}
    all_dates = [x for x in dt.dt.date.tolist()]

    for d, s, w in zip(dt.dt.date.tolist(), season.tolist(), dow.tolist()):
        k = (str(s), int(w))
        by_key.setdefault(k, []).append(d)
        by_season.setdefault(str(s), []).append(d)

    return {"by_key": by_key, "by_season": by_season, "all": all_dates}


def _synthesize_crash_date(df: pd.DataFrame, reference_raw_csv: str | None) -> pd.Series:
    """Synthesize CRASH DATE aligned to season/day_of_week, backed by reference date distribution."""
    pools = {"by_key": {}, "by_season": {}, "all": []}
    if reference_raw_csv and os.path.exists(reference_raw_csv):
        pools = _build_date_pools(reference_raw_csv)

    rng = np.random.RandomState(42)
    season = df["SEASON"].astype(str).str.lower() if "SEASON" in df.columns else pd.Series("", index=df.index)
    dow = pd.to_numeric(df["DAY_OF_WEEK"], errors="coerce").fillna(0).astype(int) if "DAY_OF_WEEK" in df.columns else pd.Series(0, index=df.index)

    picked = []
    all_dates = pools.get("all", [])
    for s, w in zip(season.tolist(), dow.tolist()):
        cand = pools.get("by_key", {}).get((str(s), int(w)), [])
        if not cand:
            cand = pools.get("by_season", {}).get(str(s), [])
        if not cand:
            cand = all_dates
        if cand:
            picked.append(cand[int(rng.randint(0, len(cand)))])
        else:
            # Fallback: valid synthetic 2017 date
            base = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(rng.randint(0, 365)))
            picked.append(base.date())

    return pd.Series(pd.to_datetime(picked).strftime("%m/%d/%Y"), index=df.index)


def _restore_vehicle_type_codes(df: pd.DataFrame) -> int:
    present = [c for c in VEHICLE_PRIORITY if c in df.columns]
    if not present:
        return 0

    out = pd.Series("", index=df.index, dtype=object)
    for c in present:
        flag = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float) > 0.5
        fill_val = VEHICLE_CANONICAL_LABEL.get(c, "")
        out = out.mask(flag & (out == ""), fill_val)

    df["VEHICLE TYPE CODE 1"] = out
    for c in VEHICLE_CODE_COLS[1:]:
        if c not in df.columns:
            df[c] = ""
    return int((out != "").sum())


def _apply_semantic_repair(df: pd.DataFrame) -> dict:
    """Apply weak post-sampling semantic consistency repair.

    This intentionally avoids training-time CE losses. It only repairs vehicle
    indicator consistency and derived count flags when the relevant columns are
    present, leaving injury/cause labels untouched for evaluation.
    """
    stats = {"vehicle_rows_filled": 0, "total_vehicles_updated": 0, "multi_vehicle_updated": 0}

    vehicle_cols = [c for c in VEHICLE_PRIORITY if c in df.columns]
    if not vehicle_cols:
        return stats

    for col in vehicle_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0) > 0.5).astype(int)

    active_count = df[vehicle_cols].sum(axis=1).astype(int)
    zero_vehicle = active_count <= 0
    fallback_col = "is_sedan" if "is_sedan" in vehicle_cols else vehicle_cols[-1]
    if zero_vehicle.any():
        df.loc[zero_vehicle, fallback_col] = 1
        stats["vehicle_rows_filled"] = int(zero_vehicle.sum())
        active_count = df[vehicle_cols].sum(axis=1).astype(int)

    if "TOTAL_VEHICLES" in df.columns:
        old_total = pd.to_numeric(df["TOTAL_VEHICLES"], errors="coerce").fillna(0).round().astype(int)
        repaired_total = np.maximum(old_total, active_count).clip(1, 5).astype(int)
        stats["total_vehicles_updated"] = int((old_total != repaired_total).sum())
        df["TOTAL_VEHICLES"] = repaired_total

    if "IS_MULTI_VEHICLE" in df.columns:
        if "TOTAL_VEHICLES" in df.columns:
            multi = (pd.to_numeric(df["TOTAL_VEHICLES"], errors="coerce").fillna(1).astype(int) >= 2).astype(int)
        else:
            multi = (active_count >= 2).astype(int)
        old_multi = (pd.to_numeric(df["IS_MULTI_VEHICLE"], errors="coerce").fillna(0) > 0.5).astype(int)
        stats["multi_vehicle_updated"] = int((old_multi != multi).sum())
        df["IS_MULTI_VEHICLE"] = multi

    return stats


def _causal_weather_resample(
    df: pd.DataFrame,
    reference_csv: str,
    condition_cols: list[str],
    min_bucket_size: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Resample weather jointly from empirical pools conditioned on causal parents."""
    stats = {
        "enabled": False,
        "reference_csv": reference_csv,
        "condition_cols": condition_cols,
        "weather_cols": [],
        "exact": 0,
        "season_fallback": 0,
        "global_fallback": 0,
        "skipped": 0,
    }
    if not reference_csv or not os.path.exists(reference_csv):
        stats["reason"] = "reference_csv_missing"
        return df, stats

    ref = pd.read_csv(reference_csv, low_memory=False)
    weather_cols = [c for c in WEATHER_COLUMNS if c in df.columns and c in ref.columns]
    condition_cols = [c for c in condition_cols if c in df.columns and c in ref.columns]
    if not weather_cols:
        stats["reason"] = "weather_cols_missing"
        return df, stats
    if not condition_cols:
        condition_cols = [c for c in ["SEASON"] if c in df.columns and c in ref.columns]
    if not condition_cols:
        stats["reason"] = "condition_cols_missing"
        return df, stats

    ref = ref.dropna(subset=weather_cols, how="all").copy()
    if len(ref) == 0:
        stats["reason"] = "empty_reference_weather"
        return df, stats

    stats["enabled"] = True
    stats["weather_cols"] = weather_cols
    rng = np.random.RandomState(seed)

    def key_from_row(row, cols):
        return tuple(_normalize_condition_value(row[c]) for c in cols)

    exact_pools: dict[tuple[str, ...], np.ndarray] = {}
    for key, idx in ref.groupby(condition_cols, dropna=False).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        norm_key = tuple(_normalize_condition_value(v) for v in key)
        exact_pools[norm_key] = np.asarray(list(idx), dtype=int)

    season_pools: dict[str, np.ndarray] = {}
    if "SEASON" in ref.columns:
        for season, idx in ref.groupby("SEASON", dropna=False).groups.items():
            season_pools[_normalize_condition_value(season)] = np.asarray(list(idx), dtype=int)

    global_pool = ref.index.to_numpy(dtype=int)
    sampled_indices: list[int] = []
    for _, row in df.iterrows():
        key = key_from_row(row, condition_cols)
        pool = exact_pools.get(key)
        source = "exact"
        if pool is None or len(pool) < min_bucket_size:
            season_value = _normalize_condition_value(row["SEASON"]) if "SEASON" in df.columns else "__NA__"
            pool = season_pools.get(season_value)
            source = "season_fallback"
        if pool is None or len(pool) == 0:
            pool = global_pool
            source = "global_fallback"
        if pool is None or len(pool) == 0:
            sampled_indices.append(-1)
            stats["skipped"] += 1
            continue
        sampled_indices.append(int(pool[rng.randint(0, len(pool))]))
        stats[source] += 1

    out = df.copy()
    valid_pos = [i for i, idx in enumerate(sampled_indices) if idx >= 0]
    if valid_pos:
        picked_ref_idx = [sampled_indices[i] for i in valid_pos]
        sampled_weather = ref.loc[picked_ref_idx, weather_cols].reset_index(drop=True)
        out.loc[out.index[valid_pos], weather_cols] = sampled_weather.to_numpy()

    for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "TEMP_C" in out.columns:
        out["TEMP_C"] = out["TEMP_C"].clip(-30, 45).round(1)
    if "prcp" in out.columns:
        out["prcp"] = out["prcp"].clip(0, 200).round(3)
        out.loc[out["prcp"].abs() < 1e-3, "prcp"] = 0.0
    if "WIND_SPEED_KMH" in out.columns:
        out["WIND_SPEED_KMH"] = out["WIND_SPEED_KMH"].clip(0, 150)

    return out, stats


def _injury_lower_bound(df: pd.DataFrame) -> pd.Series:
    present = [c for c in INJURY_BIN_COLUMNS if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index, dtype=int)
    flags = pd.DataFrame(index=df.index)
    for col in present:
        flags[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0.5).astype(int)
    return flags.sum(axis=1).clip(lower=0).astype(int)


def _causal_target_resample(
    df: pd.DataFrame,
    reference_csv: str,
    target_col: str,
    condition_cols: list[str],
    min_bucket_size: int = 30,
    seed: int = 42,
    enforce_injury_lower_bound: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Resample target from source-domain empirical conditional count distribution."""
    stats = {
        "enabled": False,
        "reference_csv": reference_csv,
        "target_col": target_col,
        "condition_cols": condition_cols,
        "exact": 0,
        "core_fallback": 0,
        "global_fallback": 0,
        "lower_bound_updates": 0,
        "skipped": 0,
    }
    if target_col not in df.columns:
        stats["reason"] = "target_missing_in_synthetic"
        return df, stats
    if not reference_csv or not os.path.exists(reference_csv):
        stats["reason"] = "reference_csv_missing"
        return df, stats

    ref = pd.read_csv(reference_csv, low_memory=False)
    if target_col not in ref.columns:
        stats["reason"] = "target_missing_in_reference"
        return df, stats

    condition_cols = [c for c in condition_cols if c in df.columns and c in ref.columns]
    core_cols = [c for c in ["TOTAL_VEHICLES", "IS_MULTI_VEHICLE"] if c in df.columns and c in ref.columns]
    if not condition_cols:
        condition_cols = core_cols
    if not condition_cols:
        stats["reason"] = "condition_cols_missing"
        return df, stats

    ref = ref.copy()
    ref[target_col] = pd.to_numeric(ref[target_col], errors="coerce")
    ref = ref.dropna(subset=[target_col])
    if len(ref) == 0:
        stats["reason"] = "empty_reference_target"
        return df, stats

    stats["enabled"] = True
    rng = np.random.RandomState(seed)

    def key_from_row(row, cols):
        return tuple(_normalize_condition_value(row[c]) for c in cols)

    exact_pools: dict[tuple[str, ...], np.ndarray] = {}
    for key, idx in ref.groupby(condition_cols, dropna=False).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        norm_key = tuple(_normalize_condition_value(v) for v in key)
        exact_pools[norm_key] = np.asarray(list(idx), dtype=int)

    core_pools: dict[tuple[str, ...], np.ndarray] = {}
    if core_cols:
        for key, idx in ref.groupby(core_cols, dropna=False).groups.items():
            if not isinstance(key, tuple):
                key = (key,)
            norm_key = tuple(_normalize_condition_value(v) for v in key)
            core_pools[norm_key] = np.asarray(list(idx), dtype=int)

    global_pool = ref.index.to_numpy(dtype=int)
    sampled_values: list[float] = []
    for _, row in df.iterrows():
        key = key_from_row(row, condition_cols)
        pool = exact_pools.get(key)
        source = "exact"
        if pool is None or len(pool) < min_bucket_size:
            core_key = key_from_row(row, core_cols) if core_cols else tuple()
            pool = core_pools.get(core_key)
            source = "core_fallback"
        if pool is None or len(pool) == 0:
            pool = global_pool
            source = "global_fallback"
        if pool is None or len(pool) == 0:
            sampled_values.append(np.nan)
            stats["skipped"] += 1
            continue
        sampled_idx = int(pool[rng.randint(0, len(pool))])
        sampled_values.append(float(ref.loc[sampled_idx, target_col]))
        stats[source] += 1

    out = df.copy()
    sampled = pd.Series(sampled_values, index=out.index)
    repaired = pd.to_numeric(sampled, errors="coerce").fillna(
        pd.to_numeric(out[target_col], errors="coerce")
    )
    repaired = repaired.clip(lower=0).round().astype(int)

    if enforce_injury_lower_bound:
        lower_bound = _injury_lower_bound(out)
        before = repaired.copy()
        repaired = np.maximum(repaired, lower_bound).astype(int)
        stats["lower_bound_updates"] = int((before != repaired).sum())

    out[target_col] = repaired
    return out, stats


def _find_reference_raw_csv(explicit_path: str | None = None) -> str | None:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    candidates = [
        CDT_ROOT.parent.parent / "tab-ddpm-main" / "nyc_2017_pristine_v9.csv",
        CDT_ROOT.parent.parent / "tab-ddpm-main" / "nyc_2017_pristine_v8.csv",
        CDT_ROOT.parent / "nyc_2017_pristine_v9.csv",
        CDT_ROOT.parent / "nyc_2017_pristine_v8.csv",
        CDT_ROOT.parent.parent / "nyc_2017_pristine_v9.csv",
        CDT_ROOT.parent.parent / "nyc_2017_pristine_v8.csv",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _export_raw_aligned(df: pd.DataFrame, output_csv: str, reference_raw_csv: str | None) -> str | None:
    if not reference_raw_csv or (not os.path.exists(reference_raw_csv)):
        return None
    raw_head = pd.read_csv(reference_raw_csv, nrows=1, low_memory=False)
    aligned = pd.DataFrame(index=df.index, columns=raw_head.columns)
    for c in aligned.columns:
        if c in df.columns:
            aligned[c] = df[c].values
        else:
            aligned[c] = ""

    base, ext = os.path.splitext(output_csv)
    out = f"{base}_raw_aligned{ext}"
    aligned.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def load_category_mappings(processed_csv: str, cat_cols: list, info: Optional[dict] = None) -> dict:
    """
    从 processed_hierarchical.csv 重建 cat.codes → 原始标签 的映射。
    pd.Categorical 默认按字母/数值序排列, codes 从 0 开始。
    """
    mappings = {}
    info_mappings = (info or {}).get("cat_label_mappings", {})
    for col in cat_cols:
        if col in info_mappings:
            mappings[col] = {int(k): v for k, v in info_mappings[col].items()}

    df = pd.read_csv(processed_csv, usecols=cat_cols, low_memory=False)
    for col in cat_cols:
        if col in mappings:
            continue
        cat_series = df[col].astype("category")
        categories = cat_series.cat.categories.tolist()
        mappings[col] = {i: v for i, v in enumerate(categories)}
    return mappings


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine 距离（米），标量或 numpy array 均可。"""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def postprocess(
    samples_csv: str,
    output_csv: str,
    processed_csv: Optional[str] = None,
    scaler_pkl: Optional[str] = None,
    column_groups_json: Optional[str] = None,
    info_json: Optional[str] = None,
    reference_raw_csv: Optional[str] = None,
    road_graphml: Optional[str] = None,
    road_signals: Optional[str] = None,
    snap_max_dist_m: float = 300.0,
    recompute_osm_after_snap: bool = True,
    # Phase 1: context_lookup 集成参数
    context_mode: Optional[str] = None,
    weather_csv: Optional[str] = None,
    context_year: Optional[int] = None,
    road_snap_cache: Optional[str] = None,
    use_candidate_snap: bool = False,
    semantic_repair: bool = False,
    causal_weather_resample: bool = False,
    weather_reference_csv: Optional[str] = None,
    weather_condition_cols: Optional[list[str]] = None,
    weather_min_bucket_size: int = 30,
    weather_resample_seed: int = 42,
    causal_target_resample: bool = False,
    target_reference_csv: Optional[str] = None,
    target_condition_cols: Optional[list[str]] = None,
    target_min_bucket_size: int = 30,
    target_resample_seed: int = 42,
):
    """
    完整后处理管线:
    1. 加载合成样本 (模型输出空间)
    2. 连续特征逆变换 → 物理值
    3. 分类特征解码 → 原始标签
    4. 目标列整理
    5. 物理合理性校验与裁剪
    6. 道路约束 snap（使用候选点集或 nearest_edges）
    7. context_lookup 补全（天气 + OSM 属性，可选）
    8. semantic_repair 弱语义一致性修复（可选，不进入训练 loss）
    """
    processed_csv = processed_csv or str(CDT_ROOT / "data" / "processed" / "processed_hierarchical.csv")
    scaler_pkl = scaler_pkl or str(CDT_ROOT / "data" / "processed" / "continuous_scaler.pkl")
    column_groups_json = column_groups_json or str(CDT_ROOT / "data" / "processed" / "column_groups.json")
    info_json = info_json or str(CDT_ROOT / "data" / "nyc_crash" / "info.json")
    ref_csv = _find_reference_raw_csv(reference_raw_csv)

    syn = pd.read_csv(samples_csv)
    print(f"[load] {len(syn)} synthetic samples from {samples_csv}")
    print(f"  columns: {syn.shape[1]}")

    with open(column_groups_json, "r", encoding="utf-8") as f:
        groups = json.load(f)
    with open(info_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    num_cols = info["num_col_names"]
    cat_cols = info["cat_col_names"]
    target_col = info.get("target_col", "NUMBER OF PERSONS INJURED")

    # ========================================
    # Step 1: 连续特征逆变换
    # ========================================
    # NOTE: sample_conditional.py 中 tensor_to_dataframe() 已通过 TabDiff 的 num_inverse
    # 完成了 QuantileTransformer 逆变换，原始 CSV 已是物理值。
    # 若再次调用 scaler.inverse_transform() 会把物理值当 Gaussian z-score 处理，导致
    # LATITUDE/LONGITUDE 等列全部坍缩为最大值（双重逆变换 bug）。
    # 因此检查连续列是否已在物理范围内：若 LATITUDE 在 [30, 50] 则跳过 scaler。
    _skip_scaler = False
    if "LATITUDE" in syn.columns:
        lat_vals = pd.to_numeric(syn["LATITUDE"], errors="coerce").dropna()
        if len(lat_vals) > 0 and lat_vals.min() > 30:
            _skip_scaler = True
            print("[inverse] Raw CSV already in physical scale (LATITUDE > 30), skipping QuantileTransformer inverse")

    if not _skip_scaler and os.path.exists(scaler_pkl):
        with open(scaler_pkl, "rb") as f:
            scaler_data = pickle.load(f)

        scaler = scaler_data["scaler"]
        scaler_columns = scaler_data["columns"]
        print(f"[inverse] QuantileTransformer loaded, columns: {scaler_columns}")

        cols_to_invert = [c for c in scaler_columns if c in syn.columns]
        if cols_to_invert:
            vals = syn[cols_to_invert].values.copy()
            vals_clipped = np.clip(vals, -5.2, 5.2)
            restored = scaler.inverse_transform(vals_clipped)
            syn[cols_to_invert] = restored
            print(f"  -> {len(cols_to_invert)} continuous columns restored to physical scale")
    elif _skip_scaler:
        pass  # already physical
    else:
        print(f"[warn] Scaler not found at {scaler_pkl}, skipping continuous inverse")

    # ========================================
    # Step 2: 分类特征解码
    # ========================================
    cat_in_syn = [c for c in cat_cols if c in syn.columns]
    if cat_in_syn:
        mappings = load_category_mappings(processed_csv, cat_in_syn, info=info)
        decoded_count = 0
        for col in cat_in_syn:
            if col in mappings:
                mapping = mappings[col]
                raw_codes = pd.to_numeric(syn[col], errors="coerce")
                raw_codes = raw_codes.replace([np.inf, -np.inf], np.nan)
                # Floor/round to nearest valid code space before mapping.
                safe_codes = raw_codes.round().astype("Int64")
                mapped = safe_codes.map(mapping)
                syn[col] = mapped
                unmapped = syn[col].isna().sum()
                if unmapped > 0:
                    # Use mapped mode first; fallback to the first known category.
                    mode_vals = syn[col].dropna().mode()
                    if len(mode_vals) > 0:
                        most_common = mode_vals.iloc[0]
                    else:
                        most_common = next(iter(mapping.values()))
                    syn[col] = syn[col].fillna(most_common)
                    print(f"  [warn] {col}: {unmapped} unmapped codes filled with '{most_common}'")
                decoded_count += 1
        print(f"[decode] {decoded_count} categorical columns decoded to original labels")

    # ========================================
    # Step 3: 目标列整理
    # ========================================
    if target_col in syn.columns:
        tgt = pd.to_numeric(syn[target_col], errors="coerce")
        tgt = tgt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        syn[target_col] = tgt.clip(lower=0).round().astype(int)
        print(f"[target] {target_col}: clipped to non-negative integers")
        print(f"  range: [{syn[target_col].min()}, {syn[target_col].max()}], "
              f"mean={syn[target_col].mean():.2f}")

    # ========================================
    # Step 4: 物理合理性校验与裁剪
    # ========================================
    print("\n[validate] Physical sanity checks:")

    if "LATITUDE" in syn.columns:
        before = (syn["LATITUDE"] < 40.4) | (syn["LATITUDE"] > 40.95)
        syn["LATITUDE"] = syn["LATITUDE"].clip(40.4, 40.95)
        print(f"  LATITUDE: {before.sum()} clipped to [40.4, 40.95]")

    if "LONGITUDE" in syn.columns:
        before = (syn["LONGITUDE"] < -74.3) | (syn["LONGITUDE"] > -73.7)
        syn["LONGITUDE"] = syn["LONGITUDE"].clip(-74.3, -73.7)
        print(f"  LONGITUDE: {before.sum()} clipped to [-74.3, -73.7]")

    # ── Road-network snap（bbox clip 之后执行）──────────────────────────
    _road_graphml = road_graphml or str(CDT_ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml")
    if "LATITUDE" in syn.columns and "LONGITUDE" in syn.columns and os.path.exists(_road_graphml):
        print(f"\n[road_snap] 加载路网: {_road_graphml} ...")
        try:
            import osmnx as ox

            def _compat_bool(v) -> bool:
                if isinstance(v, bool):
                    return v
                return str(v).lower() in ("yes", "true", "1", "on")

            G = ox.load_graphml(
                _road_graphml,
                edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
                graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
            )
            G_proj = ox.project_graph(G, to_crs="EPSG:32618")

            valid_mask = syn["LATITUDE"].notna() & syn["LONGITUDE"].notna()
            n_try = int(valid_mask.sum())
            print(f"[road_snap] 尝试 snap {n_try} 行...")

            lats = syn.loc[valid_mask, "LATITUDE"].to_numpy(dtype=float)
            lons = syn.loc[valid_mask, "LONGITUDE"].to_numpy(dtype=float)

            # 用原始地理图（EPSG:4326）做 nearest_edges，边 geometry 与坐标系一致
            ne_edges = ox.nearest_edges(G, X=lons, Y=lats)

            snap_lats, snap_lons = [], []
            for (u, v, k), orig_lat, orig_lon in zip(ne_edges, lats, lons):
                edge_data = G.get_edge_data(u, v, k) or {}
                geom = edge_data.get("geometry")
                if geom is not None:
                    try:
                        from shapely.geometry import LineString
                        mid = geom.interpolate(0.5, normalized=True)
                        snap_lons.append(mid.x)
                        snap_lats.append(mid.y)
                    except Exception:
                        # 几何操作失败，取两端点均值
                        n_data = G.nodes[u]
                        n_data2 = G.nodes[v]
                        snap_lons.append((n_data.get("x", orig_lon) + n_data2.get("x", orig_lon)) / 2)
                        snap_lats.append((n_data.get("y", orig_lat) + n_data2.get("y", orig_lat)) / 2)
                else:
                    n_data = G.nodes[u]
                    n_data2 = G.nodes[v]
                    snap_lons.append((n_data.get("x", orig_lon) + n_data2.get("x", orig_lon)) / 2)
                    snap_lats.append((n_data.get("y", orig_lat) + n_data2.get("y", orig_lat)) / 2)

            snap_lats = np.array(snap_lats)
            snap_lons = np.array(snap_lons)
            snap_dists = _haversine_m(lats, lons, snap_lats, snap_lons)

            within_mask = snap_dists < snap_max_dist_m
            n_snapped = int(within_mask.sum())
            n_skipped = n_try - n_snapped

            # 统计摘要
            print(f"  snap 总行数: {n_try}")
            print(f"  实际 snap 行: {n_snapped}  ({n_snapped/n_try*100:.1f}%)")
            print(f"  超过 {snap_max_dist_m}m 跳过: {n_skipped}  ({n_skipped/n_try*100:.1f}%)")
            if n_snapped > 0:
                d = snap_dists[within_mask]
                print(f"  snap_dist_m 统计 (snapped rows):")
                print(f"    mean={d.mean():.1f}  p50={np.percentile(d,50):.1f}"
                      f"  p95={np.percentile(d,95):.1f}  p99={np.percentile(d,99):.1f}  max={d.max():.1f}")
                print(f"    >30m: {(d>30).sum()} ({(d>30).mean()*100:.1f}%)")
                print(f"    >100m: {(d>100).sum()} ({(d>100).mean()*100:.1f}%)")

            # 更新坐标（仅在 snap_max_dist_m 内）
            valid_indices = syn.index[valid_mask]
            within_valid = np.where(within_mask)[0]
            snap_idx = valid_indices[within_valid]
            syn.loc[snap_idx, "LATITUDE"] = snap_lats[within_valid]
            syn.loc[snap_idx, "LONGITUDE"] = snap_lons[within_valid]

            # 重算 OSM 特征列（只对 snap 更新的行）
            if recompute_osm_after_snap and n_snapped > 0:
                osm_cols = [
                    "DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL", "OSM_TYPE", "OSM_ONEWAY",
                    "INFERRED_LANES", "REAL_SPEED_LIMIT", "HAS_DIVIDER",
                ]
                present_osm = [c for c in osm_cols if c in syn.columns]
                if present_osm:
                    from src.road_snap import enrich_road_context

                    _road_signals = road_signals or str(CDT_ROOT / "raw_data" / "osm" / "nyc_traffic_signals.geojson")
                    enriched_snap = enrich_road_context(
                        syn.loc[snap_idx].copy(),
                        graphml_path=_road_graphml,
                        signals_path=_road_signals,
                        columns=present_osm,
                        overwrite=True,
                        verbose=False,
                    )
                    for col in present_osm:
                        if col in enriched_snap.columns:
                            syn.loc[snap_idx, col] = enriched_snap[col].values
                    print(f"  [road_snap] OSM 特征重算完成（{len(present_osm)} 列，统一 road_context 口径）")

        except ImportError as e:
            print(f"[road_snap] 依赖缺失，跳过: {e}")
        except Exception as e:
            print(f"[road_snap] 路网 snap 失败，跳过: {e}")
    elif "LATITUDE" in syn.columns and not os.path.exists(_road_graphml):
        print(f"[road_snap] graphml 不存在，跳过: {_road_graphml}")
    # ── road snap end ────────────────────────────────────────────────────

    if "TEMP_C" in syn.columns:
        syn["TEMP_C"] = syn["TEMP_C"].clip(-30, 45)
        syn["TEMP_C"] = pd.to_numeric(syn["TEMP_C"], errors="coerce").round(1)
        print(f"  TEMP_C: clipped to [-30, 45]")

    if "prcp" in syn.columns:
        syn["prcp"] = syn["prcp"].clip(0, 200)
        syn["prcp"] = pd.to_numeric(syn["prcp"], errors="coerce").round(3)
        syn.loc[syn["prcp"].abs() < 1e-3, "prcp"] = 0.0
        print(f"  prcp: clipped to [0, 200]")

    if "WIND_SPEED_KMH" in syn.columns:
        syn["WIND_SPEED_KMH"] = syn["WIND_SPEED_KMH"].clip(0, 150)
        print(f"  WIND_SPEED_KMH: clipped to [0, 150]")

    if "CRASH_TIME_SIN" in syn.columns and "CRASH_TIME_COS" in syn.columns:
        syn["CRASH_TIME_SIN"] = syn["CRASH_TIME_SIN"].clip(-1, 1)
        syn["CRASH_TIME_COS"] = syn["CRASH_TIME_COS"].clip(-1, 1)
        angle = np.arctan2(syn["CRASH_TIME_SIN"], syn["CRASH_TIME_COS"])
        frac = (angle % (2 * np.pi)) / (2 * np.pi)
        syn["_CRASH_HOUR"] = (frac * 24).round(1)
        syn["CRASH TIME"] = _hour_to_hhmm(syn["_CRASH_HOUR"])
        print(f"  CRASH_TIME: sin/cos decoded to _CRASH_HOUR [{syn['_CRASH_HOUR'].min():.1f}, "
              f"{syn['_CRASH_HOUR'].max():.1f}]")

    # Build full datetime fields expected by raw table style.
    syn["CRASH DATE"] = _synthesize_crash_date(syn, ref_csv)
    if "CRASH TIME" in syn.columns:
        dt = pd.to_datetime(
            syn["CRASH DATE"].astype(str) + " " + syn["CRASH TIME"].astype(str),
            errors="coerce",
        )
        syn["CRASH_FULL_TIME"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    if causal_weather_resample:
        weather_ref = weather_reference_csv or processed_csv
        cond_cols = weather_condition_cols or ["SEASON", "TIME_PERIOD"]
        syn, weather_stats = _causal_weather_resample(
            syn,
            reference_csv=weather_ref,
            condition_cols=cond_cols,
            min_bucket_size=weather_min_bucket_size,
            seed=weather_resample_seed,
        )
        print(f"  [causal_weather_resample] {weather_stats}")

    if causal_target_resample:
        target_ref = target_reference_csv or processed_csv
        cond_cols = target_condition_cols or DEFAULT_TARGET_CONDITION_COLS
        syn, target_stats = _causal_target_resample(
            syn,
            reference_csv=target_ref,
            target_col=target_col,
            condition_cols=cond_cols,
            min_bucket_size=target_min_bucket_size,
            seed=target_resample_seed,
            enforce_injury_lower_bound=True,
        )
        print(f"  [causal_target_resample] {target_stats}")

    n_vehicle = _restore_vehicle_type_codes(syn)
    if "BOROUGH" not in syn.columns:
        syn["BOROUGH"] = ""
    print(f"  VEHICLE TYPE CODE 1 restored rows: {n_vehicle}")

    if semantic_repair:
        repair_stats = _apply_semantic_repair(syn)
        n_vehicle = _restore_vehicle_type_codes(syn)
        print(f"  [semantic_repair] {repair_stats}; vehicle rows restored={n_vehicle}")

    # ========================================
    # Step 5: 输出统计摘要
    # ========================================
    print("\n" + "=" * 60)
    print("=== Physical-Value Sample Summary ===")
    print("=" * 60)

    for col in num_cols:
        if col in syn.columns:
            print(f"  {col:<25} mean={syn[col].mean():>10.3f}  "
                  f"std={syn[col].std():>10.3f}  "
                  f"[{syn[col].min():>10.3f}, {syn[col].max():>10.3f}]")

    print()
    for col in cat_in_syn[:8]:
        vc = syn[col].value_counts()
        top3 = ", ".join([f"{v}={c}" for v, c in vc.head(3).items()])
        print(f"  {col:<25} {vc.shape[0]} unique  top3: {top3}")

    if target_col in syn.columns:
        print(f"\n  {target_col}: mean={syn[target_col].mean():.2f}, "
              f"max={syn[target_col].max()}")

    # ── Phase 1: 候选点集 snap（use_candidate_snap=True 时使用新模块）──
    if use_candidate_snap and "LATITUDE" in syn.columns and "LONGITUDE" in syn.columns:
        _graphml = road_graphml or str(CDT_ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml")
        if os.path.exists(_graphml):
            try:
                from src.road_snap import build_road_candidate_set, postprocess_latlon_df
                rcs = build_road_candidate_set(
                    graphml_path=_graphml,
                    cache_path=road_snap_cache,
                    verbose=True,
                )
                syn = postprocess_latlon_df(syn, rcs, jitter_m=10.0, verbose=True)
                print("[road_snap] 候选点集吸附完成（新模块）")
            except Exception as e:
                print(f"[road_snap] 候选点集吸附失败，跳过: {e}")

    # ── Phase 1: context_lookup 补全（天气 + OSM）────────────────────────
    if context_mode is not None:
        try:
            from src.context_lookup import auto_config, ContextPipeline
            _year = context_year or 2024
            cfg = auto_config(year=_year, context_mode=context_mode)
            if weather_csv:
                cfg.weather_csv = weather_csv
            if road_graphml:
                cfg.osm_graphml = road_graphml
            pipe = ContextPipeline(cfg, verbose=True)
            print(f"\n[context_lookup] 模式={context_mode}, 年份={_year}")
            syn = pipe.enrich(syn, year=_year)
            print("[context_lookup] 上下文补全完成")
        except Exception as e:
            print(f"[context_lookup] 补全失败，跳过: {e}")

    # ========================================
    # 保存
    # ========================================
    if not output_csv:
        base, ext = os.path.splitext(samples_csv)
        output_csv = f"{base}_physical{ext}"

    syn.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {output_csv} ({len(syn)} rows, {syn.shape[1]} cols)")

    aligned_out = _export_raw_aligned(syn, output_csv, ref_csv)
    if aligned_out:
        print(f"[saved] {aligned_out} (raw schema aligned)")
    else:
        print("[note] raw schema aligned export skipped (reference pristine csv not found)")

    return syn


def main():
    parser = argparse.ArgumentParser(description="Post-process synthetic samples to physical values")
    parser.add_argument("--samples_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--processed_csv", type=str, default=None)
    parser.add_argument("--scaler_pkl", type=str, default=None)
    parser.add_argument("--column_groups", type=str, default=None)
    parser.add_argument("--info_json", type=str, default=None)
    parser.add_argument("--reference_raw_csv", type=str, default=None)
    parser.add_argument("--road_graphml", type=str, default=None,
                        help="OSM graphml 路径，None 则跳过路网 snap")
    parser.add_argument("--road_signals", type=str, default=None,
                        help="信号灯 geojson 路径（可选）")
    parser.add_argument("--snap_max_dist_m", type=float, default=300.0,
                        help="snap 距离上限（米），超过此值的点保留 bbox clip 结果")
    parser.add_argument("--no_recompute_osm", action="store_true",
                        help="snap 后不重算 OSM 特征列")
    parser.add_argument("--context_mode", type=str, default=None,
                        choices=["historical_lookup", "future_simulation", "correction"],
                        help="上下文补全模式（None=跳过）")
    parser.add_argument("--weather_csv", type=str, default=None,
                        help="Open-Meteo 天气 CSV 路径（context_lookup 使用）")
    parser.add_argument("--context_year", type=int, default=None,
                        help="目标年份（自动推断 OSM/天气路径）")
    parser.add_argument("--road_snap_cache", type=str, default=None,
                        help="候选点集 BallTree 缓存 .npz 路径")
    parser.add_argument("--use_candidate_snap", action="store_true",
                        help="使用预构建候选点集做 snap（更快，适合批量生成）")
    parser.add_argument("--semantic_repair", action="store_true",
                        help="启用采样后弱语义一致性修复（车辆类型/TOTAL_VEHICLES/IS_MULTI_VEHICLE）")
    parser.add_argument("--causal_weather_resample", action="store_true",
                        help="按因果上游变量经验重采样天气列（默认 SEASON,TIME_PERIOD）")
    parser.add_argument("--weather_reference_csv", type=str, default=None,
                        help="天气经验池 CSV；迁移评估可传 data/nyc_crash_2025/test.csv，默认用 processed_csv")
    parser.add_argument("--weather_condition_cols", type=str, default="SEASON,TIME_PERIOD",
                        help="天气重采样条件列，逗号分隔，例如 SEASON,TIME_PERIOD")
    parser.add_argument("--weather_min_bucket_size", type=int, default=30,
                        help="条件桶最小样本数；不足则回退到 SEASON，再全局")
    parser.add_argument("--weather_resample_seed", type=int, default=42)
    parser.add_argument("--causal_target_resample", action="store_true",
                        help="按 Stage3 微观变量从源域经验分布重采样 NUMBER OF PERSONS INJURED")
    parser.add_argument("--target_reference_csv", type=str, default=None,
                        help="目标经验池 CSV；默认用 processed_csv。迁移评估不要传 2025 test，避免标签泄漏")
    parser.add_argument("--target_condition_cols", type=str,
                        default=",".join(DEFAULT_TARGET_CONDITION_COLS),
                        help="目标重采样条件列，逗号分隔")
    parser.add_argument("--target_min_bucket_size", type=int, default=30,
                        help="目标条件桶最小样本数；不足则回退到 TOTAL_VEHICLES/IS_MULTI_VEHICLE，再全局")
    parser.add_argument("--target_resample_seed", type=int, default=42)
    args = parser.parse_args()

    weather_condition_cols = [
        c.strip() for c in args.weather_condition_cols.split(",") if c.strip()
    ]
    target_condition_cols = [
        c.strip() for c in args.target_condition_cols.split(",") if c.strip()
    ]

    postprocess(
        samples_csv=args.samples_csv,
        output_csv=args.output_csv,
        processed_csv=args.processed_csv,
        scaler_pkl=args.scaler_pkl,
        column_groups_json=args.column_groups,
        info_json=args.info_json,
        reference_raw_csv=args.reference_raw_csv,
        road_graphml=args.road_graphml,
        road_signals=args.road_signals,
        snap_max_dist_m=args.snap_max_dist_m,
        recompute_osm_after_snap=not args.no_recompute_osm,
        context_mode=args.context_mode,
        weather_csv=args.weather_csv,
        context_year=args.context_year,
        road_snap_cache=args.road_snap_cache,
        use_candidate_snap=args.use_candidate_snap,
        semantic_repair=args.semantic_repair,
        causal_weather_resample=args.causal_weather_resample,
        weather_reference_csv=args.weather_reference_csv,
        weather_condition_cols=weather_condition_cols,
        weather_min_bucket_size=args.weather_min_bucket_size,
        weather_resample_seed=args.weather_resample_seed,
        causal_target_resample=args.causal_target_resample,
        target_reference_csv=args.target_reference_csv,
        target_condition_cols=target_condition_cols,
        target_min_bucket_size=args.target_min_bucket_size,
        target_resample_seed=args.target_resample_seed,
    )


if __name__ == "__main__":
    main()
