"""
structured_eval.py — 五大语义变量专项评估模块
===============================================
实现 Section 1.11 要求的五类核心变量评估指标：

1. eval_time(real_df, synth_df)
   - SEASON 分布 JS 散度
   - IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK 分布差异
   - 时间组合合法性（IS_AM_PEAK + IS_PM_PEAK 同为 1 的比例）

2. eval_spatial(real_df, synth_df, rcs)
   - 经纬度 Wasserstein 距离（lat/lon 分别）
   - 到最近道路距离分布（均值/中位/P95）
   - 生成点落路率 / 偏离率
   - borough / OSM_TYPE 分布 JS

3. eval_vehicle(real_df, synth_df)
   - 每类车辆边际分布（精确率/召回率/F1 可选）
   - 多标签组合 Jaccard
   - TOTAL_VEHICLES 一致性 MAE
   - IS_MULTI_VEHICLE 一致性

4. eval_cause(real_df, synth_df)
   - 每类事故原因边际分布差异
   - 原因共现矩阵 Frobenius 距离
   - 原因-车辆类型联合分布 JS

5. eval_injury(real_df, synth_df)
   - 每个 bin 列分布 JS
   - NUMBER OF PERSONS INJURED 分布（Wasserstein + zero_ratio）
   - count MAE / calibration

6. eval_context(real_df, synth_df, mode)
   - historical_lookup 模式：天气列缺失率
   - correction 模式：calibration 误差统计（若存在 _GEN 列）

7. eval_all(real_df, synth_df, rcs, mode) → 汇总 dict + Markdown 报告
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "eval_time",
    "eval_spatial",
    "eval_vehicle",
    "eval_cause",
    "eval_injury",
    "eval_context",
    "eval_all",
    "report_markdown",
]

# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个离散分布的 JS 散度（0~1）。"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # 归一化
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    # KL(P||M) + KL(Q||M) 除以 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(np.where(p > 0, p / np.where(m > 0, m, 1e-12), 1)), 0)
        kl_qm = np.where(q > 0, q * np.log(np.where(q > 0, q / np.where(m > 0, m, 1e-12), 1)), 0)
    return float(0.5 * (kl_pm.sum() + kl_qm.sum()))


def _dist_js_categorical(real_s: pd.Series, synth_s: pd.Series) -> float:
    """对类别列计算 JS 散度。"""
    cats = set(real_s.dropna().unique()) | set(synth_s.dropna().unique())
    if not cats:
        return float("nan")
    cats_sorted = sorted(cats, key=str)
    p = np.array([real_s.value_counts().get(c, 0) for c in cats_sorted], dtype=float)
    q = np.array([synth_s.value_counts().get(c, 0) for c in cats_sorted], dtype=float)
    return _js_divergence(p, q)


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein-1 距离。"""
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(x, y))
    except ImportError:
        # 手动实现
        n  = len(x)
        m  = len(y)
        combined = np.concatenate([x, y])
        combined.sort()
        cdf_x = np.searchsorted(x, combined, side="right") / n
        cdf_y = np.searchsorted(y, combined, side="right") / m
        diffs  = np.diff(combined)
        return float(np.sum(np.abs(cdf_x[:-1] - cdf_y[:-1]) * diffs))


def _binary_stats(real_s: pd.Series, synth_s: pd.Series) -> dict:
    """对 0/1 二值列计算统计。"""
    r = pd.to_numeric(real_s,  errors="coerce").dropna()
    s = pd.to_numeric(synth_s, errors="coerce").dropna()
    return {
        "real_rate":  float(r.mean()) if len(r) > 0 else float("nan"),
        "synth_rate": float(s.mean()) if len(s) > 0 else float("nan"),
        "rate_diff":  float(abs(r.mean() - s.mean())) if len(r) > 0 and len(s) > 0 else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. 时间评估
# ──────────────────────────────────────────────────────────────────────────────

TIME_BINARY_COLS = ["IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK"]


def eval_time(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    评估事故时间变量生成质量。

    评估指标：
    - SEASON 分布 JS
    - IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK 分布差异
    - 时间组合合法性（同时 AM_PEAK+PM_PEAK = 1 的比例）
    """
    result: dict = {}

    # SEASON JS
    if "SEASON" in real_df.columns and "SEASON" in synth_df.columns:
        result["season_js"] = _dist_js_categorical(real_df["SEASON"], synth_df["SEASON"])

    # 二值时间列
    for col in TIME_BINARY_COLS:
        if col in real_df.columns and col in synth_df.columns:
            result[f"{col.lower()}"] = _binary_stats(real_df[col], synth_df[col])

    # 时间组合合法性：IS_AM_PEAK + IS_PM_PEAK 不应同时为 1
    if "IS_AM_PEAK" in synth_df.columns and "IS_PM_PEAK" in synth_df.columns:
        am = pd.to_numeric(synth_df["IS_AM_PEAK"], errors="coerce").fillna(0)
        pm = pd.to_numeric(synth_df["IS_PM_PEAK"], errors="coerce").fillna(0)
        both_peak = ((am == 1) & (pm == 1)).mean()
        result["both_peak_ratio"] = float(both_peak)

    # DAY_OF_WEEK JS（若有）
    if "DAY_OF_WEEK" in real_df.columns and "DAY_OF_WEEK" in synth_df.columns:
        result["day_of_week_js"] = _dist_js_categorical(
            real_df["DAY_OF_WEEK"].astype(str), synth_df["DAY_OF_WEEK"].astype(str)
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. 空间评估
# ──────────────────────────────────────────────────────────────────────────────

def eval_spatial(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    rcs=None,  # RoadCandidateSet（可选）
    osm_type_col: str = "OSM_TYPE",
) -> dict:
    """
    评估经纬度生成质量。

    评估指标：
    - Wasserstein(lat) / Wasserstein(lon)
    - 到最近道路距离分布（需 rcs）
    - 落路率 / 偏离率
    - OSM_TYPE 分布 JS（若有）
    - SNAP_DIST_M 分布（若有）
    """
    result: dict = {}

    # ── Wasserstein ────────────────────────────────────────────────────────
    for coord in ("LATITUDE", "LONGITUDE"):
        if coord in real_df.columns and coord in synth_df.columns:
            r = real_df[coord].to_numpy(dtype=float)
            s = synth_df[coord].to_numpy(dtype=float)
            result[f"wasserstein_{coord.lower()}"] = _wasserstein_1d(r, s)

    # ── 道路距离（需 rcs）──────────────────────────────────────────────────
    if rcs is not None:
        from src.road_snap import validate_points
        if "LATITUDE" in synth_df.columns and "LONGITUDE" in synth_df.columns:
            lats_s = synth_df["LATITUDE"].to_numpy(dtype=float)
            lons_s = synth_df["LONGITUDE"].to_numpy(dtype=float)
            valid_s = np.isfinite(lats_s) & np.isfinite(lons_s)
            if valid_s.sum() > 0:
                stats_s = validate_points(lats_s[valid_s], lons_s[valid_s], rcs)
                result["road_dist_synth"] = {
                    "mean_m":    stats_s["mean_dist_m"],
                    "median_m":  stats_s["median_dist_m"],
                    "p95_m":     stats_s["p95_dist_m"],
                    "on_road_pct": stats_s["on_road_pct"],
                    "off_road_ratio": stats_s["off_road_ratio"],
                }

        if "LATITUDE" in real_df.columns and "LONGITUDE" in real_df.columns:
            lats_r = real_df["LATITUDE"].to_numpy(dtype=float)
            lons_r = real_df["LONGITUDE"].to_numpy(dtype=float)
            valid_r = np.isfinite(lats_r) & np.isfinite(lons_r)
            if valid_r.sum() > 0:
                stats_r = validate_points(lats_r[valid_r], lons_r[valid_r], rcs)
                result["road_dist_real"] = {
                    "mean_m":    stats_r["mean_dist_m"],
                    "median_m":  stats_r["median_dist_m"],
                    "p95_m":     stats_r["p95_dist_m"],
                    "on_road_pct": stats_r["on_road_pct"],
                    "off_road_ratio": stats_r["off_road_ratio"],
                }

    # ── SNAP_DIST_M（若 postprocess 后已有此列）────────────────────────────
    if "SNAP_DIST_M" in synth_df.columns:
        sd = pd.to_numeric(synth_df["SNAP_DIST_M"], errors="coerce").dropna()
        if len(sd) > 0:
            result["snap_dist_m"] = {
                "mean":   float(sd.mean()),
                "median": float(sd.median()),
                "p95":    float(sd.quantile(0.95)),
                "far_200m_ratio": float((sd > 200).mean()),
            }

    # ── OSM_TYPE 分布 ────────────────────────────────────────────────────
    if osm_type_col in real_df.columns and osm_type_col in synth_df.columns:
        result["osm_type_js"] = _dist_js_categorical(
            real_df[osm_type_col], synth_df[osm_type_col]
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. 车辆类型评估
# ──────────────────────────────────────────────────────────────────────────────

VEHICLE_COLS = [
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
    "is_bus", "is_van", "is_motorcycle", "is_bicycle", "is_emergency",
]


def eval_vehicle(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    评估车辆类型生成质量。

    评估指标：
    - 每类车辆边际分布（real_rate vs synth_rate vs rate_diff）
    - 多标签 Jaccard 相似度
    - TOTAL_VEHICLES MAE（若有）
    - IS_MULTI_VEHICLE 一致性
    """
    result: dict = {}

    avail_cols = [c for c in VEHICLE_COLS if c in real_df.columns and c in synth_df.columns]

    # ── 边际分布 ──────────────────────────────────────────────────────────
    per_vehicle: Dict[str, dict] = {}
    for col in avail_cols:
        per_vehicle[col] = _binary_stats(real_df[col], synth_df[col])
    result["per_vehicle_rate"] = per_vehicle

    # ── 多标签 Jaccard（每行 intersection/union）──────────────────────────
    if len(avail_cols) >= 2:
        r_mat = real_df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(int)
        s_mat = synth_df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(int)
        # 均值分布级 Jaccard（逐列汇聚）
        intersection = np.minimum(r_mat, s_mat).sum(axis=0)
        union        = np.maximum(r_mat, s_mat).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_col_j = np.where(union > 0, intersection / union, 0.0)
        result["vehicle_jaccard_mean"] = float(per_col_j.mean())
        result["vehicle_jaccard_per"]  = {
            col: float(j) for col, j in zip(avail_cols, per_col_j)
        }

    # ── TOTAL_VEHICLES ─────────────────────────────────────────────────────
    if "TOTAL_VEHICLES" in real_df.columns and "TOTAL_VEHICLES" in synth_df.columns:
        r_tv = pd.to_numeric(real_df["TOTAL_VEHICLES"],  errors="coerce").dropna()
        s_tv = pd.to_numeric(synth_df["TOTAL_VEHICLES"], errors="coerce").dropna()
        result["total_vehicles"] = {
            "real_mean":  float(r_tv.mean()),
            "synth_mean": float(s_tv.mean()),
            "mae":        float(abs(s_tv.mean() - r_tv.mean())),
            "wasserstein": _wasserstein_1d(r_tv.values, s_tv.values),
        }

    # ── IS_MULTI_VEHICLE ──────────────────────────────────────────────────
    if "IS_MULTI_VEHICLE" in real_df.columns and "IS_MULTI_VEHICLE" in synth_df.columns:
        result["is_multi_vehicle"] = _binary_stats(
            real_df["IS_MULTI_VEHICLE"], synth_df["IS_MULTI_VEHICLE"]
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. 事故原因评估
# ──────────────────────────────────────────────────────────────────────────────

CAUSE_COLS = [
    "is_distracted", "is_speeding", "is_failure_to_yield",
    "is_following_too_closely", "is_drunk_driving", "is_fatigue",
    "is_view_obstructed", "is_vehicle_defect", "is_backing_unsafely",
    "is_pedestrian_related", "is_inexperience", "is_pavement_slippery",
]

# 额外的事故类型列（1.4 中新增）
CRASH_TYPE_COLS = [
    "is_rear_end", "is_lane_change_related",
    "is_pedestrian_involved", "is_cyclist_involved",
]


def eval_cause(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    评估事故原因/类型生成质量。

    评估指标：
    - 每类原因边际分布差异
    - 原因共现矩阵 Frobenius 距离
    - 原因-车辆类型联合分布 JS
    """
    result: dict = {}

    all_cause_cols = CAUSE_COLS + CRASH_TYPE_COLS
    avail_cols = [c for c in all_cause_cols
                  if c in real_df.columns and c in synth_df.columns]

    # ── 边际分布 ──────────────────────────────────────────────────────────
    per_cause: Dict[str, dict] = {}
    for col in avail_cols:
        per_cause[col] = _binary_stats(real_df[col], synth_df[col])
    result["per_cause_rate"] = per_cause

    # ── 共现矩阵 Frobenius 距离 ───────────────────────────────────────────
    if len(avail_cols) >= 2:
        r_mat = real_df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        s_mat = synth_df[avail_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

        # 归一化行数后的共现矩阵
        def _cooccur(mat):
            m = mat.T @ mat
            n = max(len(mat), 1)
            return m / n

        r_co = _cooccur(r_mat)
        s_co = _cooccur(s_mat)
        frob = float(np.linalg.norm(r_co - s_co, "fro"))
        result["cause_cooccurrence_frobenius"] = frob

    # ── 原因-车辆类型联合分布 JS ────────────────────────────────────────
    veh_avail = [c for c in VEHICLE_COLS if c in real_df.columns and c in synth_df.columns]
    if veh_avail and avail_cols:
        # 简化：计算各 cause × vehicle 联合频率的 JS
        joint_js_vals = []
        for cause in avail_cols[:6]:  # 取前 6 个原因避免计算量过大
            for veh in veh_avail[:4]:
                real_joint  = real_df[[cause, veh]].apply(pd.to_numeric, errors="coerce")
                synth_joint = synth_df[[cause, veh]].apply(pd.to_numeric, errors="coerce")
                # 4 种组合 (0,0),(0,1),(1,0),(1,1)
                combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
                p = np.array([
                    ((real_joint[cause] == c) & (real_joint[veh] == v)).mean()
                    for c, v in combos
                ], dtype=float)
                q = np.array([
                    ((synth_joint[cause] == c) & (synth_joint[veh] == v)).mean()
                    for c, v in combos
                ], dtype=float)
                js = _js_divergence(p, q)
                if not np.isnan(js):
                    joint_js_vals.append(js)

        if joint_js_vals:
            result["cause_vehicle_joint_js_mean"] = float(np.mean(joint_js_vals))
            result["cause_vehicle_joint_js_max"]  = float(np.max(joint_js_vals))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. 伤亡评估
# ──────────────────────────────────────────────────────────────────────────────

INJURY_BIN_COLS = [
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",     "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",    "NUMBER_OF_MOTORIST_KILLED_BIN",
]
INJURY_COUNT_COL = "NUMBER OF PERSONS INJURED"


def eval_injury(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    """
    评估伤亡类别与人数生成质量。

    评估指标：
    - 每个 bin 列分布 JS
    - 伤亡人数 Wasserstein
    - zero_ratio
    - count MAE / calibration（均值 & 分位数比较）
    """
    result: dict = {}

    # ── Bin 列 JS ─────────────────────────────────────────────────────────
    bin_js: Dict[str, float] = {}
    for col in INJURY_BIN_COLS:
        if col in real_df.columns and col in synth_df.columns:
            bin_js[col] = _dist_js_categorical(
                real_df[col].astype(str), synth_df[col].astype(str)
            )
    result["injury_bin_js"] = bin_js
    if bin_js:
        result["injury_bin_js_mean"] = float(np.nanmean(list(bin_js.values())))

    # ── 伤亡人数 ──────────────────────────────────────────────────────────
    if INJURY_COUNT_COL in real_df.columns and INJURY_COUNT_COL in synth_df.columns:
        r_cnt = pd.to_numeric(real_df[INJURY_COUNT_COL],  errors="coerce").dropna()
        s_cnt = pd.to_numeric(synth_df[INJURY_COUNT_COL], errors="coerce").dropna()

        # Clamp 负值
        r_cnt = r_cnt.clip(lower=0)
        s_cnt = s_cnt.clip(lower=0)

        r_zero = float((r_cnt == 0).mean())
        s_zero = float((s_cnt == 0).mean())

        result["injury_count"] = {
            "real_mean":       float(r_cnt.mean()),
            "synth_mean":      float(s_cnt.mean()),
            "real_zero_ratio": r_zero,
            "synth_zero_ratio": s_zero,
            "zero_ratio_diff": abs(r_zero - s_zero),
            "mae":             float(abs(s_cnt.mean() - r_cnt.mean())),
            "wasserstein":     _wasserstein_1d(r_cnt.values, s_cnt.values),
            "real_p25":        float(r_cnt.quantile(0.25)),
            "real_p50":        float(r_cnt.quantile(0.50)),
            "real_p75":        float(r_cnt.quantile(0.75)),
            "synth_p25":       float(s_cnt.quantile(0.25)),
            "synth_p50":       float(s_cnt.quantile(0.50)),
            "synth_p75":       float(s_cnt.quantile(0.75)),
            # 负值率（合理性检查）
            "neg_ratio":       float((synth_df[INJURY_COUNT_COL].to_numpy(dtype=float) < 0).mean())
                               if INJURY_COUNT_COL in synth_df.columns else 0.0,
        }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. 路网/天气上下文评估
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_COLS    = ["TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION"]
OSM_ATTR_COLS   = ["DIST_TO_SIGNAL_M", "INFERRED_LANES", "OSM_TYPE", "OSM_ONEWAY", "HAS_TRAFFIC_SIGNAL"]


def eval_context(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    mode: str = "historical_lookup",
) -> dict:
    """
    评估路网/天气上下文变量质量。

    historical_lookup: 报告 lookup 覆盖率、缺失率
    future_simulation: 报告天气物理一致性
    correction: 报告生成上下文与真实 lookup 的误差（若有 _GEN 列）
    """
    result: dict = {"mode": mode}

    # ── 天气列覆盖率 ─────────────────────────────────────────────────────
    weather_coverage: Dict[str, float] = {}
    for col in WEATHER_COLS:
        if col in synth_df.columns:
            n_missing = synth_df[col].isna().sum()
            if col == "WEATHER_CONDITION":
                n_missing += (synth_df[col] == "Unknown").sum()
            weather_coverage[col] = float(1.0 - n_missing / max(len(synth_df), 1))
    result["weather_coverage"] = weather_coverage

    # ── historical_lookup：分布对比 ────────────────────────────────────
    if mode == "historical_lookup":
        for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH"]:
            if col in real_df.columns and col in synth_df.columns:
                r = pd.to_numeric(real_df[col],  errors="coerce").dropna().values
                s = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
                result[f"{col}_wasserstein"] = _wasserstein_1d(r, s)
        if "WEATHER_CONDITION" in real_df.columns and "WEATHER_CONDITION" in synth_df.columns:
            result["weather_condition_js"] = _dist_js_categorical(
                real_df["WEATHER_CONDITION"], synth_df["WEATHER_CONDITION"]
            )

    # ── correction：calibration 误差 ─────────────────────────────────────
    if mode == "correction":
        calib: Dict[str, dict] = {}
        for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH"]:
            gen_col = f"{col}_GEN"
            if gen_col in synth_df.columns and col in synth_df.columns:
                gen_v  = pd.to_numeric(synth_df[gen_col], errors="coerce")
                real_v = pd.to_numeric(synth_df[col],     errors="coerce")
                valid  = gen_v.notna() & real_v.notna()
                if valid.sum() > 0:
                    diff = (gen_v - real_v)[valid]
                    calib[col] = {
                        "mae":  float(diff.abs().mean()),
                        "bias": float(diff.mean()),
                        "rmse": float(np.sqrt((diff ** 2).mean())),
                    }
        result["correction_calib"] = calib

    # ── future_simulation：天气物理一致性 ─────────────────────────────────
    if mode == "future_simulation":
        phys: Dict[str, float] = {}
        if "WEATHER_CONDITION" in synth_df.columns and "TEMP_C" in synth_df.columns:
            snow_mask  = synth_df["WEATHER_CONDITION"].str.contains("Snow", na=False)
            rain_mask  = synth_df["WEATHER_CONDITION"].str.contains("Rain|Shower|Drizzle", na=False)
            temp_s     = pd.to_numeric(synth_df["TEMP_C"], errors="coerce")
            # Snow 时温度应 <= 5°C
            if snow_mask.sum() > 0:
                phys["snow_low_temp_pct"] = float((temp_s[snow_mask] <= 5).mean())
        if "prcp" in synth_df.columns and "WEATHER_CONDITION" in synth_df.columns:
            prcp_s = pd.to_numeric(synth_df["prcp"], errors="coerce").fillna(0)
            clear  = synth_df["WEATHER_CONDITION"].str.contains("Clear", na=False)
            if clear.sum() > 0:
                phys["clear_zero_prcp_pct"] = float((prcp_s[clear] < 0.1).mean())
        result["weather_physics"] = phys

    # ── OSM 属性分布 ─────────────────────────────────────────────────────
    for col in ["DIST_TO_SIGNAL_M", "INFERRED_LANES"]:
        if col in real_df.columns and col in synth_df.columns:
            r = pd.to_numeric(real_df[col],  errors="coerce").dropna().values
            s = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
            result[f"{col}_wasserstein"] = _wasserstein_1d(r, s)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 7. 综合评估入口
# ──────────────────────────────────────────────────────────────────────────────

def eval_all(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    rcs=None,
    context_mode: str = "historical_lookup",
) -> dict:
    """
    运行所有五大类 + 上下文评估，返回汇总字典。
    """
    return {
        "time":    eval_time(real_df, synth_df),
        "spatial": eval_spatial(real_df, synth_df, rcs=rcs),
        "vehicle": eval_vehicle(real_df, synth_df),
        "cause":   eval_cause(real_df, synth_df),
        "injury":  eval_injury(real_df, synth_df),
        "context": eval_context(real_df, synth_df, mode=context_mode),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Markdown 报告生成
# ──────────────────────────────────────────────────────────────────────────────

def report_markdown(metrics: dict, title: str = "Structured Evaluation Report") -> str:
    """将 eval_all 输出的 dict 渲染为 Markdown 报告字符串。"""
    lines = [f"# {title}\n"]

    def _section(name: str, data: dict):
        lines.append(f"## {name}\n")
        _render_dict(data, indent=0)
        lines.append("")

    def _render_dict(d: dict, indent: int):
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}- **{k}**:")
                _render_dict(v, indent + 1)
            elif isinstance(v, float):
                lines.append(f"{prefix}- {k}: `{v:.4f}`")
            else:
                lines.append(f"{prefix}- {k}: `{v}`")

    section_names = {
        "time":    "1. 时间评估 (TimeDiff)",
        "spatial": "2. 空间评估 (GeoRoadDiff)",
        "vehicle": "3. 车辆类型评估 (VehicleMultiLabel)",
        "cause":   "4. 事故原因评估 (CauseMultiLabel)",
        "injury":  "5. 伤亡评估 (InjuryCountSeverity)",
        "context": "6. 路网/天气上下文评估 (Context)",
    }

    for key, sec_name in section_names.items():
        if key in metrics:
            _section(sec_name, metrics[key])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI：对两个 CSV 执行完整评估
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    p = argparse.ArgumentParser(description="五大语义变量专项评估")
    p.add_argument("--real_csv",   required=True)
    p.add_argument("--synth_csv",  required=True)
    p.add_argument("--osm_graphml", default=None, help="OSM GraphML（可选，用于空间道路距离评估）")
    p.add_argument("--context_mode", default="historical_lookup",
                   choices=["historical_lookup", "future_simulation", "correction"])
    p.add_argument("--out_json",   default=None, help="指标输出 JSON 路径")
    p.add_argument("--out_md",     default=None, help="Markdown 报告输出路径")
    args = p.parse_args()

    real_df  = pd.read_csv(args.real_csv)
    synth_df = pd.read_csv(args.synth_csv)

    rcs = None
    if args.osm_graphml:
        from src.road_snap import build_road_candidate_set
        rcs = build_road_candidate_set(args.osm_graphml)

    metrics = eval_all(real_df, synth_df, rcs=rcs, context_mode=args.context_mode)

    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"[structured_eval] 已保存 JSON: {args.out_json}")

    if args.out_md:
        md = report_markdown(metrics)
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[structured_eval] 已保存 Markdown: {args.out_md}")
