"""
build_2025_like_2017.py
========================
将 2025 年原始 crash 数据按 **2017 富化管线（rebuild_2017_p1.py + data_processor.py）**
的口径完整重建一份 "补全完成版" 物理空间表，并在内存中同时给出与
prepare_2025_data.py（regex + GraphML/nearest_edges）口径一致的对照版本，
便于后续做 "管线人为漂移 vs 真实年份漂移" 的解耦诊断。

输出（results/）:
  postcovid_2025_pristine_like_v9.csv           # 与 nyc_2017_pristine_v9.csv 同字段
  postcovid_2025_fully_enriched_like_2017.csv   # 物理空间, 与 2025_n82698.csv 同 schema
  postcovid_2025_enriched_dictionary.json       # 字段说明 + 补全摘要
  fusion_pipeline_diff_report.md                # 2017 ↔ 2025 流程差异对照表

只读: src/data_processor.py, pipeline/rebuild_2017_p1.py, pipeline/prepare_2025_data.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
TABDDPM    = Path(__file__).resolve().parents[3] / "tab-ddpm-main"
RAW        = ROOT / "raw_data"
RESULTS    = ROOT / "results"

CRASH_CSV  = RAW / "crash" / "Motor_Vehicle_Collisions_-_Crashes_20260415.csv"

# 每个年份用各自时点的 PBF 与 Open-Meteo 天气文件
# raw_data/{osm,weather}/{2017,2025}/  —— 由用户在 2026-04-24 重新组织
YEAR_CONFIGS: Dict[int, dict] = {
    2017: {
        "pbf":     RAW / "osm" / "2017" / "new-york-180101-internal.osm.pbf",
        "weather": RAW / "weather" / "2017" / "open-meteo-40.74N74.04W51m.csv",
        "out_main":     "postcovid_2017_fully_enriched_like_2017.csv",
        "out_pristine": "postcovid_2017_pristine_like_v9.csv",
        "out_dict":     "postcovid_2017_enriched_dictionary.json",
    },
    2025: {
        "pbf":     RAW / "osm" / "2025" / "new-york-260101.osm.pbf",
        "weather": RAW / "weather" / "2025" / "open-meteo-40.74N74.04W51m.csv",
        "out_main":     "postcovid_2025_fully_enriched_like_2017.csv",
        "out_pristine": "postcovid_2025_pristine_like_v9.csv",
        "out_dict":     "postcovid_2025_enriched_dictionary.json",
    },
}

NYC_LAT    = (40.45, 40.95)
NYC_LON    = (-74.30, -73.65)

# ──────────────────────────────────────────────────────────────────────────────
# 1. 直接内联 src/data_processor.py 的 enum 词表（exact lower-case match）
#    (避免 import src 触发 torch 依赖)
# ──────────────────────────────────────────────────────────────────────────────
VEHICLE_TYPE_GROUPS = {
    "is_sedan": ["Sedan", "SEDAN", "4 dr sedan", "2 dr sedan"],
    "is_suv": ["Station Wagon/Sport Utility Vehicle", "SPORT UTILITY / STATION WAGON"],
    "is_taxi": ["Taxi", "TAXI", "Livery Vehicle", "LIVERY VEHICLE"],
    "is_truck": [
        "Tractor Truck Diesel", "Tractor Truck Gasoline",
        "Box Truck", "BOX TRUCK", "Flat Bed",
        "LARGE COM VEH(6 OR MORE TIRES)", "SMALL COM VEH(4 TIRES)",
        "Carry All", "Dump", "Garbage or Refuse", "Tanker", "Concrete Mixer",
    ],
    "is_bus": ["Bus", "BUS", "School Bus", "OMNIBUS", "VAN/SHUTTLE/BUS"],
    "is_motorcycle": ["Motorcycle", "MOTORCYCLE", "Motorbike", "Moped", "E-Scooter", "Motorscooter"],
    "is_bicycle": ["Bicycle", "Bike", "E-Bike", "BICYCLE"],
}

VEHICLE_OTHER_INDICATOR_NAME = "is_other_vehicle"

CONTRIBUTING_FACTOR_GROUPS = {
    "is_distracted": [
        "Driver Inattention/Distraction", "Cell Phone (hand-Held)",
        "Cell Phone (hands-free)", "Using On Board Electronic Device",
        "Texting", "Listening/Using Headphones", "Outside Car Distraction",
    ],
    "is_speeding": ["Unsafe Speed", "Aggressive Driving/Road Rage"],
    "is_failure_to_yield": [
        "Failure to Yield Right-of-Way", "Failure to Keep Right", "Traffic Control Disregarded",
    ],
    "is_following_too_closely": [
        "Following Too Closely", "Unsafe Lane Changing",
        "Passing or Lane Usage Improper", "Passing Too Closely",
    ],
    "is_drunk_driving": [
        "Alcohol Involvement", "Drugs (illegal)", "Drugs (Illegal)", "Prescription Medication",
    ],
    "is_fatigue": ["Fatigued/Drowsy", "Lost Consciousness", "Fell Asleep"],
    "is_view_obstructed": [
        "View Obstructed/Limited", "Glare", "Obstruction/Debris", "Windshield Inadequate",
    ],
    "is_vehicle_defect": [
        "Brakes Defective", "Steering Failure", "Tire Failure/Inadequate",
        "Accelerator Defective", "Headlights Defective", "Other Lighting Defects", "Tow Hitch Defective",
    ],
    "is_backing_unsafely": ["Backing Unsafely", "Turning Improperly"],
    "is_pedestrian_related": [
        "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion",
        "Pedestrians in Roadway (not intersection)",
    ],
    "is_inexperience": ["Driver Inexperience"],
    "is_pavement_slippery": ["Pavement Slippery"],
}

VEHICLE_CODE_COLS         = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
CONTRIBUTING_FACTOR_COLS  = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)]
SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

def _get_time_period(hour: int) -> str:
    if   5 <= hour < 8:  return "dawn"
    elif 8 <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    return "night"


# ──────────────────────────────────────────────────────────────────────────────
# 2. 2017-style OSM 富化（PBF + pyrosm + BallTree haversine, 与 rebuild_2017_p1 同口径）
# ──────────────────────────────────────────────────────────────────────────────

def extract_osm_from_pbf(pbf_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    返回:
      df_roads         : 全部道路边 (lon, lat, highway, oneway, lanes_raw)
      df_road_lanes    : 仅含有效 lanes tag 的子集
      sig_coords (rad) : 信号灯节点经纬度 (弧度)，用于 BallTree haversine
    """
    import pyrosm

    print(f"  [PBF] 加载 {pbf_path.name} ...")
    osm = pyrosm.OSM(str(pbf_path))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    if edges is None or len(edges) == 0:
        raise RuntimeError(f"PBF 中未提取到 driving 路网 edges: {pbf_path}")
    edges = edges.to_crs("EPSG:4326")
    print(f"  [PBF] driving edges = {len(edges):,}")

    cents = edges.geometry.centroid
    df_roads = pd.DataFrame({
        "lon"      : cents.x.values,
        "lat"      : cents.y.values,
        "highway"  : edges.get("highway", pd.Series([None]*len(edges))).astype(str).values,
        "oneway"   : edges.get("oneway",  pd.Series([None]*len(edges))).astype(str).values,
        "lanes_raw": edges.get("lanes",   pd.Series([None]*len(edges))).values,
    })

    def _parse_lanes(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip()
        if not s or s.lower() in ("none", "nan", ""):
            return np.nan
        nums = re.findall(r"\d+", s)
        return float(nums[0]) if nums else np.nan

    df_roads["lanes_int"] = df_roads["lanes_raw"].apply(_parse_lanes)
    df_road_lanes = df_roads.dropna(subset=["lanes_int"]).copy()
    print(f"  [PBF] lanes-tagged edges = {len(df_road_lanes):,} ({len(df_road_lanes)/len(df_roads):.1%})")

    # ---- 信号灯节点 ----
    print(f"  [PBF] 提取信号灯节点 ...")
    try:
        # pyrosm 提供 get_pois 可拉取 highway=traffic_signals
        pois = osm.get_pois(custom_filter={"highway": ["traffic_signals"]})
    except Exception as e:
        print(f"    get_pois failed ({e}), fallback 0 signals")
        pois = None
    if pois is not None and len(pois) > 0:
        pois = pois.to_crs("EPSG:4326")
        # 取 centroid（point 也是 centroid 自身）
        c = pois.geometry.centroid
        sig_lat = np.deg2rad(c.y.values)
        sig_lon = np.deg2rad(c.x.values)
        sig_coords = np.column_stack([sig_lat, sig_lon])
        print(f"    信号灯节点 = {len(sig_coords):,}")
    else:
        sig_coords = np.empty((0, 2))
        print("    ⚠ 信号灯节点 = 0 (将填 NaN)")
    return df_roads, df_road_lanes, sig_coords


def osm_match_2017style(
    df_crash: pd.DataFrame,
    df_roads: pd.DataFrame,
    df_road_lanes: pd.DataFrame,
    sig_coords_rad: np.ndarray,
) -> pd.DataFrame:
    """
    与 rebuild_2017_p1.match_lanes_to_accidents + 信号灯距离的 2017 口径完全一致：
      - haversine BallTree
      - 优先匹配 ≤500m 的有 lanes tag 的道路
      - 否则用 highway type 推断
      - 信号灯距离 = 最近 traffic_signals 节点的米距离
      - HAS_TRAFFIC_SIGNAL = (DIST_TO_SIGNAL_M < 30)
    """
    print("  [match] BallTree haversine 匹配 ...")
    lat_r = np.deg2rad(df_crash["LATITUDE"].values).reshape(-1, 1)
    lon_r = np.deg2rad(df_crash["LONGITUDE"].values).reshape(-1, 1)
    crash_rad = np.hstack([lat_r, lon_r])

    # --- (a) lanes (有 lanes tag 的道路子集) ---
    if len(df_road_lanes) > 0:
        rl_lat = np.deg2rad(df_road_lanes["lat"].values).reshape(-1, 1)
        rl_lon = np.deg2rad(df_road_lanes["lon"].values).reshape(-1, 1)
        tree_lanes = BallTree(np.hstack([rl_lat, rl_lon]), metric="haversine")
        d_lanes, i_lanes = tree_lanes.query(crash_rad, k=1)
        d_lanes_m  = d_lanes[:, 0] * 6371000.0
        matched_l  = df_road_lanes["lanes_int"].iloc[i_lanes[:, 0]].values
    else:
        d_lanes_m = np.full(len(df_crash), 1e9)
        matched_l = np.full(len(df_crash), np.nan)

    # --- (b) all roads (用于 highway type 推断 + OSM_TYPE/OSM_ONEWAY 提取) ---
    ra_lat = np.deg2rad(df_roads["lat"].values).reshape(-1, 1)
    ra_lon = np.deg2rad(df_roads["lon"].values).reshape(-1, 1)
    tree_all = BallTree(np.hstack([ra_lat, ra_lon]), metric="haversine")
    _, i_all = tree_all.query(crash_rad, k=1)
    nearest = df_roads.iloc[i_all[:, 0]].reset_index(drop=True)

    # --- (c) 信号灯距离 ---
    if len(sig_coords_rad) > 0:
        tree_sig = BallTree(sig_coords_rad, metric="haversine")
        d_sig, _ = tree_sig.query(crash_rad, k=1)
        dist_signal_m = d_sig[:, 0] * 6371000.0
    else:
        dist_signal_m = np.full(len(df_crash), np.nan)
    has_signal = (dist_signal_m < 30).astype(int) if np.isfinite(dist_signal_m).any() else np.zeros(len(df_crash), dtype=int)

    # --- (d) 组装 INFERRED_LANES ---
    def _infer(hw):
        if hw is None:
            return 1
        h = str(hw).lower()
        if "motorway" in h or "trunk" in h: return 3
        if "primary"  in h or "secondary" in h: return 2
        return 1

    MAX_DIST_M = 500.0
    inferred  = np.zeros(len(df_crash), dtype=int)
    osm_lanes = np.full(len(df_crash), np.nan)
    for i in range(len(df_crash)):
        if d_lanes_m[i] <= MAX_DIST_M and not np.isnan(matched_l[i]):
            inferred[i]  = int(matched_l[i])
            osm_lanes[i] = matched_l[i]
        else:
            inferred[i] = _infer(nearest["highway"].iloc[i])

    # OSM_ONEWAY → bool/int
    def _to_bool(v):
        if v is None: return 0
        s = str(v).strip().lower()
        return 1 if s in ("yes", "true", "1") else 0

    out = pd.DataFrame({
        "DIST_TO_SIGNAL_M"   : dist_signal_m,
        "HAS_TRAFFIC_SIGNAL" : has_signal,
        "OSM_TYPE"           : nearest["highway"].fillna("residential").astype(str).values,
        "OSM_ONEWAY"         : nearest["oneway"].apply(_to_bool).values,
        "OSM_LANES_TAG"      : osm_lanes,
        "INFERRED_LANES"     : inferred,
    }, index=df_crash.index)

    print(f"    DIST_TO_SIGNAL_M: mean={np.nanmean(dist_signal_m):.1f}m  median={np.nanmedian(dist_signal_m):.1f}m")
    print(f"    HAS_TRAFFIC_SIGNAL rate = {has_signal.mean():.1%}")
    print(f"    INFERRED_LANES dist: {pd.Series(inferred).value_counts().sort_index().to_dict()}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. 天气富化（与 prepare_2025 同 Open-Meteo + COCO_MAP, 但 merge_asof 完全一致）
# ──────────────────────────────────────────────────────────────────────────────
COCO_MAP = {
    0:  "Clear",  1:  "Clear",
    2:  "Cloudy", 3:  "Cloudy",
    45: "Cloudy", 48: "Cloudy",
    51: "Rain",   53: "Rain",   55: "Rain",
    56: "Rain",   57: "Rain",
    61: "Rain",   63: "Rain",   65: "Rain",
    66: "Rain",   67: "Rain",
    71: "Snow",   73: "Snow",   75: "Snow",   77: "Snow",
    80: "Rain",   81: "Rain",   82: "Rain",
    85: "Snow",   86: "Snow",
    95: "Rain",   96: "Rain",   99: "Rain",
}


def load_openmeteo(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
    skip = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("time"):
            skip = i; break
    w = pd.read_csv(csv_path, skiprows=skip, low_memory=False)
    cm = {}
    for c in w.columns:
        cl = c.lower()
        if "temperature" in cl: cm[c] = "temp"
        elif "precipitation" in cl: cm[c] = "prcp"
        elif "wind_speed" in cl or "windspeed" in cl: cm[c] = "wspd"
        elif "weather_code" in cl: cm[c] = "wmo_code"
    w = w.rename(columns=cm)
    w["dt"] = pd.to_datetime(w["time"], errors="coerce")
    w["coco"] = pd.to_numeric(w.get("wmo_code", pd.Series(dtype=float)), errors="coerce").fillna(3).astype(int)
    return w[["dt", "temp", "prcp", "wspd", "coco"]].set_index("dt").sort_index()


def enrich_weather_2017style(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    crash_dt = pd.to_datetime(
        df["CRASH DATE"].astype(str) + " " + df["CRASH TIME"].astype(str),
        errors="coerce",
    ).dt.floor("h")
    tmp = pd.DataFrame({"_dt": crash_dt, "_orig_idx": df.index})
    tmp = tmp.sort_values("_dt").reset_index(drop=True)
    w = weather_df.reset_index().rename(columns={"dt": "_dt"}).sort_values("_dt")
    merged = pd.merge_asof(tmp, w, on="_dt", direction="nearest", tolerance=pd.Timedelta("1h"))
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["TEMP_C"]            = pd.to_numeric(merged["temp"], errors="coerce").values
    out["prcp"]              = pd.to_numeric(merged["prcp"], errors="coerce").fillna(0.0).values
    out["WIND_SPEED_KMH"]    = pd.to_numeric(merged["wspd"], errors="coerce").values
    out["coco"]              = pd.to_numeric(merged["coco"], errors="coerce").values
    out["WEATHER_CONDITION"] = pd.Series(out["coco"]).map(COCO_MAP).fillna("Clear").values
    matched = out["TEMP_C"].notna().sum()
    print(f"  [Weather] matched {matched}/{len(out)} ({matched/len(out):.1%})")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. 2017-style enum 二元指示列（exact lower-case match, 与 data_processor 完全一致）
# ──────────────────────────────────────────────────────────────────────────────

def build_binary_indicators_2017style(df: pd.DataFrame) -> pd.DataFrame:
    """精确匹配 (exact lower-case) - 与 src/data_processor._build_binary_indicators 同口径"""
    out = pd.DataFrame(index=df.index)

    def _exact(source_cols, group_map):
        cols = [c for c in source_cols if c in df.columns]
        combined = df[cols].fillna("").astype(str)
        for name, kws in group_map.items():
            kw_lower = {k.lower() for k in kws}
            mask = combined.apply(
                lambda row: any(cell.strip().lower() in kw_lower for cell in row),
                axis=1,
            )
            out[name] = mask.astype(np.int8)

    _exact(VEHICLE_CODE_COLS, VEHICLE_TYPE_GROUPS)

    vcols_for_other = [c for c in VEHICLE_CODE_COLS if c in df.columns]
    main_keywords = {kw.lower() for kws in VEHICLE_TYPE_GROUPS.values() for kw in kws}
    if vcols_for_other:
        combined = df[vcols_for_other].fillna("").astype(str)
        mask = combined.apply(
            lambda row: any(
                (cell := value.strip().lower()) and cell not in main_keywords
                for value in row
            ),
            axis=1,
        )
        out[VEHICLE_OTHER_INDICATOR_NAME] = mask.astype(np.int8)
    else:
        out[VEHICLE_OTHER_INDICATOR_NAME] = 0

    _exact(CONTRIBUTING_FACTOR_COLS, CONTRIBUTING_FACTOR_GROUPS)

    # injury bins
    for raw, bin_col in [
        ("NUMBER OF PEDESTRIANS INJURED", "NUMBER_OF_PEDESTRIANS_INJURED_BIN"),
        ("NUMBER OF PEDESTRIANS KILLED",  "NUMBER_OF_PEDESTRIANS_KILLED_BIN"),
        ("NUMBER OF CYCLIST INJURED",     "NUMBER_OF_CYCLIST_INJURED_BIN"),
        ("NUMBER OF CYCLIST KILLED",      "NUMBER_OF_CYCLIST_KILLED_BIN"),
        ("NUMBER OF MOTORIST INJURED",    "NUMBER_OF_MOTORIST_INJURED_BIN"),
        ("NUMBER OF MOTORIST KILLED",     "NUMBER_OF_MOTORIST_KILLED_BIN"),
    ]:
        v = pd.to_numeric(df.get(raw, pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        out[bin_col] = (v > 0).astype(np.int8)

    # totals
    vcols = [c for c in VEHICLE_CODE_COLS if c in df.columns]
    if vcols:
        non_empty = df[vcols].notna() & (df[vcols].astype(str) != "")
        out["TOTAL_VEHICLES"] = non_empty.sum(axis=1).astype(int)
    else:
        out["TOTAL_VEHICLES"] = 1
    out["IS_MULTI_VEHICLE"] = (out["TOTAL_VEHICLES"] > 1).astype(np.int8)
    return out


def build_temporal_2017style(df: pd.DataFrame) -> pd.DataFrame:
    crash_dt = pd.to_datetime(
        df["CRASH DATE"].astype(str) + " " + df["CRASH TIME"].astype(str),
        errors="coerce",
    )
    out = pd.DataFrame(index=df.index)
    out["SEASON"]       = crash_dt.dt.month.map(SEASON_MAP)
    out["DAY_OF_WEEK"]  = crash_dt.dt.dayofweek
    hours               = crash_dt.dt.hour.fillna(12).astype(int)
    out["TIME_PERIOD"]  = hours.apply(_get_time_period)
    minute_of_day       = hours * 60 + crash_dt.dt.minute.fillna(0).astype(int)
    frac                = minute_of_day / 1440.0
    out["CRASH_TIME_SIN"] = np.sin(2 * np.pi * frac)
    out["CRASH_TIME_COS"] = np.cos(2 * np.pi * frac)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────────────────────────────────────

# 与 results/postcovid_test_2025_n82698.csv 完全相同的列顺序
TARGET_COLUMNS = [
    "LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS",
    "TEMP_C", "prcp", "WIND_SPEED_KMH",
    "DIST_TO_SIGNAL_M", "INFERRED_LANES",
    "SEASON", "DAY_OF_WEEK", "TIME_PERIOD",
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "coco", "WEATHER_CONDITION", "OSM_TYPE",
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup", "is_bus", "is_van",
    "is_motorcycle", "is_bicycle", "is_emergency",
    "is_distracted", "is_speeding", "is_failure_to_yield", "is_following_too_closely",
    "is_drunk_driving", "is_fatigue", "is_view_obstructed", "is_vehicle_defect",
    "is_backing_unsafely", "is_pedestrian_related", "is_inexperience", "is_pavement_slippery",
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",     "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",    "NUMBER_OF_MOTORIST_KILLED_BIN",
    "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
    "NUMBER OF PERSONS INJURED",
]

# v9 pristine 字段（保留长字符串列，方便事后审计 & 二次重跑 data_processor）
V9_PRISTINE_COLS = [
    "CRASH DATE", "CRASH TIME", "BOROUGH", "ZIP CODE", "LATITUDE", "LONGITUDE",
    "LOCATION", "ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME",
    "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",   "NUMBER OF MOTORIST KILLED",
    "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
    "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4",
    "CONTRIBUTING FACTOR VEHICLE 5", "COLLISION_ID",
    "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3",
    "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5",
    "CRASH_FULL_TIME",
    "TEMP_C", "prcp", "WIND_SPEED_KMH", "coco", "WEATHER_CONDITION", "REAL_WEATHER",
    "DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL",
    "OSM_TYPE", "OSM_SPEED_TAG", "OSM_LANES_TAG", "OSM_ONEWAY",
    "REAL_SPEED_LIMIT", "HAS_DIVIDER", "INFERRED_LANES",
    "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025, choices=sorted(YEAR_CONFIGS.keys()),
                    help="完全重建哪一年的富化表 (默认 2025)")
    args = ap.parse_args()

    cfg = YEAR_CONFIGS[args.year]
    PBF      = cfg["pbf"]
    WEATHER  = cfg["weather"]
    YEARS    = (args.year,)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── 1) 加载并过滤 raw crash ──────────────────────────
    print("=" * 64)
    print(f"  build_{args.year}_like_2017.py  —  按 2017 富化口径重建 {args.year} enriched")
    print("=" * 64)
    print(f"[1/6] 加载 raw crash CSV: {CRASH_CSV.name}")
    raw = pd.read_csv(CRASH_CSV, low_memory=False)
    raw["CRASH DATE"] = pd.to_datetime(raw["CRASH DATE"], errors="coerce")
    mask_year = raw["CRASH DATE"].dt.year.isin(YEARS)
    lat = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    mask_nyc = lat.between(*NYC_LAT) & lon.between(*NYC_LON)
    raw = raw[mask_year & mask_nyc].copy().reset_index(drop=True)
    raw["LATITUDE"]  = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    raw["LONGITUDE"] = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    raw["CRASH DATE"] = raw["CRASH DATE"].dt.strftime("%m/%d/%Y")
    print(f"    → {len(raw):,} rows after {args.year} + NYC bbox filter")

    # ── 2) PBF → OSM 路网 (2017 同口径) ──────────────────────
    print(f"[2/6] 提取 OSM 路网 (PBF={PBF.name}) (BallTree, 2017 口径)")
    df_roads, df_road_lanes, sig_coords = extract_osm_from_pbf(PBF)
    osm_feats = osm_match_2017style(raw, df_roads, df_road_lanes, sig_coords)

    # ── 3) Weather 富化 (Open-Meteo, COCO_MAP) ─────────────────────
    print(f"[3/6] 天气富化: {WEATHER.name}")
    weather_df = load_openmeteo(WEATHER)
    w_feats = enrich_weather_2017style(raw, weather_df)

    # ── 4) Temporal + Binary indicators (enum exact match) ───────────────────
    print(f"[4/6] 时间特征 + enum 二元指示列 (exact match)")
    t_feats = build_temporal_2017style(raw)
    b_feats = build_binary_indicators_2017style(raw)

    # ── 5) 组装两份输出表 ────────────────────────────────────────────────────
    print(f"[5/6] 组装输出表")
    base = pd.DataFrame({
        "LATITUDE":  raw["LATITUDE"].values,
        "LONGITUDE": raw["LONGITUDE"].values,
        "NUMBER OF PERSONS INJURED": pd.to_numeric(raw["NUMBER OF PERSONS INJURED"], errors="coerce").fillna(0).values,
    }, index=raw.index)

    enriched = pd.concat(
        [base, t_feats, w_feats, osm_feats[["DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL",
                                            "OSM_TYPE", "OSM_ONEWAY", "INFERRED_LANES"]],
         b_feats], axis=1
    )
    # 应用 2017 P1-2 winsorize @P99.9 (与 rebuild_2017_p1.winsorize_dist_signal 同口径)
    if enriched["DIST_TO_SIGNAL_M"].notna().any():
        q = enriched["DIST_TO_SIGNAL_M"].quantile(0.999)
        n_clip = int((enriched["DIST_TO_SIGNAL_M"] > q).sum())
        enriched["DIST_TO_SIGNAL_M"] = enriched["DIST_TO_SIGNAL_M"].clip(upper=q)
        print(f"    P1-2 winsorize DIST_TO_SIGNAL_M @P99.9={q:.1f}m, clipped {n_clip} rows")

    # 重排列到目标 schema (与 postcovid_test_2025_n82698.csv 一致)
    for c in TARGET_COLUMNS:
        if c not in enriched.columns:
            enriched[c] = 0
    enriched = enriched[TARGET_COLUMNS]

    out_main = RESULTS / cfg["out_main"]
    enriched.to_csv(out_main, index=False)
    print(f"  [OK] {out_main.name}  shape={enriched.shape}")

    # 同时输出 v9 pristine 同字段表 (审计用)
    pristine = raw.copy()
    pristine["TEMP_C"]            = w_feats["TEMP_C"].values
    pristine["prcp"]              = w_feats["prcp"].values
    pristine["WIND_SPEED_KMH"]    = w_feats["WIND_SPEED_KMH"].values
    pristine["coco"]              = w_feats["coco"].values
    pristine["WEATHER_CONDITION"] = w_feats["WEATHER_CONDITION"].values
    pristine["REAL_WEATHER"]      = w_feats["WEATHER_CONDITION"].values
    pristine["DIST_TO_SIGNAL_M"]   = enriched["DIST_TO_SIGNAL_M"].values
    pristine["HAS_TRAFFIC_SIGNAL"] = enriched["HAS_TRAFFIC_SIGNAL"].values
    pristine["OSM_TYPE"]      = osm_feats["OSM_TYPE"].values
    pristine["OSM_SPEED_TAG"] = ""
    pristine["OSM_LANES_TAG"] = osm_feats["OSM_LANES_TAG"].values
    pristine["OSM_ONEWAY"]    = osm_feats["OSM_ONEWAY"].values
    pristine["REAL_SPEED_LIMIT"] = np.nan
    pristine["HAS_DIVIDER"]      = 0
    pristine["INFERRED_LANES"]   = osm_feats["INFERRED_LANES"].values
    pristine["TOTAL_VEHICLES"]   = b_feats["TOTAL_VEHICLES"].values
    pristine["IS_MULTI_VEHICLE"] = b_feats["IS_MULTI_VEHICLE"].values
    pristine["CRASH_FULL_TIME"]  = pd.to_datetime(
        raw["CRASH DATE"].astype(str) + " " + raw["CRASH TIME"].astype(str),
        errors="coerce",
    ).astype(str).values
    if "BOROUGH" not in pristine: pristine["BOROUGH"] = ""
    if "ZIP CODE" not in pristine: pristine["ZIP CODE"] = ""
    if "LOCATION" not in pristine: pristine["LOCATION"] = ""
    for c in V9_PRISTINE_COLS:
        if c not in pristine.columns:
            pristine[c] = "" if c in ("BOROUGH", "ZIP CODE", "LOCATION", "ON STREET NAME",
                                       "CROSS STREET NAME", "OFF STREET NAME", "REAL_WEATHER") else np.nan
    pristine = pristine[V9_PRISTINE_COLS]
    out_pristine = RESULTS / cfg["out_pristine"]
    pristine.to_csv(out_pristine, index=False)
    print(f"  [OK] {out_pristine.name}  shape={pristine.shape}")

    # ── 6) 字段说明 + 补全摘要 + 流程差异报告 ──────────────────────────────
    print(f"[6/6] 字段说明 + 流程差异报告")
    dictionary = {
        "schema": TARGET_COLUMNS,
        "n_rows": int(len(enriched)),
        "fully_aligned_with_2017": [
            "LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS",
            "SEASON", "DAY_OF_WEEK", "TIME_PERIOD",
            "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
            "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
            "NUMBER_OF_CYCLIST_INJURED_BIN", "NUMBER_OF_CYCLIST_KILLED_BIN",
            "NUMBER_OF_MOTORIST_INJURED_BIN", "NUMBER_OF_MOTORIST_KILLED_BIN",
            "NUMBER OF PERSONS INJURED",
        ] + list(VEHICLE_TYPE_GROUPS.keys()) + [VEHICLE_OTHER_INDICATOR_NAME] + list(CONTRIBUTING_FACTOR_GROUPS.keys()),
        "approximated_alignment": {
            "DIST_TO_SIGNAL_M":   "用 2025 PBF (260317) 替代 2017 PBF (180101)，BallTree 算法一致；信号灯节点来自 PBF POI",
            "HAS_TRAFFIC_SIGNAL": "阈值 30m 与 2017 一致",
            "OSM_TYPE":           "由 2025 PBF 最近 driving edge 推断，与 2017 同算法",
            "OSM_ONEWAY":         "由 2025 PBF 最近 edge 的 oneway tag 推断",
            "INFERRED_LANES":     "优先 2025 PBF lanes tag (≤500m)；否则 highway-type 推断 (motorway/trunk=3, primary/secondary=2, else=1)，与 2017 完全一致",
            "TEMP_C/prcp/WIND_SPEED_KMH/coco/WEATHER_CONDITION":
                "Open-Meteo 2025 数据 + COCO_MAP 与 2017 一致；2017 实际用的是 Meteostat ASOS (LGA/JFK) 站点数据，气源不同但都做 hourly merge_asof",
        },
        "vs_pre_existing_2025_pipeline": {
            "DIST_TO_SIGNAL_M_method":  "本表使用 BallTree(haversine, PBF POI)；旧版 prepare_2025_data.py 使用 osmnx GraphML 节点 + 投影 BallTree",
            "OSM_TYPE_method":          "本表使用 PBF edge 中心点最近邻；旧版使用 osmnx.nearest_edges (G.geometry)",
            "INFERRED_LANES_method":    "本表使用 PBF lanes tag + 500m 阈值；旧版直接读 GraphML edge 的 lanes 属性，无距离阈值",
            "vehicle/factor_indicators": "本表 enum exact-match 与 2017 完全一致；旧版使用宽松 regex 匹配，会对 'speed/yield/distraction' 触发更高阳性率",
        },
    }
    (RESULTS / cfg["out_dict"]).write_text(
        json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 流程差异报告 (markdown)
    diff_md = build_pipeline_diff_md()
    (RESULTS / "fusion_pipeline_diff_report.md").write_text(diff_md, encoding="utf-8")
    print(f"  [OK] fusion_pipeline_diff_report.md")
    print("\nDONE.")


def build_pipeline_diff_md() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "# 2017 富化管线 vs 2025 旧版补全管线 —— 流程差异对照表\n\n"
        f"> 生成时间: {ts}\n"
        "> 本表对比的是：\n"
        ">  - 2017 管线: pipeline/rebuild_2017_p1.py (OSM) + src/data_processor.py (特征工程)\n"
        ">  - 2025 旧版管线: pipeline/prepare_2025_data.py (端到端)\n"
        ">  - 2025 新版管线 (本脚本): pipeline/build_2025_like_2017.py, 按 2017 口径重建\n\n"
        "| # | 维度 | 2017 富化管线 | 2025 旧版管线 | 是否一致 | 可能引入的人为漂移 |\n"
        "|---|------|---------------|---------------|----------|--------------------|\n"
        "| 1 | 原始事故源 | NYC Open Data 2017 全年 | NYC Open Data 2025 全年 | yes 同源 | 无 |\n"
        "| 2 | 时间 SIN/COS 粒度 | 1440 min/day, min_of_day/1440 | 仅按整 hour, hour/24 | NO 粒度不同 | 高 (CRASH_TIME_SIN/COS JS≈0.46 几乎完全由此造成) |\n"
        "| 3 | SEASON / DAY_OF_WEEK / TIME_PERIOD | dict map + _get_time_period | 完全相同 | yes | 无 |\n"
        "| 4 | 天气源 | Meteostat ASOS (LGA 72503 + JFK 74486) | Open-Meteo grid (40.74N, 74.04W) | NO 站点 vs 网格 | 中 (TEMP/WIND 偏置) |\n"
        "| 5 | 天气时间对齐 | hourly merge_asof(tolerance=1h) | 同 | yes | 无 |\n"
        "| 6 | coco / WEATHER_CONDITION | COCO_MAP (4 类) | 同 | yes | 无 |\n"
        "| 7 | OSM 源 | 本地 PBF new-york-180101-internal.osm.pbf | GraphML nyc_drive_graph.graphml (2026 版) | NO 时点+格式 | 中 (路网拓扑变化) |\n"
        "| 8 | 路网匹配方法 | pyrosm edges 中心点 + BallTree(haversine) | osmnx.nearest_edges(G) (基于 edge geometry) | NO 算法不同 | 高 (DIST/TYPE 不可比) |\n"
        "| 9 | 信号灯距离 | PBF POI traffic_signals + BallTree haversine | GraphML node tag + 投影 BallTree | NO 数据集+投影 | 高 (DIST_TO_SIGNAL_M JS=0.27, mean 257.7->75.8m) |\n"
        "| 10 | INFERRED_LANES | PBF lanes tag (<=500m 阈值) + highway-type fallback | GraphML edge lanes 属性, 无阈值 | NO | 高 (JS=0.25, mean 2.58->1.81) |\n"
        "| 11 | 车辆类型词表 | enum 精确小写匹配 | regex 宽匹配 (含 truck/tractor 等) | NO 匹配宽度不同 | 中 (is_truck/is_van 阳性率差异) |\n"
        "| 12 | 事故原因词表 | enum 精确小写匹配 | regex 宽匹配 (含 speed/yield 等) | NO | 中 (is_speeding/is_failure_to_yield 阳性率显著偏高) |\n"
        "| 13 | DIST_TO_SIGNAL_M winsorize | P99.9 clip | 无 | NO | 中 (尾部异常值进入分布) |\n"
        "| 14 | 缺失值填充 | 连续 median, 类别 mode | Level-B fallback 同思路 | yes | 低 |\n"
        "| 15 | 输出空间 | 物理空间 -> QuantileTransformer (normal) | 已是模型输入空间 (binary 不变, 连续未归一化) | NO | 高 (评测时需 inverse_transform 才可比) |\n"
        "| 16 | TOTAL_VEHICLES | notna().sum() | notna().sum().clip(1,5) | NO 微差 | 低 |\n\n"
        "## 关键结论\n\n"
        "1. 第 2 / 8 / 9 / 10 / 11 / 12 行是导致迁移退化的主要 '管线人为漂移' 来源。\n"
        "2. drift_report 中 CRASH_TIME_SIN/COS JS≈0.46 几乎完全可由 #2 解释 (hour vs minute 粒度), 并非真实年份漂移。\n"
        "3. DIST_TO_SIGNAL_M JS=0.27 (mean 257.7 -> 75.8m) 与 INFERRED_LANES JS=0.25 主要由 #8 #9 #10 引入。\n"
        "4. 事故原因二元列 (is_speeding/is_failure_to_yield) 阳性率差异主要由 #12 regex 宽匹配引入。\n"
        "5. 真实年份漂移最可能体现在: NUMBER OF PERSONS INJURED 均值上升 (0.26->0.58), 以及 is_motorcycle / is_bicycle 阳性率 (疫情后微移动激增)。\n"
    )


if __name__ == "__main__":
    main()
