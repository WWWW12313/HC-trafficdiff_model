"""
2024→2025 迁移学习数据准备管线（新版字段方案）
================================================
相较旧版 prepare_2025_data.py 的主要改动：
  1. 新增 IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK，移除 DAY_OF_WEEK / TIME_PERIOD
  2. 将12个行为因子(is_distracted, is_speeding…)替换为4个宏观事故类型指示列
     (is_rear_end, is_lane_change_related, is_pedestrian_involved, is_cyclist_involved)
  3. 移除 coco（WMO 原始码），保留 WEATHER_CONDITION
  4. 移除 is_emergency（稀有类别，迁移不稳定）
  5. 从 raw_data/osm/{year}/ 和 raw_data/weather/{year}/ 读取年份对应数据
  6. 2024 训练集直接定义 schema（不依赖旧 train.csv），2025 测试集对齐 2024 schema

用法:
  # Step1: 先运行 PBF 转换（如果 raw_data/osm/2024/nyc_drive_graph.graphml 不存在）
  python pipeline/convert_pbf_to_year_graphml.py --years 2024 2025

  # Step2: 生成 2024 训练集（--n_sample -1 全量）
  python pipeline/prepare_transfer_data.py --mode train --years 2024 --n_sample -1

  # Step3: 生成 2025 测试集（使用 2024 schema 对齐）
  python pipeline/prepare_transfer_data.py --mode test --years 2025 --n_sample -1

  # 一键全量
  python pipeline/prepare_transfer_data.py --mode all --n_sample -1

输出:
  data/nyc_crash_2024/train.csv  ← 2024 训练集（80%）
  data/nyc_crash_2024/test.csv   ← 2024 域内测试集（20%）
  data/nyc_crash_2025/test.csv   ← 2025 迁移测试集（全量）
  data/nyc_crash_2024/schema.json ← 字段映射说明
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "raw_data"
DATA    = ROOT / "data"

# ──────────────────────────────────────────────────────────────────────────────
# 新版目标 Schema 定义
# ──────────────────────────────────────────────────────────────────────────────

# 连续特征（共 9 列）
CONTINUOUS_COLS: List[str] = [
    "LATITUDE", "LONGITUDE",                              # 空间锚点
    "CRASH_TIME_SIN", "CRASH_TIME_COS",                   # 时间圆周编码
    "TEMP_C", "prcp", "WIND_SPEED_KMH",                   # 天气
    "DIST_TO_SIGNAL_M", "INFERRED_LANES",                 # OSM 路网连续
]

# 离散特征（共 29 列）
CATEGORICAL_COLS: List[str] = [
    # Stage1: 稳定时间锚点 (4)
    "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK",
    # Stage2: 路网上下文 (4)
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "WEATHER_CONDITION", "OSM_TYPE",
    # Stage3: 宏观事故类型 0/1 (4)
    "is_rear_end", "is_lane_change_related",
    "is_pedestrian_involved", "is_cyclist_involved",
    # Stage3: 车辆类型 0/1 (9)
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
    "is_bus", "is_van", "is_motorcycle", "is_bicycle",
    # Stage3: 伤亡二元 (6)
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",     "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",    "NUMBER_OF_MOTORIST_KILLED_BIN",
    # Stage3: 规模 (2)
    "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
]

ALL_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS  # 38 列

# 字段映射说明（用于 schema.json）
FIELD_MAPPING = {
    "IS_WEEKEND":             "crash_dt.dayofweek >= 5",
    "IS_AM_PEAK":             "hour in [7, 8, 9]",
    "IS_PM_PEAK":             "hour in [16, 17, 18, 19]",
    "is_rear_end":            "CONTRIBUTING FACTOR: 'following too closely'",
    "is_lane_change_related": "CONTRIBUTING FACTOR: 'unsafe lane changing|passing improper'",
    "is_pedestrian_involved": "CONTRIBUTING FACTOR: 'pedestrian' OR PEDESTRIANS INJURED/KILLED > 0",
    "is_cyclist_involved":    "VEHICLE TYPE: 'bicycle' OR CYCLIST INJURED/KILLED > 0",
    "removed_fields": [
        "DAY_OF_WEEK → replaced by IS_WEEKEND",
        "TIME_PERIOD → replaced by IS_AM_PEAK / IS_PM_PEAK",
        "coco → merged into WEATHER_CONDITION",
        "is_distracted, is_speeding, is_failure_to_yield, is_drunk_driving, "
        "is_fatigue, is_view_obstructed, is_vehicle_defect, is_backing_unsafely, "
        "is_inexperience, is_pavement_slippery → 行为因子，迁移稳定性差，已移除",
        "is_emergency → 稀有类别（<0.3%），已移除",
    ],
}

# WMO 天气码映射
COCO_MAP = {
    0: "Clear", 1: "Clear", 2: "Cloudy", 3: "Cloudy",
    45: "Cloudy", 48: "Cloudy",
    51: "Rain", 53: "Rain", 55: "Rain", 56: "Rain", 57: "Rain",
    61: "Rain", 63: "Rain", 65: "Rain", 66: "Rain", 67: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain", 81: "Rain", 82: "Rain", 85: "Snow", 86: "Snow",
    95: "Rain", 96: "Rain", 99: "Rain",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Level-A 特征工程
# ──────────────────────────────────────────────────────────────────────────────

def _season_from_month(month: int) -> str:
    if month in (12, 1, 2): return "winter"
    if month in (3, 4, 5):  return "spring"
    if month in (6, 7, 8):  return "summer"
    return "autumn"


def _vehicle_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    vcols  = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6) if f"VEHICLE TYPE CODE {i}" in df.columns]
    merged = df[vcols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    return {
        "is_sedan":      merged.str.contains(r"sedan|4 dr",         regex=True),
        "is_suv":        merged.str.contains(r"suv|sport utility",  regex=True),
        "is_taxi":       merged.str.contains(r"taxi|cab",           regex=True),
        "is_truck":      merged.str.contains(r"truck|tractor",      regex=True),
        "is_pickup":     merged.str.contains(r"pickup",             regex=True),
        "is_bus":        merged.str.contains(r"bus",                regex=True),
        "is_van":        merged.str.contains(r"van",                regex=True),
        "is_motorcycle": merged.str.contains(r"motorcycle|scooter", regex=True),
        "is_bicycle":    merged.str.contains(r"bicycle|bike",       regex=True),
    }


def _accident_type_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    从 CONTRIBUTING FACTOR 和伤亡列派生宏观事故类型 0/1 列。
    替代旧版细粒度 _factor_flags（12列）→ 4列语义更稳定的宏观分类。
    """
    fcols  = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)
              if f"CONTRIBUTING FACTOR VEHICLE {i}" in df.columns]
    merged = df[fcols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

    # 追尾
    is_rear_end = merged.str.contains(
        r"following too closely", regex=True, case=False
    )
    # 变道 / 超车
    is_lane_change = merged.str.contains(
        r"unsafe lane changing|passing or lane usage improper|passing too closely",
        regex=True, case=False,
    )
    # 涉行人：因子中提到 pedestrian，OR 行人有伤亡
    ped_inj = pd.to_numeric(
        df.get("NUMBER OF PEDESTRIANS INJURED", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    ped_kil = pd.to_numeric(
        df.get("NUMBER OF PEDESTRIANS KILLED", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    is_pedestrian = (
        merged.str.contains(r"pedestrian", regex=True, case=False)
        | (ped_inj > 0) | (ped_kil > 0)
    )
    # 涉骑行：车辆类型含 bicycle/bike，OR 骑行者有伤亡
    vcols  = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)
              if f"VEHICLE TYPE CODE {i}" in df.columns]
    vmerged = df[vcols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    cyc_inj = pd.to_numeric(
        df.get("NUMBER OF CYCLIST INJURED", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    cyc_kil = pd.to_numeric(
        df.get("NUMBER OF CYCLIST KILLED", pd.Series(0, index=df.index)),
        errors="coerce"
    ).fillna(0)
    is_cyclist = (
        vmerged.str.contains(r"bicycle|bike", regex=True, case=False)
        | (cyc_inj > 0) | (cyc_kil > 0)
    )

    return {
        "is_rear_end":            is_rear_end,
        "is_lane_change_related": is_lane_change,
        "is_pedestrian_involved": is_pedestrian,
        "is_cyclist_involved":    is_cyclist,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. OSM 路网特征（复用 prepare_2025_data.py 的 enrich_osm）
# ──────────────────────────────────────────────────────────────────────────────

def enrich_osm(df: pd.DataFrame, graphml_path: Path, signals_path: Optional[Path]) -> pd.DataFrame:
    """与 prepare_2025_data.py 保持一致的 OSM 匹配逻辑。"""
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.neighbors import BallTree

    def _compat_bool(v) -> bool:
        if isinstance(v, bool): return v
        return str(v).lower() in ("yes", "true", "1", "on")

    print(f"  [OSM] 加载路网: {graphml_path.name} ...")
    G = ox.load_graphml(
        str(graphml_path),
        edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
        graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
    )
    G_proj = ox.project_graph(G, to_crs="EPSG:32618")

    valid_mask = df["LATITUDE"].notna() & df["LONGITUDE"].notna()
    df = df.copy()
    df_valid = df[valid_mask].copy()
    geometry = [Point(lon, lat) for lon, lat in zip(df_valid["LONGITUDE"], df_valid["LATITUDE"])]
    gdf      = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:32618")

    # ── 信号灯距离 ─────────────────────────────────────────────────────────────
    dist_arr = np.full(len(df_valid), np.nan)
    has_sig  = np.zeros(len(df_valid), dtype=int)
    sig_loaded = False

    if signals_path is not None and signals_path.exists():
        try:
            sig_gdf    = gpd.read_file(str(signals_path)).to_crs("EPSG:32618")
            sig_coords = np.column_stack([
                np.asarray(sig_gdf.geometry.x, dtype=float),
                np.asarray(sig_gdf.geometry.y, dtype=float),
            ])
            sig_loaded = True
        except Exception:
            pass

    if not sig_loaded:
        sig_xy = []
        for _, attrs in G_proj.nodes(data=True):
            tags = attrs.get("tags", {})
            hw = tags.get("highway", "") if isinstance(tags, dict) else str(tags)
            if "traffic_signals" in str(hw):
                try:
                    sig_xy.append((float(attrs["x"]), float(attrs["y"])))
                except (KeyError, ValueError):
                    pass
        if sig_xy:
            sig_coords = np.array(sig_xy)
            sig_loaded = True
            print(f"  [OSM] 信号灯节点: {len(sig_xy):,} 个")

    if sig_loaded and len(sig_coords) > 0:
        pt_coords = np.column_stack([
            np.asarray(gdf_proj.geometry.x, dtype=float),
            np.asarray(gdf_proj.geometry.y, dtype=float),
        ])
        tree     = BallTree(sig_coords, leaf_size=15)
        dists, _ = tree.query(pt_coords, k=1)
        dist_arr = dists[:, 0]
        has_sig  = (dist_arr < 30).astype(int)
        print(f"  [OSM] 信号灯匹配 <30m 占比: {has_sig.mean():.1%}")

    df.loc[valid_mask, "DIST_TO_SIGNAL_M"]   = dist_arr
    df.loc[valid_mask, "HAS_TRAFFIC_SIGNAL"] = has_sig

    # ── 最近道路属性 ─────────────────────────────────────────────────────────────
    print("  [OSM] 匹配最近道路边...")
    ne_edges = ox.nearest_edges(
        G,
        X=np.asarray(df_valid["LONGITUDE"], dtype=float),
        Y=np.asarray(df_valid["LATITUDE"],  dtype=float),
    )
    osm_type_list, osm_lanes_list, osm_oneway_list = [], [], []
    for u, v, key in ne_edges:
        edge = G.get_edge_data(u, v, key) or {}
        def _get(k, default=None):
            val = edge.get(k, default)
            return val[0] if isinstance(val, list) else val
        osm_type_list.append(str(_get("highway", "residential")))
        osm_lanes_list.append(_get("lanes"))
        osm_oneway_list.append(bool(_get("oneway", False)))

    def _infer_lanes(raw_lanes, h_type: str) -> int:
        try:
            return int(float(str(raw_lanes)))
        except (ValueError, TypeError):
            pass
        h = str(h_type).lower()
        if "motorway" in h: return 3
        if "trunk"    in h: return 3
        if "primary"  in h: return 2
        return 1

    df.loc[valid_mask, "OSM_TYPE"]       = osm_type_list
    df.loc[valid_mask, "OSM_ONEWAY"]     = [int(b) for b in osm_oneway_list]
    df.loc[valid_mask, "INFERRED_LANES"] = [
        _infer_lanes(l, t) for l, t in zip(osm_lanes_list, osm_type_list)
    ]
    print("  [OSM] ✓ 路网特征提取完毕")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. 天气特征（复用 prepare_2025_data.py 的 enrich_weather）
# ──────────────────────────────────────────────────────────────────────────────

def _load_openmeteo_csv(csv_path: Path) -> pd.DataFrame:
    with open(str(csv_path), encoding="utf-8") as f:
        lines = f.readlines()
    skiprows = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time"):
            skiprows = i
            break
    weather = pd.read_csv(str(csv_path), skiprows=skiprows, low_memory=False)
    col_map = {}
    for c in weather.columns:
        cl = c.lower()
        if "temperature" in cl:   col_map[c] = "temp"
        elif "precipitation" in cl: col_map[c] = "prcp"
        elif "wind_speed" in cl or "windspeed" in cl: col_map[c] = "wspd"
        elif "weather_code" in cl: col_map[c] = "wmo_code"
    weather = weather.rename(columns=col_map)
    weather["dt"]   = pd.to_datetime(weather["time"], errors="coerce")
    weather["coco"] = pd.to_numeric(weather.get("wmo_code", pd.Series(dtype=float)),
                                    errors="coerce").fillna(3).astype(int)
    keep = [c for c in ["dt", "temp", "prcp", "wspd", "coco"] if c in weather.columns]
    return weather[keep].set_index("dt").sort_index()


def _load_meteostat_csv(csv_path: Path) -> pd.DataFrame:
    import gzip
    opener = gzip.open if str(csv_path).endswith(".gz") else open
    with opener(str(csv_path), "rt", encoding="utf-8") as f:
        head = f.readline().strip()
    has_header = "date" in head.lower() or "temp" in head.lower()
    col_names = ["date","hour","temp","dwpt","rhum","prcp","snow","wdir","wspd","wpgt","pres","tsun","coco"]
    weather = pd.read_csv(str(csv_path),
                          header=0 if has_header else None,
                          names=None if has_header else col_names,
                          low_memory=False)
    if has_header:
        weather.columns = [c.lower().strip() for c in weather.columns]
    weather["dt"] = pd.to_datetime(weather["date"].astype(str)) + pd.to_timedelta(
        pd.to_numeric(weather["hour"], errors="coerce").fillna(0).astype(int), unit="h"
    )
    return weather.set_index("dt").sort_index()


def _detect_and_load_weather(csv_path: Path) -> pd.DataFrame:
    with open(str(csv_path), encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if "latitude" in first or "longitude" in first or "elevation" in first:
        return _load_openmeteo_csv(csv_path)
    return _load_meteostat_csv(csv_path)


def enrich_weather(df: pd.DataFrame, weather_csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in weather_csv_paths:
        if p.exists():
            try:
                print(f"  [Weather] 加载: {p.name} ...")
                frames.append(_detect_and_load_weather(p))
            except Exception as e:
                print(f"  [Weather] ⚠ {p.name}: {e}")
    if not frames:
        print("  [Weather] ⚠ 无有效天气文件，跳过")
        return df

    merged_w = pd.concat(frames).groupby(level=0).mean(numeric_only=True)
    crash_dt = pd.to_datetime(
        df["CRASH DATE"].astype(str) + " " + df["CRASH TIME"].astype(str),
        errors="coerce",
    ).dt.floor("h")
    df = df.copy()
    df["_crash_dt"] = crash_dt
    w_reset = merged_w.reset_index().rename(columns={"dt": "_crash_dt"})
    temp_df = pd.merge_asof(
        df.sort_values("_crash_dt"),
        w_reset[["_crash_dt","temp","prcp","wspd","coco"]].sort_values("_crash_dt"),
        on="_crash_dt", direction="nearest", tolerance=pd.Timedelta("1h"),
    )
    temp_df = temp_df.set_index(df.sort_values("_crash_dt").index).reindex(df.index)
    df["TEMP_C"]            = pd.to_numeric(temp_df["temp"], errors="coerce")
    df["prcp"]              = pd.to_numeric(temp_df["prcp"], errors="coerce").fillna(0.0)
    df["WIND_SPEED_KMH"]    = pd.to_numeric(temp_df["wspd"], errors="coerce")
    df["coco_raw"]          = pd.to_numeric(temp_df["coco"], errors="coerce")
    df["WEATHER_CONDITION"] = df["coco_raw"].map(COCO_MAP).fillna("Clear")
    df.drop(columns=["_crash_dt", "coco_raw"], inplace=True, errors="ignore")

    matched = df["TEMP_C"].notna().sum()
    print(f"  [Weather] 匹配成功 {matched}/{len(df)} 行 ({matched/len(df):.1%})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. 核心特征构建（新 Schema）
# ──────────────────────────────────────────────────────────────────────────────

def build_features_new(
    raw_df: pd.DataFrame,
    year: int,
    n_sample: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    按新版 Schema 构建特征，不再依赖旧 train.csv 作为 schema 参考。
    """
    if n_sample > 0 and len(raw_df) > n_sample:
        raw_df = raw_df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)

    crash_dt = pd.to_datetime(raw_df["CRASH DATE"], errors="coerce")
    crash_tm = pd.to_datetime(raw_df["CRASH TIME"], format="%H:%M", errors="coerce")
    hour     = crash_tm.dt.hour.fillna(12).astype(int)
    dow      = crash_dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon, 6=Sun

    out = pd.DataFrame(index=raw_df.index)

    # ── Level-A: 时间 ─────────────────────────────────────────────────────────
    out["LATITUDE"]       = pd.to_numeric(raw_df["LATITUDE"],  errors="coerce")
    out["LONGITUDE"]      = pd.to_numeric(raw_df["LONGITUDE"], errors="coerce")
    ang = 2.0 * np.pi * (hour.astype(float) / 24.0)
    out["CRASH_TIME_SIN"] = np.sin(ang)
    out["CRASH_TIME_COS"] = np.cos(ang)
    out["SEASON"]         = crash_dt.dt.month.fillna(1).astype(int).map(_season_from_month)
    out["IS_WEEKEND"]     = (dow >= 5).astype(int)
    out["IS_AM_PEAK"]     = hour.isin([7, 8, 9]).astype(int)
    out["IS_PM_PEAK"]     = hour.isin([16, 17, 18, 19]).astype(int)

    # ── Level-A: 车辆类型（0/1）────────────────────────────────────────────────
    for k, v in _vehicle_flags(raw_df).items():
        out[k] = v.astype(int)

    # ── Level-A: 宏观事故类型（0/1）──────────────────────────────────────────
    for k, v in _accident_type_flags(raw_df).items():
        out[k] = v.astype(int)

    # ── Level-A: 伤亡二元 ────────────────────────────────────────────────────
    for raw_col, out_col in [
        ("NUMBER OF PEDESTRIANS INJURED", "NUMBER_OF_PEDESTRIANS_INJURED_BIN"),
        ("NUMBER OF PEDESTRIANS KILLED",  "NUMBER_OF_PEDESTRIANS_KILLED_BIN"),
        ("NUMBER OF CYCLIST INJURED",     "NUMBER_OF_CYCLIST_INJURED_BIN"),
        ("NUMBER OF CYCLIST KILLED",      "NUMBER_OF_CYCLIST_KILLED_BIN"),
        ("NUMBER OF MOTORIST INJURED",    "NUMBER_OF_MOTORIST_INJURED_BIN"),
        ("NUMBER OF MOTORIST KILLED",     "NUMBER_OF_MOTORIST_KILLED_BIN"),
    ]:
        out[out_col] = (
            pd.to_numeric(raw_df.get(raw_col, pd.Series(0, index=raw_df.index)),
                          errors="coerce").fillna(0) > 0
        ).astype(int)

    # ── Level-A: 车辆规模 ────────────────────────────────────────────────────
    vcols = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6) if f"VEHICLE TYPE CODE {i}" in raw_df.columns]
    total_veh = raw_df[vcols].notna().sum(axis=1).clip(lower=1, upper=5)
    out["TOTAL_VEHICLES"]  = total_veh.astype(int)
    out["IS_MULTI_VEHICLE"] = (total_veh > 1).astype(int)

    # 保留日期/时间列供天气匹配，稍后删除
    out["CRASH DATE"] = raw_df["CRASH DATE"].values
    out["CRASH TIME"] = raw_df["CRASH TIME"].values

    # ── Level-B: OSM 路网（年份特定 graphml）────────────────────────────────
    year_osm_dir = RAW / "osm" / str(year)
    graphml_path = year_osm_dir / "nyc_drive_graph.graphml"
    signals_path = year_osm_dir / "nyc_traffic_signals.geojson"

    if graphml_path.exists():
        try:
            out = enrich_osm(out, graphml_path, signals_path if signals_path.exists() else None)
        except Exception as e:
            print(f"  [OSM] ⚠ 路网匹配失败: {e}，将用默认值填充")
    else:
        print(f"  [OSM] ⚠ {graphml_path} 不存在，请先运行 convert_pbf_to_year_graphml.py")

    # ── Level-B: 天气（年份特定 CSV）────────────────────────────────────────
    year_weather_dir = RAW / "weather" / str(year)
    weather_paths = (
        sorted(year_weather_dir.glob("*.csv.gz")) +
        sorted(year_weather_dir.glob("*.csv"))
    ) if year_weather_dir.exists() else []

    if weather_paths:
        try:
            out = enrich_weather(out, weather_paths)
        except Exception as e:
            print(f"  [Weather] ⚠ 天气匹配失败: {e}")

    # 删除临时列
    out.drop(columns=["CRASH DATE", "CRASH TIME"], inplace=True, errors="ignore")

    # ── 填充 Level-B 缺失值（保守 fallback）───────────────────────────────
    _fill_missing_level_b(out)

    # ── 对齐到目标 Schema ─────────────────────────────────────────────────────
    out = _align_to_schema(out)

    return out


def _fill_missing_level_b(out: pd.DataFrame) -> None:
    """对未能填充的 Level-B 列使用保守默认值。"""
    defaults_num = {
        "DIST_TO_SIGNAL_M": 50.0,
        "INFERRED_LANES":   2,
        "TEMP_C":           15.0,
        "prcp":             0.0,
        "WIND_SPEED_KMH":   10.0,
    }
    defaults_cat = {
        "HAS_TRAFFIC_SIGNAL": 0,
        "OSM_ONEWAY":         0,
        "WEATHER_CONDITION":  "Clear",
        "OSM_TYPE":           "residential",
    }
    for col, val in defaults_num.items():
        if col not in out.columns:
            out[col] = val
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(val)

    for col, val in defaults_cat.items():
        if col not in out.columns:
            out[col] = val
        elif out[col].isna().any():
            out[col] = out[col].fillna(val)


def _align_to_schema(out: pd.DataFrame) -> pd.DataFrame:
    """对齐到 ALL_COLS 顺序，填充缺失列。"""
    for col in ALL_COLS:
        if col not in out.columns:
            out[col] = 0 if col in CATEGORICAL_COLS else 0.0
    return out[ALL_COLS]


# ──────────────────────────────────────────────────────────────────────────────
# 5. 数据质量报告
# ──────────────────────────────────────────────────────────────────────────────

def generate_quality_report(df: pd.DataFrame, year: int) -> dict:
    """生成数据质量报告。"""
    total = len(df)
    report = {
        "year":          year,
        "total_rows":    total,
        "missing_rates": {},
        "category_counts": {},
        "binary_rates":  {},
    }
    # 缺失率
    for col in ALL_COLS:
        if col in df.columns:
            miss = df[col].isna().sum()
            report["missing_rates"][col] = round(miss / total, 4) if total > 0 else 0.0

    # 类别统计
    for col in ["SEASON", "WEATHER_CONDITION", "OSM_TYPE"]:
        if col in df.columns:
            report["category_counts"][col] = df[col].value_counts().to_dict()

    # 二元特征阳性率
    binary_cols = (
        [c for c in CATEGORICAL_COLS if c.startswith("is_") or c.endswith("_BIN")]
        + ["IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK", "IS_MULTI_VEHICLE"]
    )
    for col in binary_cols:
        if col in df.columns:
            rate = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 0.0
            report["binary_rates"][col] = round(float(rate), 4)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────────────────────

def run_year(
    year: int,
    raw_csv: Path,
    out_dir: Path,
    n_sample: int,
    test_ratio: float,
    seed: int,
) -> dict:
    """处理单个年份，返回质量报告。"""
    print(f"\n{'='*64}")
    print(f"  处理年份: {year}")
    print(f"{'='*64}")

    # 加载并过滤原始数据
    print(f"[1/4] 加载 crash CSV ...")
    raw = pd.read_csv(raw_csv, low_memory=False)
    raw["CRASH DATE"] = pd.to_datetime(raw["CRASH DATE"], errors="coerce")
    mask_year = raw["CRASH DATE"].dt.year == year
    mask_nyc  = (
        pd.to_numeric(raw["LATITUDE"],  errors="coerce").between(40.45, 41.15)
        & pd.to_numeric(raw["LONGITUDE"], errors="coerce").between(-74.30, -73.65)
    )
    raw_f = raw[mask_year & mask_nyc].copy()
    raw_f["CRASH DATE"] = raw_f["CRASH DATE"].dt.strftime("%m/%d/%Y")
    print(f"    → 过滤后 {len(raw_f):,} 行（{year} 年，NYC 范围内）")

    if len(raw_f) == 0:
        raise SystemExit(f"⚠ {year} 年无有效数据，请检查 crash CSV")

    print(f"[2/4] 构建特征（n_sample={n_sample if n_sample > 0 else '全量'}）...")
    df_out = build_features_new(raw_f, year=year, n_sample=n_sample, seed=seed)
    print(f"    → 输出 {len(df_out):,} 行，{len(df_out.columns)} 列")

    print(f"[3/4] 划分 train/test ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_ratio >= 1.0:
        # 全部作为测试集
        df_train, df_test = df_out.iloc[0:0], df_out
    else:
        from sklearn.model_selection import train_test_split
        idx_tr, idx_te = train_test_split(
            df_out.index, test_size=test_ratio, random_state=seed
        )
        df_train = df_out.loc[idx_tr].reset_index(drop=True)
        df_test  = df_out.loc[idx_te].reset_index(drop=True)

    if len(df_train) > 0:
        df_train.to_csv(out_dir / "train.csv", index=False)
        print(f"    ✓ train.csv: {len(df_train):,} 行")
    df_test.to_csv(out_dir / "test.csv", index=False)
    print(f"    ✓ test.csv:  {len(df_test):,} 行")

    print(f"[4/4] 生成质量报告 ...")
    report = generate_quality_report(df_out, year)
    report.update({
        "train_rows": len(df_train),
        "test_rows":  len(df_test),
        "schema":     {
            "continuous_cols":   CONTINUOUS_COLS,
            "categorical_cols":  CATEGORICAL_COLS,
            "total_cols":        len(ALL_COLS),
            "field_mapping":     FIELD_MAPPING,
        },
    })
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    ✓ quality_report.json")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2024→2025 迁移学习数据准备（新版字段方案）"
    )
    crash_dir = RAW / "crash"
    csv_candidates = sorted(crash_dir.glob("Motor_Vehicle_Collisions_-_Crashes_*.csv"), reverse=True)
    default_csv = str(csv_candidates[0]) if csv_candidates else str(
        crash_dir / "Motor_Vehicle_Collisions_-_Crashes_20260415.csv"
    )

    parser.add_argument("--raw_csv",  default=default_csv,
                        help="原始 crash CSV（自动检测最新文件）")
    parser.add_argument("--mode",     choices=["train", "test", "all"], default="all",
                        help="train=仅2024, test=仅2025, all=两者")
    parser.add_argument("--years",    nargs="+", type=int, default=None,
                        help="手动指定年份（覆盖 mode 参数）")
    parser.add_argument("--n_sample", type=int, default=-1,
                        help="每年最多保留行数（-1 全量）")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="训练集内留出的测试比例（默认 0.2）")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    if not raw_csv.exists():
        raise SystemExit(f"Crash CSV 不存在: {raw_csv}")

    # 决定年份
    if args.years:
        year_configs = [(y, DATA / f"nyc_crash_{y}", 0.2) for y in args.years]
    elif args.mode == "train":
        year_configs = [(2024, DATA / "nyc_crash_2024", args.test_ratio)]
    elif args.mode == "test":
        year_configs = [(2025, DATA / "nyc_crash_2025", 1.0)]
    else:  # all
        year_configs = [
            (2024, DATA / "nyc_crash_2024", args.test_ratio),
            (2025, DATA / "nyc_crash_2025", 1.0),
        ]

    print("=" * 64)
    print("  NYC Crash 迁移学习数据准备 (prepare_transfer_data.py)")
    print("=" * 64)
    print(f"  crash CSV : {raw_csv.name}")
    print(f"  模式      : {args.mode}  年份: {[y for y,_,_ in year_configs]}")
    print(f"  n_sample  : {args.n_sample if args.n_sample > 0 else '全量'}")
    print(f"  schema    : {len(ALL_COLS)} 列 "
          f"({len(CONTINUOUS_COLS)} 连续 + {len(CATEGORICAL_COLS)} 离散)")

    all_reports = {}
    for year, out_dir, test_ratio in year_configs:
        report = run_year(
            year=year,
            raw_csv=raw_csv,
            out_dir=out_dir,
            n_sample=args.n_sample,
            test_ratio=test_ratio,
            seed=args.seed,
        )
        all_reports[str(year)] = report

    # 保存全局 schema 说明
    schema_out = DATA / "nyc_crash_2024" / "schema.json"
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    schema_data = {
        "generated_at":     datetime.now().strftime("%Y%m%d_%H%M%S"),
        "description":      "2024→2025 迁移学习新版字段方案",
        "total_cols":       len(ALL_COLS),
        "continuous_cols":  CONTINUOUS_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "stage1_features":  ["LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS",
                             "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK"],
        "stage2_features":  ["TEMP_C", "prcp", "WIND_SPEED_KMH", "DIST_TO_SIGNAL_M",
                             "INFERRED_LANES", "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY",
                             "WEATHER_CONDITION", "OSM_TYPE"],
        "stage3_features":  [c for c in ALL_COLS if c not in [
            "LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS",
            "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK",
            "TEMP_C", "prcp", "WIND_SPEED_KMH", "DIST_TO_SIGNAL_M",
            "INFERRED_LANES", "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY",
            "WEATHER_CONDITION", "OSM_TYPE",
        ]],
        "field_mapping":    FIELD_MAPPING,
        "removed_vs_v1":    FIELD_MAPPING["removed_fields"],
    }
    schema_out.write_text(
        json.dumps(schema_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n全局 schema 说明 → {schema_out}")
    print("\n所有年份处理完毕！")
    for y, report in all_reports.items():
        tr = report.get("train_rows", 0)
        te = report.get("test_rows", 0)
        print(f"  {y}: train={tr:,}  test={te:,}")


if __name__ == "__main__":
    main()
