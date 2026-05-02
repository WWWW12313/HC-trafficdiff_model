"""
NYC 2025 事故数据全量处理管线
==============================
解决 prepare_postcovid_data.py 的"静态问题"：
  - OSM 路网特征原先对所有行填同一个训练集中位数（导致全行 DIST_TO_SIGNAL_M 相同）
  - 天气特征原先对所有行填同一个训练集 mode

本脚本：
  - Level-B 优先使用本地离线文件做真实空间/时间匹配
  - 无本地文件时自动 fallback 回 median（保持对旧管线的兼容）

数据准备（先下载，放到 raw_data/ 相应子目录）：
  raw_data/crash/Motor_Vehicle_Collisions_-_Crashes_20250929.csv  ← NYC Open Data
  raw_data/osm/nyc_drive_graph.graphml                            ← 运行 download_osm_cache.py 生成
  raw_data/weather/72503.csv.gz                                   ← Meteostat (LaGuardia)
  raw_data/weather/74486.csv.gz                                   ← Meteostat (JFK)

用法示例:
  python pipeline/prepare_2025_data.py
  python pipeline/prepare_2025_data.py --years 2024 2025 --n_sample 10000
  python pipeline/prepare_2025_data.py --years 2022 2023 --n_sample 5000 --out_dir data/nyc_crash_postcovid
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "raw_data"

# ──────────────────────────────────────────────────────────────────────────────
# 1. 基础特征工程（与 prepare_postcovid_data.py 完全对齐）
# ──────────────────────────────────────────────────────────────────────────────

def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):  return "winter"
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    return "autumn"


def _time_period(hour: int) -> str:
    if 5  <= hour < 8:  return "dawn"
    if 8  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def _vehicle_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    vcols  = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
    # guard: only include cols that exist
    vcols  = [c for c in vcols if c in df.columns]
    merged = df[vcols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    flags = {
        "is_sedan":      merged.str.contains(r"sedan|4 dr",        regex=True),
        "is_suv":        merged.str.contains(r"suv|sport utility", regex=True),
        "is_taxi":       merged.str.contains(r"taxi|cab",          regex=True),
        "is_truck":      merged.str.contains(r"truck|tractor",     regex=True),
        "is_bus":        merged.str.contains(r"bus",               regex=True),
        "is_motorcycle": merged.str.contains(r"motorcycle|scooter",regex=True),
        "is_bicycle":    merged.str.contains(r"bicycle|bike",      regex=True),
    }
    has_vehicle = merged.str.replace(" | ", "", regex=False).str.len() > 0
    matched_main = pd.concat(flags.values(), axis=1).any(axis=1)
    flags["is_other_vehicle"] = has_vehicle & ~matched_main
    return flags


def _factor_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    fcols  = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)]
    fcols  = [c for c in fcols if c in df.columns]
    merged = df[fcols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    return {
        "is_distracted":            merged.str.contains(r"distraction|inattention",  regex=True),
        "is_speeding":              merged.str.contains(r"speed",                    regex=True),
        "is_failure_to_yield":      merged.str.contains(r"yield|right.of.way",       regex=True),
        # 与 src/data_processor.py 保持一致：宽定义（4 类行为）
        "is_following_too_closely": merged.str.contains(
            r"following too closely|unsafe lane changing|passing or lane usage improper|passing too closely",
            regex=True, case=False),
        "is_drunk_driving":         merged.str.contains(r"alcohol|drugs|intoxicated",regex=True),
        "is_fatigue":               merged.str.contains(r"fatigue|fell asleep",      regex=True),
        "is_view_obstructed":       merged.str.contains(r"view obstructed|visibility",regex=True),
        "is_vehicle_defect":        merged.str.contains(r"defective|brake|tire|steering", regex=True),
        "is_backing_unsafely":      merged.str.contains(r"backing unsafely",         regex=True),
        "is_pedestrian_related":    merged.str.contains(r"pedestrian",               regex=True),
        "is_inexperience":          merged.str.contains(r"inexperience",             regex=True),
        "is_pavement_slippery":     merged.str.contains(r"pavement slippery",        regex=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. OSM 路网特征（本地 GraphML 离线匹配）
#    修复：每行独立空间 BallTree 匹配，不再使用全局中位数
# ──────────────────────────────────────────────────────────────────────────────

def enrich_osm(df: pd.DataFrame, graphml_path: Path, signals_path: Optional[Path]) -> pd.DataFrame:
    """
    为 df 中每个事故坐标匹配 OSM 路网特征。
    新增/更新列: DIST_TO_SIGNAL_M, HAS_TRAFFIC_SIGNAL, OSM_TYPE, OSM_ONEWAY, INFERRED_LANES
    """
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point
    from sklearn.neighbors import BallTree

    print(f"  [OSM] 加载路网: {graphml_path.name} ...")
    # 旧版 osmnx 保存 graphml 时用 'yes'/'no'/'reversible' 等字符串表示布尔属性，
    # osmnx 2.x 的 _convert_bool_string 只认 'True'/'False'（大写），会抛 ValueError。
    # 直接通过 load_graphml 的 edge_dtypes/graph_dtypes 参数传入宽松转换器，
    # 无需 monkey-patch。
    def _compat_bool(v) -> bool:
        if isinstance(v, bool): return v
        vl = str(v).lower()
        return vl in ("yes", "true", "1", "on")  # 其余值（含 'reversible'/'no'/'none'）→ False

    G = ox.load_graphml(
        str(graphml_path),
        edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
        graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
    )
    # 注意：旧版 osmnx 保存的 graphml 中，边的 geometry WKT 仍是经纬度坐标，
    # ox.project_graph 只重投影节点 x/y 而不重投影边 geometry，导致 nearest_edges
    # （基于边 geometry 的空间索引）坐标系错位。
    # 修复：nearest_edges 直接用原始地理图 G（EPSG:4326）+ lon/lat 坐标；
    #       信号灯距离仍用投影图节点（x/y 属性已正确投影到 UTM）。
    G_proj  = ox.project_graph(G, to_crs="EPSG:32618")
    NYC_CRS = "EPSG:32618"

    # 过滤无效坐标
    valid_mask = df["LATITUDE"].notna() & df["LONGITUDE"].notna()
    df = df.copy()

    # GeoDataFrame（仅处理有效行）
    df_valid = df[valid_mask].copy()
    geometry = [Point(lon, lat) for lon, lat in zip(df_valid["LONGITUDE"], df_valid["LATITUDE"])]
    gdf      = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(NYC_CRS)

    # ── 2a. 信号灯距离 ─────────────────────────────────────────────────────
    dist_arr = np.full(len(df_valid), np.nan)
    has_sig  = np.zeros(len(df_valid), dtype=int)

    def _is_sig(v) -> bool:
        # 兼容多种存储格式：
        #   - 字符串 "traffic_signals"
        #   - 旧版 osmnx 保存的 dict repr "{'highway': 'traffic_signals'}"
        #   - 列表 ['traffic_signals']
        return "traffic_signals" in str(v)

    # 优先用独立 GeoJSON（体积小、加载快）
    sig_loaded = False
    if signals_path is not None and signals_path.exists():
        try:
            sig_gdf   = gpd.read_file(str(signals_path)).to_crs(NYC_CRS)
            sig_coords = np.column_stack([
                np.asarray(sig_gdf.geometry.x, dtype=float),
                np.asarray(sig_gdf.geometry.y, dtype=float),
            ])
            sig_loaded = True
        except Exception:
            pass

    if not sig_loaded:
        # 直接遍历 G_proj 节点，从 tags 字典提取投影坐标（x/y 已是 UTM 米制）
        # 旧版 osmnx 将 OSM 标签存储为 attrs['tags'] dict，不展开为单独属性列
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
            print(f"  [OSM] 找到 {len(sig_xy):,} 个信号灯节点（从路网图提取）")

    if sig_loaded and len(sig_coords) > 0:
        pt_coords = np.column_stack([
            np.asarray(gdf_proj.geometry.x, dtype=float),
            np.asarray(gdf_proj.geometry.y, dtype=float),
        ])
        tree      = BallTree(sig_coords, leaf_size=15)
        dists, _  = tree.query(pt_coords, k=1)
        dist_arr  = dists[:, 0]
        has_sig   = (dist_arr < 30).astype(int)
        sig_match_rate = has_sig.mean()
        print(f"  [OSM] 信号灯匹配完成，<30m 占比 {sig_match_rate:.1%}")
    else:
        print("  [OSM] ⚠ 无信号灯数据，DIST_TO_SIGNAL_M/HAS_TRAFFIC_SIGNAL 保持 NaN/0")

    df.loc[valid_mask, "DIST_TO_SIGNAL_M"]   = dist_arr
    df.loc[valid_mask, "HAS_TRAFFIC_SIGNAL"] = has_sig

    # ── 2b. 最近道路属性 ──────────────────────────────────────────────────
    # 用原始地理图 G（EPSG:4326）做 nearest_edges，传入 lon/lat；
    # 旧版 graphml 边 geometry 仍是经纬度，与 G 的 CRS 一致，空间索引正确。
    print("  [OSM] 匹配最近道路边...")
    ne_edges = ox.nearest_edges(
        G,
        X=np.asarray(df_valid["LONGITUDE"], dtype=float),
        Y=np.asarray(df_valid["LATITUDE"],  dtype=float),
    )

    osm_type_list, osm_lanes_list, osm_oneway_list = [], [], []
    osm_speed_list, has_divider_list = [], []
    for u, v, key in ne_edges:
        edge = G.get_edge_data(u, v, key) or {}

        def _get(k, default=None):
            val = edge.get(k, default)
            return val[0] if isinstance(val, list) else val

        h_type = str(_get("highway", "residential"))
        osm_type_list.append(h_type)
        osm_lanes_list.append(_get("lanes"))
        osm_oneway_list.append(bool(_get("oneway", False)))
        osm_speed_list.append(_get("maxspeed"))
        divider_raw = " ".join(
            str(_get(k, ""))
            for k in ["divider", "median", "separation", "barrier"]
            if _get(k, "") not in (None, "")
        ).lower()
        has_divider_list.append(int(any(token in divider_raw for token in ["yes", "median", "divider", "barrier", "kerb"])))

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

    def _infer_speed_limit_mph(raw_speed, h_type: str) -> float:
        import re

        text = str(raw_speed or "").lower()
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        if nums:
            speed = float(np.median(nums))
            if "km" in text or "kph" in text:
                speed *= 0.621371
            return float(np.clip(round(speed / 5.0) * 5.0, 5.0, 70.0))

        # OSM maxspeed 缺失时使用纽约市默认限速规则，而不是训练集统计量。
        h = str(h_type).lower()
        if "motorway" in h:
            return 50.0
        if "trunk" in h:
            return 40.0
        if "primary" in h or "secondary" in h:
            return 30.0
        return 25.0

    df.loc[valid_mask, "OSM_TYPE"]       = osm_type_list
    df.loc[valid_mask, "OSM_ONEWAY"]     = [int(b) for b in osm_oneway_list]
    df.loc[valid_mask, "REAL_SPEED_LIMIT"] = [
        _infer_speed_limit_mph(s, t) for s, t in zip(osm_speed_list, osm_type_list)
    ]
    df.loc[valid_mask, "HAS_DIVIDER"] = has_divider_list
    df.loc[valid_mask, "INFERRED_LANES"] = [
        _infer_lanes(l, t) for l, t in zip(osm_lanes_list, osm_type_list)
    ]
    tagged_speed = sum(str(x or "").strip() != "" for x in osm_speed_list)
    print(f"  [OSM] maxspeed 标签覆盖 {tagged_speed:,}/{len(osm_speed_list):,} 行；缺失处使用 NYC/道路等级默认限速")

    print("  [OSM] 路网特征提取完毕")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. 天气特征（本地 Meteostat bulk CSV 离线匹配）
#    修复：每行 merge_asof 时间匹配，不再使用全局 mode
# ──────────────────────────────────────────────────────────────────────────────

# WMO weather code (Open-Meteo) → WEATHER_CONDITION 字符串
# 与 2017 训练集保持完全一致：仅使用 Clear / Cloudy / Rain / Snow 四类
# 注意：coco 列直接存 WMO 原始码，不再做二次映射
COCO_MAP = {
    0:  "Clear",  1:  "Clear",                          # Clear / Mainly clear
    2:  "Cloudy", 3:  "Cloudy",                         # Partly cloudy / Overcast
    45: "Cloudy", 48: "Cloudy",                         # Fog
    51: "Rain",   53: "Rain",   55: "Rain",             # Drizzle
    56: "Rain",   57: "Rain",                           # Freezing drizzle
    61: "Rain",   63: "Rain",   65: "Rain",             # Rain
    66: "Rain",   67: "Rain",                           # Freezing rain
    71: "Snow",   73: "Snow",   75: "Snow",   77: "Snow", # Snowfall
    80: "Rain",   81: "Rain",   82: "Rain",             # Rain showers
    85: "Snow",   86: "Snow",                           # Snow showers
    95: "Rain",   96: "Rain",   99: "Rain",             # Thunderstorm
}


def _load_meteostat_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取 Meteostat bulk hourly CSV（支持 .csv 和 .csv.gz）。
    格式：date,hour,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco
    """
    opener = gzip.open if str(csv_path).endswith(".gz") else open
    with opener(str(csv_path), "rt", encoding="utf-8") as f:
        head = f.readline().strip()
    has_header = "date" in head.lower() or "temp" in head.lower()

    col_names = [
        "date", "hour", "temp", "dwpt", "rhum", "prcp",
        "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco",
    ]
    weather = pd.read_csv(
        str(csv_path),
        header=0    if has_header else None,
        names=None  if has_header else col_names,
        low_memory=False,
    )
    if has_header:
        weather.columns = [c.lower().strip() for c in weather.columns]

    # 构建 datetime 索引
    weather["dt"] = pd.to_datetime(weather["date"].astype(str)) + pd.to_timedelta(
        pd.to_numeric(weather["hour"], errors="coerce").fillna(0).astype(int), unit="h"
    )
    return weather.set_index("dt").sort_index()


def _load_openmeteo_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取 Open-Meteo hourly CSV。
    格式: 前两行元数据（latitude/longitude...），之后是带列头的数据。
    时间列为本地时间（America/New_York），与 NYC crash 时间戳一致，可直接匹配。
    """
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
        if "temperature" in cl:
            col_map[c] = "temp"
        elif "precipitation" in cl:
            col_map[c] = "prcp"
        elif "wind_speed" in cl or "windspeed" in cl:
            col_map[c] = "wspd"
        elif "weather_code" in cl:
            col_map[c] = "wmo_code"
    weather = weather.rename(columns=col_map)
    weather["dt"] = pd.to_datetime(weather["time"], errors="coerce")
    # 直接使用 WMO 原始码作为 coco，与 2017 训练集保持一致
    weather["coco"] = (
        pd.to_numeric(weather.get("wmo_code", pd.Series(dtype=float)), errors="coerce")
        .fillna(3)
        .astype(int)
    )
    keep = [c for c in ["dt", "temp", "prcp", "wspd", "coco"] if c in weather.columns]
    return weather[keep].set_index("dt").sort_index()


def _detect_and_load_weather(csv_path: Path) -> pd.DataFrame:
    """自动检测天气 CSV 格式（Meteostat 或 Open-Meteo）并加载。"""
    with open(str(csv_path), encoding="utf-8") as f:
        first_line = f.readline().strip().lower()
    if "latitude" in first_line or "longitude" in first_line or "elevation" in first_line:
        return _load_openmeteo_csv(csv_path)   # Open-Meteo 格式
    return _load_meteostat_csv(csv_path)       # Meteostat 格式


def enrich_weather(df: pd.DataFrame, weather_csv_paths: List[Path]) -> pd.DataFrame:
    """
    为 df 中每个事故行按时间戳匹配天气特征。
    需要 df 有 'CRASH DATE' 和 'CRASH TIME' 列（原始格式）。
    新增/更新列: TEMP_C, prcp, WIND_SPEED_KMH, coco, WEATHER_CONDITION
    支持 Meteostat（.csv.gz）和 Open-Meteo（.csv）两种格式，自动检测。
    """
    frames = []
    for p in weather_csv_paths:
        if p.exists():
            try:
                print(f"  [Weather] 加载: {p.name} ...")
                frames.append(_detect_and_load_weather(p))
            except Exception as e:
                print(f"  [Weather] ⚠ 加载失败: {p.name}: {e}")

    if not frames:
        print("  [Weather] ⚠ 无有效天气文件，跳过实时匹配")
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
        w_reset[["_crash_dt", "temp", "prcp", "wspd", "coco"]].sort_values("_crash_dt"),
        on="_crash_dt",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    # 恢复原始行序
    temp_df = temp_df.set_index(df.sort_values("_crash_dt").index).reindex(df.index)

    df["TEMP_C"]            = pd.to_numeric(temp_df["temp"], errors="coerce")
    df["prcp"]              = pd.to_numeric(temp_df["prcp"], errors="coerce").fillna(0.0)
    df["WIND_SPEED_KMH"]    = pd.to_numeric(temp_df["wspd"], errors="coerce")
    df["coco"]              = pd.to_numeric(temp_df["coco"], errors="coerce")
    df["WEATHER_CONDITION"] = df["coco"].map(COCO_MAP).fillna("Clear")
    df.drop(columns=["_crash_dt"], inplace=True, errors="ignore")

    matched = df["TEMP_C"].notna().sum()
    print(f"  [Weather] 匹配成功 {matched}/{len(df)} 行 ({matched/len(df):.1%})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. 主特征构建
# ──────────────────────────────────────────────────────────────────────────────

# Level-B 列（路网 + 天气），用于 fallback 判断
_LEVEL_B_OSM     = {"DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL", "OSM_TYPE", "OSM_ONEWAY", "INFERRED_LANES"}
_LEVEL_B_WEATHER = {"TEMP_C", "prcp", "WIND_SPEED_KMH", "coco", "WEATHER_CONDITION"}
_LEVEL_B_ALL     = _LEVEL_B_OSM | _LEVEL_B_WEATHER


def build_features(
    raw_df: pd.DataFrame,
    train_ref: pd.DataFrame,
    graphml_path: Optional[Path] = None,
    signals_path: Optional[Path] = None,
    weather_paths: Optional[List[Path]] = None,
    n_sample: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    核心特征构建函数。

    Level-A（时间/行为/类型）：从 raw_df 列直接推导，不依赖外部资源。
    Level-B（路网/天气）：
        - 优先使用本地离线文件做真实空间/时间匹配（彻底解决静态问题）
        - 访问每行独立的坐标/时间 → 每行特征值不同
        - 无本地文件时 fallback 回 train_ref 的 median/mode
    """
    if n_sample > 0 and len(raw_df) > n_sample:
        raw_df = raw_df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)  # n_sample=-1 或数据量不足时保留全量

    crash_dt = pd.to_datetime(raw_df["CRASH DATE"], errors="coerce")
    crash_tm = pd.to_datetime(raw_df["CRASH TIME"], format="%H:%M", errors="coerce")
    hour     = crash_tm.dt.hour.fillna(12).astype(int)

    out = pd.DataFrame(index=raw_df.index)

    # ── Level-A ───────────────────────────────────────────────────────────────
    out["LATITUDE"]       = pd.to_numeric(raw_df["LATITUDE"],  errors="coerce")
    out["LONGITUDE"]      = pd.to_numeric(raw_df["LONGITUDE"], errors="coerce")
    ang = 2.0 * np.pi * (hour.astype(float) / 24.0)
    out["CRASH_TIME_SIN"] = np.sin(ang)
    out["CRASH_TIME_COS"] = np.cos(ang)
    out["SEASON"]         = crash_dt.dt.month.fillna(1).astype(int).map(_season_from_month)
    out["DAY_OF_WEEK"]    = crash_dt.dt.dayofweek.fillna(0).astype(int)
    out["TIME_PERIOD"]    = hour.map(_time_period)
    # ── 新增字段（2026-04-26 重构）────────────────────────────────────────
    dow = crash_dt.dt.dayofweek.fillna(0).astype(int)
    out["IS_WEEKEND"]  = (dow >= 5).astype(int)
    out["IS_AM_PEAK"]  = ((hour >= 7) & (hour <= 9)).astype(int)
    out["IS_PM_PEAK"]  = ((hour >= 17) & (hour <= 19)).astype(int)

    veh_flags    = _vehicle_flags(raw_df)
    factor_flags = _factor_flags(raw_df)
    for k, v in {**veh_flags, **factor_flags}.items():
        out[k] = v.astype(int)

    # ── 事故类型二值标志（从 contributing factors 派生）────────────────────
    fcols_raw = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)]
    fcols_raw = [c for c in fcols_raw if c in raw_df.columns]
    fcols_merged = raw_df[fcols_raw].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    out["is_rear_end"]            = fcols_merged.str.contains(
        r"following too closely|unsafe following", regex=True).astype(int)
    out["is_lane_change_related"] = fcols_merged.str.contains(
        r"unsafe lane changing|passing or lane usage improper|passing too closely", regex=True).astype(int)
    out["is_pedestrian_involved"] = (
        factor_flags.get("is_pedestrian_related",
                         pd.Series(0, index=raw_df.index)) > 0
    ).astype(int)
    vcols_raw = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
    vcols_raw = [c for c in vcols_raw if c in raw_df.columns]
    veh_merged = raw_df[vcols_raw].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    out["is_cyclist_involved"]    = (
        veh_merged.str.contains(r"bicycle|bike", regex=True) |
        factor_flags.get("is_pedestrian_related",
                         pd.Series(0, index=raw_df.index)).astype(bool)
    ).astype(int)

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

    vcols     = [c for c in [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)] if c in raw_df.columns]
    total_veh = raw_df[vcols].notna().sum(axis=1).clip(lower=1, upper=5)
    out["TOTAL_VEHICLES"]    = total_veh.astype(int)
    out["IS_MULTI_VEHICLE"]  = (total_veh > 1).astype(int)
    out["NUMBER OF PERSONS INJURED"] = (
        pd.to_numeric(raw_df["NUMBER OF PERSONS INJURED"], errors="coerce").fillna(0.0)
    )

    # 保留原始日期/时间列，供天气匹配使用（稍后删除）
    out["CRASH DATE"] = raw_df["CRASH DATE"].values
    out["CRASH TIME"] = raw_df["CRASH TIME"].values

    # ── Level-B：真实匹配（静态修复核心）────────────────────────────────────
    osm_done     = False
    weather_done = False

    if graphml_path is not None and graphml_path.exists():
        try:
            out = enrich_osm(out, graphml_path, signals_path)
            osm_done = True
        except Exception as e:
            print(f"  [OSM] ⚠ 路网匹配失败（{e}），回退 median")

    if weather_paths:
        valid_wpaths = [p for p in weather_paths if p.exists()]
        if valid_wpaths:
            try:
                out = enrich_weather(out, valid_wpaths)
                weather_done = True
            except Exception as e:
                print(f"  [Weather] ⚠ 天气匹配失败（{e}），回退 median")

    # 删除临时列
    out.drop(columns=["CRASH DATE", "CRASH TIME"], inplace=True, errors="ignore")

    # ── Level-B Fallback：仍缺失的列 → train_ref median/mode ────────────────
    fallback_cols = []
    for c in _LEVEL_B_ALL:
        if c in out.columns:
            continue  # 已由真实匹配填充
        if c not in train_ref.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_ref[c]):
            out[c] = pd.to_numeric(train_ref[c], errors="coerce").median()
        else:
            mode = train_ref[c].mode(dropna=True)
            out[c] = mode.iloc[0] if len(mode) else train_ref[c].dropna().iloc[0]
        fallback_cols.append(c)

    if fallback_cols:
        print(f"  ⚠ 以下列使用静态 fallback: {fallback_cols}")

    # ── 对齐 schema ───────────────────────────────────────────────────────────
    # 优先使用 column_groups.json 定义的新 schema（2026-04-26 重构）
    _col_groups_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "column_groups.json"
    _target_cols = None
    if _col_groups_path.exists():
        import json as _json
        _cg = _json.loads(_col_groups_path.read_text(encoding="utf-8"))
        _target_cols = (
            _cg.get("stage1_continuous", []) +
            _cg.get("stage1_categorical", []) +
            _cg.get("stage2_continuous", []) +
            _cg.get("stage2_categorical", []) +
            _cg.get("stage3_categorical", [])
        )
        # 去重保序
        seen = set()
        _target_cols = [c for c in _target_cols if not (c in seen or seen.add(c))]

    target_col = "NUMBER OF PERSONS INJURED"
    target_cols = _target_cols if _target_cols else list(train_ref.columns)
    if target_col not in target_cols:
        target_cols = target_cols + [target_col]
    out = out.reindex(columns=target_cols)
    for c in target_cols:
        if c not in out.columns or out[c].isna().all():
            if c in train_ref.columns:
                if pd.api.types.is_numeric_dtype(train_ref[c]):
                    out[c] = pd.to_numeric(train_ref[c], errors="coerce").median()
                else:
                    mode = train_ref[c].mode(dropna=True)
                    out[c] = mode.iloc[0] if len(mode) else 0
            else:
                out[c] = 0

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC crash data 处理管线（离线 OSM + 离线 Weather，支持任意年份）"
    )
    # 自动搜索 raw_data/crash/ 下最新的 crash CSV
    _crash_dir = RAW / "crash"
    _csv_candidates = sorted(_crash_dir.glob("Motor_Vehicle_Collisions_-_Crashes_*.csv"), reverse=True)
    _default_csv = str(_csv_candidates[0]) if _csv_candidates else str(
        _crash_dir / "Motor_Vehicle_Collisions_-_Crashes_20260415.csv"
    )
    parser.add_argument(
        "--raw_csv",
        default=_default_csv,
        help="原始 crash CSV 路径（自动检测 raw_data/crash/ 下最新文件）",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2025],
        help="要处理的年份（可多个，如 2025 或 2022 2023）",
    )
    parser.add_argument("--n_sample", type=int, default=10000,
                        help="每年最多保留的行数（-1 = 全量）")
    parser.add_argument("--seed",     type=int, default=42)
    # 注意：--osm_graphml / --weather_csvs 默认值在 parse 后按年份动态确定
    parser.add_argument(
        "--osm_graphml",
        default=None,
        help="本地路网 GraphML（默认自动查找 raw_data/osm/{year}/nyc_drive_graph.graphml）",
    )
    parser.add_argument(
        "--osm_signals",
        default=None,
        help="信号灯 GeoJSON（可选，留空时从路网节点提取）",
    )
    parser.add_argument(
        "--weather_csvs", nargs="*",
        default=None,
        help="天气 CSV（默认自动查找 raw_data/weather/{year}/*.csv）",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "data" / "nyc_crash_2025"),
        help="输出目录（train.csv / test.csv / info.json）",
    )
    parser.add_argument(
        "--train_ref",
        default=str(ROOT / "data" / "nyc_crash" / "train.csv"),
        help="参考 schema 所在的已有 train.csv",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="测试集比例（默认 0.2）",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 年份专属路径自动解析 ──────────────────────────────────────────────────
    # 支持多年份时取第一个年份确定 OSM/天气路径（多年份共享同一路网）
    _primary_year = args.years[0]

    if args.osm_graphml is not None:
        graphml_path = Path(args.osm_graphml)
    else:
        _year_gml = RAW / "osm" / str(_primary_year) / "nyc_drive_graph.graphml"
        _flat_gml = RAW / "osm" / "nyc_drive_graph.graphml"
        graphml_path = _year_gml if _year_gml.exists() else _flat_gml

    if args.osm_signals is not None:
        signals_path = Path(args.osm_signals)
    else:
        _year_sig = RAW / "osm" / str(_primary_year) / "nyc_traffic_signals.geojson"
        signals_path = _year_sig if _year_sig.exists() else Path("/nonexistent")

    if args.weather_csvs is not None:
        weather_paths = [Path(p) for p in args.weather_csvs]
    else:
        _year_wdir = RAW / "weather" / str(_primary_year)
        if _year_wdir.exists():
            weather_paths = (
                sorted(_year_wdir.glob("*.csv.gz")) +
                sorted(_year_wdir.glob("*.csv"))
            )
        else:
            weather_paths = (
                sorted((RAW / "weather").glob("*.csv.gz")) +
                sorted((RAW / "weather").glob("*.csv"))
            )

    if not raw_path.exists():
        raise SystemExit(f"Crash CSV 不存在: {raw_path}")
    if not Path(args.train_ref).exists():
        raise SystemExit(f"参考 train.csv 不存在: {args.train_ref}")

    # ── 打印准备状态 ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  NYC Crash Data Pipeline  (prepare_2025_data.py)")
    print("=" * 64)
    print(f"  crash CSV   : {raw_path.name}")
    print(f"  years       : {args.years}")
    print(f"  n_sample    : {args.n_sample}")
    print(f"  OSM graphml : {'✅ 存在' if graphml_path.exists() else '⚠ 未找到（将用 median fallback）'}")
    print(f"  OSM signals : {'✅ 存在' if signals_path.exists() else '— 未找到（从路网提取）'}")
    for wp in weather_paths:
        print(f"  weather     : {wp.name}  {'✅' if wp.exists() else '⚠ 未找到'}")
    print()

    # ── 加载原始数据 ──────────────────────────────────────────────────────────
    print("[1/5] 加载原始 crash CSV ...")
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["CRASH DATE"] = pd.to_datetime(raw["CRASH DATE"], errors="coerce")

    mask_year = raw["CRASH DATE"].dt.year.isin(args.years)
    mask_nyc  = (
        pd.to_numeric(raw["LATITUDE"],  errors="coerce").between(40.45, 41.15)
        & pd.to_numeric(raw["LONGITUDE"], errors="coerce").between(-74.30, -73.65)
    )
    raw_filtered = raw[mask_year & mask_nyc].copy()
    raw_filtered["CRASH DATE"] = raw_filtered["CRASH DATE"].dt.strftime("%m/%d/%Y")
    print(f"    → 过滤后 {len(raw_filtered):,} 行（年份 {args.years}，NYC 范围内）")

    if len(raw_filtered) == 0:
        raise SystemExit(
            "过滤后无数据。请确认：\n"
            "  1. CSV 中确有对应年份的记录\n"
            "  2. CRASH DATE 格式正确（如 01/15/2025）\n"
            "  3. LATITUDE/LONGITUDE 在 NYC 范围内"
        )

    # ── 加载参考 schema ───────────────────────────────────────────────────────
    print("[2/5] 加载参考 schema ...")
    train_ref = pd.read_csv(args.train_ref)

    # ── 构建特征 ──────────────────────────────────────────────────────────────
    n = args.n_sample if args.n_sample > 0 else len(raw_filtered)
    print(f"[3/5] 构建特征（Level-A + Level-B，n_sample={n}）...")
    df_out = build_features(
        raw_df        = raw_filtered,
        train_ref     = train_ref,
        graphml_path  = graphml_path if graphml_path.exists() else None,
        signals_path  = signals_path if signals_path.exists() else None,
        weather_paths = weather_paths,
        n_sample      = n,
        seed          = args.seed,
    )

    # ── train / test 划分 ─────────────────────────────────────────────────────
    print("[4/5] 划分 train / test ...")
    # test_ratio=1.0 表示全部用作迁移测试集（不拆分，常用于 postcovid 评估）
    if args.test_ratio >= 1.0:
        df_train = df_out.iloc[0:0].reset_index(drop=True)  # 空 DataFrame，占位
        df_test  = df_out.reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            df_out.index, test_size=args.test_ratio, random_state=args.seed
        )
        df_train = df_out.loc[idx_train].reset_index(drop=True)
        df_test  = df_out.loc[idx_test].reset_index(drop=True)

    if len(df_train) > 0:
        df_train.to_csv(out_dir / "train.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    # 同时写一份到 results/，供 evaluate_postcovid_transfer.py 使用
    # 文件名包含年份和实际行数，便于追踪
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    year_str = "_".join(str(y) for y in args.years)
    postcovid_path = results_dir / f"postcovid_test_{year_str}_n{len(df_test)}.csv"
    df_test.to_csv(postcovid_path, index=False)
    print(f"  → 迁移测试集副本: {postcovid_path.name} ({len(df_test):,} 行)")

    # 基于当前输出 schema 生成 info.json（不再直接复制旧版 2017 info）
    col_groups_path = ROOT / "data" / "processed" / "column_groups.json"
    with open(col_groups_path, encoding="utf-8") as f:
        col_groups = json.load(f)

    num_col_names = list(col_groups.get("continuous_cols", []))
    cat_col_names = list(col_groups.get("categorical_cols", []))
    target_col = "NUMBER OF PERSONS INJURED"

    # 仅保留实际存在于输出表中的列，避免 schema 漂移造成 info 异常
    num_col_names = [c for c in num_col_names if c in df_out.columns]
    cat_col_names = [c for c in cat_col_names if c in df_out.columns]

    info = {
        "task_type": "regression",
        "n_num_features": len(num_col_names),
        "n_cat_features": len(cat_col_names),
        "n_classes": None,
        "train_size": len(df_train),
        "test_size": len(df_test),
        "num_col_names": num_col_names,
        "cat_col_names": cat_col_names,
        "cat_sizes": [
            int(pd.concat([
                df_train[[c]] if len(df_train) else df_test[[c]],
                train_ref[[c]] if c in train_ref.columns else pd.DataFrame({c: []}),
            ], axis=0)[c].nunique(dropna=True))
            for c in cat_col_names
        ],
        "target_col": target_col,
        "num_col_idx": list(range(len(num_col_names))),
        "cat_col_idx": list(range(len(num_col_names), len(num_col_names) + len(cat_col_names))),
    }
    (out_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at":    ts,
        "years":           args.years,
        "n_sample":        n,
        "total_rows":      len(df_out),
        "train_rows":      len(df_train),
        "test_rows":       len(df_test),
        "osm_mode":        "real_spatial"  if graphml_path.exists() else "static_median",
        "weather_mode":    "real_temporal" if any(p.exists() for p in weather_paths) else "static_median",
        "output_dir":      str(out_dir),
        "postcovid_file":  str(postcovid_path),
    }
    (out_dir / "prep_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[5/5] 完成 → {out_dir}")
    if len(df_train) > 0:
        print(f"      train:   {len(df_train):,} 行")
    print(f"      test:    {len(df_test):,} 行")
    print(f"      OSM:     {report['osm_mode']}")
    print(f"      Weather: {report['weather_mode']}")


if __name__ == "__main__":
    main()
