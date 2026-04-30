"""
离线 OSM + Meteostat 天气补全脚本（2017 NYC）

功能概览：
1) 读取事故数据（包含经纬度与时间）
2) 使用本地 Geofabrik PBF（pyrosm）提取道路/信号灯/交叉口并做最近邻匹配
3) 使用 Meteostat 拉取 2017 逐小时天气并按小时对齐
4) 输出补全后的结果 CSV

依赖：
- pandas
- geopandas
- shapely
- pyrosm
- meteostat
- scipy

示例：
python scripts/merge_offline_osm_weather_2017.py \
  --accident_csv data/nyc_2017_pristine_v8.csv \
  --pbf_path osmdata/new-york-180101-internal.osm.pbf \
  --output_csv exp/nyc_crash_v8_ablation/nyc_2017_enriched_offline.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from meteostat import Hourly, Stations
from pyrosm import OSM
from scipy.spatial import KDTree
from shapely.geometry import Point

LOGGER = logging.getLogger("offline.osm.weather")


# =========================
# 模块 1：数据加载与时间解析
# =========================


@dataclass
class Config:
    accident_csv: Path
    pbf_path: Path
    output_csv: Path
    lon_col: str = "LONGITUDE"
    lat_col: str = "LATITUDE"
    date_col: str = "CRASH DATE"
    time_col: str = "CRASH TIME"
    datetime_col: Optional[str] = None
    local_tz: str = "America/New_York"
    metric_crs: str = "EPSG:2263"  # 纽约常用米制投影
    signal_threshold_m: float = 20.0
    weather_rounding: str = "floor"  # floor or round
    weather_csv: Optional[Path] = None


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_accident_datetime(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """将事故时间解析为带时区时间（纽约本地时间）。"""
    if cfg.datetime_col and cfg.datetime_col in df.columns:
        dt = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    elif cfg.date_col in df.columns and cfg.time_col in df.columns:
        dt = pd.to_datetime(
            df[cfg.date_col].astype(str).str.strip() + " " + df[cfg.time_col].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        raise ValueError(
            f"无法解析事故时间：未找到 datetime_col={cfg.datetime_col} 或 (date_col={cfg.date_col}, time_col={cfg.time_col})"
        )

    # 事故时间通常是本地时间（无时区），统一本地化到纽约时区
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(cfg.local_tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        dt = dt.dt.tz_convert(cfg.local_tz)
    return dt


def load_accident_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.accident_csv)
    # Some CSVs contain padded headers (e.g., leading spaces before CRASH DATE).
    df.columns = [str(c).strip() for c in df.columns]
    if cfg.lon_col not in df.columns or cfg.lat_col not in df.columns:
        raise ValueError(f"缺少经纬度列：{cfg.lon_col}, {cfg.lat_col}")

    df[cfg.lon_col] = pd.to_numeric(df[cfg.lon_col], errors="coerce")
    df[cfg.lat_col] = pd.to_numeric(df[cfg.lat_col], errors="coerce")
    df = df.dropna(subset=[cfg.lon_col, cfg.lat_col]).copy()

    df["__CRASH_DT_LOCAL"] = _parse_accident_datetime(df, cfg)
    df = df.dropna(subset=["__CRASH_DT_LOCAL"]).copy()

    LOGGER.info("事故数据加载完成：%d 行", len(df))
    return df


# =========================
# 模块 2：OSM 离线匹配（PBF）
# =========================


def _safe_series(df: pd.DataFrame, col: str, default: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _to_gdf_points(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col], crs="EPSG:4326"),
    )


def _nearest_distance_kdtree(src_points: gpd.GeoSeries, tgt_points: gpd.GeoSeries) -> np.ndarray:
    """当 sjoin_nearest 不可用时，用 KDTree 计算最近距离（单位：投影坐标单位，米）。"""
    if len(tgt_points) == 0:
        return np.full(len(src_points), np.nan, dtype=float)

    src_xy = np.column_stack([src_points.x.to_numpy(), src_points.y.to_numpy()])
    tgt_xy = np.column_stack([tgt_points.x.to_numpy(), tgt_points.y.to_numpy()])
    tree = KDTree(tgt_xy)
    dist, _ = tree.query(src_xy, k=1)
    return dist.astype(float)


def _extract_osm_objects(cfg: Config) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    从本地 PBF 提取：
    - drive/highway 道路（含 maxspeed/oneway/highway）
    - 信号灯节点（highway=traffic_signals）
    - 交叉口节点（基于路网节点度数）
    """
    osm = OSM(str(cfg.pbf_path))

    # roads: driving network edges + nodes
    net = osm.get_network(network_type="driving", nodes=True)
    if net is None:
        raise RuntimeError("PBF 中未提取到 driving 路网")
    nodes, edges = net
    if edges is None or len(edges) == 0:
        raise RuntimeError("PBF 中未提取到 driving 路网 edges")

    roads = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326").copy()
    node_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326").copy()

    # 尽量提取信号灯 POI
    try:
        signals = osm.get_pois(custom_filter={"highway": ["traffic_signals"]})
        if signals is None or len(signals) == 0:
            raise RuntimeError("no signals in POI")
        signal_gdf = gpd.GeoDataFrame(signals, geometry="geometry", crs="EPSG:4326").copy()
    except Exception:
        # 兜底：从网络节点中找 highway=traffic_signals
        if "highway" in node_gdf.columns:
            signal_gdf = node_gdf[node_gdf["highway"].astype(str) == "traffic_signals"].copy()
        else:
            signal_gdf = node_gdf.iloc[0:0].copy()

    # 交叉口：利用 edges 的 u/v 计算节点度数
    intersection_gdf = node_gdf.iloc[0:0].copy()
    if "u" in roads.columns and "v" in roads.columns and "id" in node_gdf.columns:
        deg = pd.concat([roads["u"], roads["v"]], axis=0).value_counts()
        # 度>=3 更像真实路口；若太少再回退到 >=2
        inter_ids = deg[deg >= 3].index
        if len(inter_ids) < 100:
            inter_ids = deg[deg >= 2].index
        intersection_gdf = node_gdf[node_gdf["id"].isin(inter_ids)].copy()

    LOGGER.info("OSM 提取完成：roads=%d, signals=%d, intersections=%d", len(roads), len(signal_gdf), len(intersection_gdf))
    return roads, signal_gdf, intersection_gdf


def _sjoin_nearest_with_fallback(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    right_cols: list[str],
    distance_col: str,
) -> gpd.GeoDataFrame:
    """优先使用 sjoin_nearest；失败时回退到 KDTree 距离计算。"""
    if len(right) == 0:
        out = left.copy()
        for c in right_cols:
            out[c] = np.nan
        out[distance_col] = np.nan
        return out

    try:
        # sjoin_nearest 会自动返回最近右表记录和距离
        out = gpd.sjoin_nearest(
            left,
            right[right_cols + ["geometry"]],
            how="left",
            distance_col=distance_col,
        )
        if "index_right" in out.columns:
            out = out.drop(columns=["index_right"])
        return out
    except Exception as e:
        LOGGER.warning("sjoin_nearest 失败，回退 KDTree: %s", e)
        out = left.copy()
        # 仅计算最近距离，属性列无法直接对应，尽量置空
        for c in right_cols:
            out[c] = np.nan
        out[distance_col] = _nearest_distance_kdtree(out.geometry, right.geometry)
        return out


def enrich_with_osm(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    roads, signal_gdf, intersection_gdf = _extract_osm_objects(cfg)

    acc_gdf = _to_gdf_points(df, cfg.lon_col, cfg.lat_col)

    # 统一投影到米制坐标系
    acc_m = acc_gdf.to_crs(cfg.metric_crs)
    roads_m = roads.to_crs(cfg.metric_crs)
    signals_m = signal_gdf.to_crs(cfg.metric_crs)
    inter_m = intersection_gdf.to_crs(cfg.metric_crs)

    # 1) 最近道路匹配
    road_cols = [c for c in ["highway", "maxspeed", "oneway", "osm_type"] if c in roads_m.columns]
    joined = _sjoin_nearest_with_fallback(acc_m, roads_m, road_cols, "__dist_to_road_m")

    # 2) 最近信号灯距离
    joined_sig = _sjoin_nearest_with_fallback(joined, signals_m, [], "DIST_TO_SIGNAL_M")

    # 3) 最近交叉口距离
    joined_int = _sjoin_nearest_with_fallback(joined_sig, inter_m, [], "CTX_DIST_TO_INTERSECTION")

    out = pd.DataFrame(joined_int.drop(columns="geometry"))

    # 字段映射（严格按用户给定命名）
    out["CTX_HIGHWAY"] = _safe_series(out, "highway", "N/A")
    out["CTX_MAXSPEED"] = _safe_series(out, "maxspeed", "N/A")
    out["OSM_SPEED_TAG"] = out["CTX_MAXSPEED"]

    oneway_raw = _safe_series(out, "oneway", "0").str.lower().str.strip()
    oneway_bool = oneway_raw.isin(["1", "true", "yes", "y", "t"])
    out["CTX_ONEWAY"] = oneway_bool.astype(int)
    out["OSM_ONEWAY"] = out["CTX_ONEWAY"]

    if "osm_type" in out.columns:
        out["OSM_TYPE"] = _safe_series(out, "osm_type", "way")
    else:
        # pyrosm 的道路通常为 way
        out["OSM_TYPE"] = "way"

    out["DIST_TO_SIGNAL_M"] = pd.to_numeric(out["DIST_TO_SIGNAL_M"], errors="coerce")
    out["CTX_DIST_TO_INTERSECTION"] = pd.to_numeric(out["CTX_DIST_TO_INTERSECTION"], errors="coerce")

    signalized = out["DIST_TO_SIGNAL_M"].notna() & (out["DIST_TO_SIGNAL_M"] <= cfg.signal_threshold_m)
    out["CTX_IS_SIGNALIZED"] = signalized.astype(int)
    out["HAS_TRAFFIC_SIGNAL"] = out["CTX_IS_SIGNALIZED"]

    out["CTX_OSM_SOURCE"] = "Geofabrik_Offline_PBF"

    LOGGER.info(
        "OSM 匹配完成：有道路匹配=%d, 信号灯阈值内=%d",
        int((out["CTX_HIGHWAY"] != "N/A").sum()),
        int(out["CTX_IS_SIGNALIZED"].sum()),
    )
    return out


# =========================
# 模块 3：Meteostat 天气匹配
# =========================


def _load_offline_weather_csv(weather_csv: Path, local_tz: str) -> pd.DataFrame:
    """读取 Open-Meteo 离线小时天气 CSV，并映射到统一字段。"""
    raw = pd.read_csv(weather_csv)
    # 文件前两行是站点元信息，后面才是 hourly 表头。
    if "time" not in raw.columns:
        raw = pd.read_csv(weather_csv, skiprows=3)

    required_cols = [
        "time",
        "temperature_2m (°C)",
        "precipitation (mm)",
        "wind_speed_10m (km/h)",
        "weather_code (wmo code)",
    ]

    missing = [k for k in required_cols if k not in raw.columns]
    if missing:
        raise RuntimeError(f"离线天气文件缺少字段: {missing}")

    out = pd.DataFrame()
    dt = pd.to_datetime(raw["time"], errors="coerce")
    # Open-Meteo 该文件已是纽约本地时间序列（含 timezone 元信息），按本地时区处理。
    dt = dt.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="shift_forward")

    out["__WEATHER_DT_LOCAL"] = dt
    out["TEMP_C"] = pd.to_numeric(raw["temperature_2m (°C)"], errors="coerce")
    out["CTX_TEMP"] = out["TEMP_C"]
    out["prcp"] = pd.to_numeric(raw["precipitation (mm)"], errors="coerce")
    out["CTX_PRCP"] = out["prcp"]
    out["WIND_SPEED_KMH"] = pd.to_numeric(raw["wind_speed_10m (km/h)"], errors="coerce")
    out["CTX_WSPD"] = out["WIND_SPEED_KMH"]
    out["coco"] = pd.to_numeric(raw["weather_code (wmo code)"], errors="coerce")
    out["CTX_COCO"] = out["coco"]
    out["CTX_WEATHER_SOURCE"] = "Meteostat_Hourly"

    out = out.dropna(subset=["__WEATHER_DT_LOCAL"]).copy()
    LOGGER.info("离线天气加载完成：%s rows=%d", weather_csv.as_posix(), len(out))
    return out


def _fetch_weather_2017(local_tz: str) -> pd.DataFrame:
    """
    拉取 2017 年逐小时天气。
    优先纽约中央公园附近站点，若失败则回退最近站点。
    """
    # 中央公园坐标
    cp_lat, cp_lon = 40.7829, -73.9654

    stations = Stations().nearby(cp_lat, cp_lon).fetch(20)
    if stations.empty:
        raise RuntimeError("Meteostat 未找到 NYC 附近可用站点")

    # Meteostat 常用 UTC 索引；统一拉全 2017 年
    start_utc = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    end_utc = pd.Timestamp("2017-12-31 23:59:59", tz="UTC")

    weather = None
    station_id = None
    last_err: Optional[BaseException] = None
    for sid in stations.index.tolist():
        try:
            cand = Hourly(sid, start_utc, end_utc).fetch()
            if cand is not None and len(cand) > 0:
                weather = cand
                station_id = sid
                break
        except Exception as e:
            last_err = e
            continue

    if weather is None or station_id is None:
        raise RuntimeError(f"Meteostat 无可用小时数据站点，last_err={last_err}")

    # 统一天气索引时区到纽约本地，便于和事故时间对齐
    idx = pd.DatetimeIndex(weather.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(local_tz)
    weather = weather.copy()
    weather.index = idx

    # 仅保留需求字段并严格重命名
    out = pd.DataFrame(index=weather.index)
    out["TEMP_C"] = weather.get("temp")
    out["CTX_TEMP"] = out["TEMP_C"]

    out["prcp"] = weather.get("prcp")
    out["CTX_PRCP"] = out["prcp"]

    out["WIND_SPEED_KMH"] = weather.get("wspd")
    out["CTX_WSPD"] = out["WIND_SPEED_KMH"]

    out["coco"] = weather.get("coco")
    out["CTX_COCO"] = out["coco"]

    out["CTX_WEATHER_SOURCE"] = "Meteostat_Hourly"

    out = out.reset_index().rename(columns={"index": "__WEATHER_DT_LOCAL"})
    LOGGER.info("天气数据拉取完成：station=%s, rows=%d", station_id, len(out))
    return out


def merge_weather(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.weather_csv and cfg.weather_csv.exists():
        weather = _load_offline_weather_csv(cfg.weather_csv, cfg.local_tz)
    else:
        weather = _fetch_weather_2017(cfg.local_tz)

    out = df.copy()

    # 事故时间按小时对齐（floor 或 round）
    if cfg.weather_rounding == "round":
        out["__HOUR_KEY"] = out["__CRASH_DT_LOCAL"].dt.round("h")
    else:
        out["__HOUR_KEY"] = out["__CRASH_DT_LOCAL"].dt.floor("h")

    weather["__HOUR_KEY"] = pd.to_datetime(weather["__WEATHER_DT_LOCAL"], errors="coerce")

    merged = out.merge(
        weather.drop(columns=["__WEATHER_DT_LOCAL"]),
        on="__HOUR_KEY",
        how="left",
    )

    # 少量缺失填 N/A（按你的要求）
    weather_cols = [
        "TEMP_C", "CTX_TEMP", "prcp", "CTX_PRCP", "WIND_SPEED_KMH", "CTX_WSPD", "coco", "CTX_COCO", "CTX_WEATHER_SOURCE"
    ]
    for c in weather_cols:
        if c in merged.columns:
            merged[c] = merged[c].where(merged[c].notna(), "N/A")

    LOGGER.info("天气匹配完成：缺失天气记录=%d", int((merged["CTX_WEATHER_SOURCE"] == "N/A").sum()))
    return merged


# =========================
# 模块 4：最终导出
# =========================


def run_pipeline(cfg: Config) -> pd.DataFrame:
    base_df = load_accident_data(cfg)
    osm_df = enrich_with_osm(base_df, cfg)
    final_df = merge_weather(osm_df, cfg)

    # 清理中间键，保留主数据列
    drop_cols = [c for c in ["__HOUR_KEY"] if c in final_df.columns]
    final_df = final_df.drop(columns=drop_cols)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    LOGGER.info("导出完成：%s (rows=%d, cols=%d)", cfg.output_csv.as_posix(), len(final_df), len(final_df.columns))
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 OSM + Meteostat 事故数据补全")
    parser.add_argument("--accident_csv", type=str, required=True, help="事故 CSV 路径")
    parser.add_argument("--pbf_path", type=str, required=True, help="本地 Geofabrik OSM PBF 文件路径")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 CSV 路径")

    parser.add_argument("--lon_col", type=str, default="LONGITUDE")
    parser.add_argument("--lat_col", type=str, default="LATITUDE")
    parser.add_argument("--date_col", type=str, default="CRASH DATE")
    parser.add_argument("--time_col", type=str, default="CRASH TIME")
    parser.add_argument("--datetime_col", type=str, default=None)

    parser.add_argument("--local_tz", type=str, default="America/New_York")
    parser.add_argument("--metric_crs", type=str, default="EPSG:2263")
    parser.add_argument("--signal_threshold_m", type=float, default=20.0)
    parser.add_argument("--weather_rounding", type=str, choices=["floor", "round"], default="floor")
    parser.add_argument("--weather_csv", type=str, default="weather/open-meteo-40.74N74.04W51m.csv", help="离线天气 CSV 路径")

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    cfg = Config(
        accident_csv=Path(args.accident_csv),
        pbf_path=Path(args.pbf_path),
        output_csv=Path(args.output_csv),
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        date_col=args.date_col,
        time_col=args.time_col,
        datetime_col=args.datetime_col,
        local_tz=args.local_tz,
        metric_crs=args.metric_crs,
        signal_threshold_m=float(args.signal_threshold_m),
        weather_rounding=args.weather_rounding,
        weather_csv=Path(args.weather_csv) if args.weather_csv else None,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
