"""
Stage 2 离线物理上下文 — 占位与接口说明
========================================
根据分层管线：在 Stage1 生成 LAT/LON、时间锚点后，本模块负责补全
TEMP_C、PRCP、WEATHER、OSM_TYPE、REAL_SPEED_LIMIT 等，以及由经纬度反查 BOROUGH。

实现约束（实验环境内完成，不依赖删库或改全局设置）：
- 使用本地 .pbf（OSM）、Open-Meteo 历史 CSV、可选 geopandas 形状文件做 Point-in-Polygon。
- 所有可选依赖用 try/import，缺失时明确报错信息，避免静默失败。

当前状态：仅定义函数签名与数据契约，具体 IO 在后续迭代中填充。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Stage1Anchor:
    """Stage1 输出（或等价列）的最小字段。"""

    latitude: float
    longitude: float
    season_code: int
    day_of_week: int
    time_period: int
    time_sin: float
    time_cos: float


@dataclass
class Stage2Context:
    """Stage2 补全后的环境/路况字段（与 data_processor 列名对齐）。"""

    temp_c: Optional[float] = None
    prcp: Optional[float] = None
    weather_condition: Optional[str] = None
    osm_type: Optional[str] = None
    real_speed_limit: Optional[float] = None
    borough: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def lookup_stage2_context(
    anchor: Stage1Anchor,
    *,
    osm_pbf_path: Optional[str] = None,
    meteo_csv_path: Optional[str] = None,
    borough_gpkg_path: Optional[str] = None,
) -> Stage2Context:
    """
    由时空锚点查询离线库，返回 Stage2 上下文。

    尚未实现：请在本文件内接入 osmnx / pyrosm / geopandas 与气象表 join。
    """
    raise NotImplementedError(
        "Stage2 离线查询未实现：请配置 osm_pbf_path、meteo_csv_path 后在此函数内填充逻辑。"
    )


def borough_from_lon_lat(
    lon: float,
    lat: float,
    borough_gpkg_path: str,
) -> str:
    """
    Point-in-Polygon 反查 BOROUGH；需 geopandas + 本地行政区边界数据。
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError("请先在 crashgen 环境中安装 geopandas、shapely") from e

    gdf = gpd.read_file(borough_gpkg_path)
    pt = Point(lon, lat)
    hit = gdf[gdf.contains(pt)]
    if hit.empty:
        return "UNKNOWN"
    name_col = "boro_name" if "boro_name" in hit.columns else hit.columns[0]
    return str(hit.iloc[0][name_col])
