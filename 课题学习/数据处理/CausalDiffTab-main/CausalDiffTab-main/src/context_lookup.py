"""
context_lookup.py — 路网与天气上下文查表模块
==============================================
实现 Section 1.8 中三种上下文模式的 historical_lookup 和辅助工具：

1. WeatherLookup
   - 从 Open-Meteo 小时级 CSV 加载天气数据
   - 根据 (date, hour) 最近邻匹配 TEMP_C / prcp / WIND_SPEED_KMH / WEATHER_CONDITION
   - 支持多站点聚合（均值/最近）

2. OSMContextLookup
   - 从 OSM GraphML 对生成的事故坐标查路网属性
   - 复用 prepare_2025_data.enrich_osm 核心逻辑，但以轻量缓存形式封装
   - 返回 DIST_TO_SIGNAL_M / HAS_TRAFFIC_SIGNAL / OSM_TYPE / OSM_ONEWAY / INFERRED_LANES

3. ContextPipeline
   - 统一入口：根据 context_mode 分发
     * "historical_lookup" → WeatherLookup + OSMContextLookup
     * "future_simulation" → 条件天气生成（Phase 3 实现）
     * "correction"        → lookup 同时保留原始生成列做 calibration loss

用法示例
--------
    from src.context_lookup import ContextPipeline, ContextConfig

    cfg = ContextConfig(
        context_mode="historical_lookup",
        osm_graphml="raw_data/osm/2024/nyc_drive_graph.graphml",
        weather_csv="raw_data/weather/2024/open-meteo-40.74N74.04W51m.csv",
    )
    pipe = ContextPipeline(cfg)
    df_enriched = pipe.enrich(df_synth, year=2024)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ContextConfig",
    "WeatherLookup",
    "OSMContextLookup",
    "ContextPipeline",
]

# WMO 天气码 → WEATHER_CONDITION 文字标签（简化映射）
WMO_TO_CONDITION: Dict[int, str] = {
    0:  "Clear",
    1:  "Clear",    2:  "Partly Cloudy",  3:  "Overcast",
    45: "Fog",      48: "Fog",
    51: "Drizzle",  53: "Drizzle",  55: "Drizzle",
    61: "Rain",     63: "Rain",     65: "Heavy Rain",
    71: "Snow",     73: "Snow",     75: "Heavy Snow",
    80: "Showers",  81: "Showers",  82: "Heavy Showers",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def _wmo_to_condition(code) -> str:
    try:
        return WMO_TO_CONDITION.get(int(code), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


# ──────────────────────────────────────────────────────────────────────────────
# 1. ContextConfig
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextConfig:
    """上下文管线配置。"""

    context_mode: str = "historical_lookup"
    """
    取值：
      "historical_lookup"  — 用真实 OSM/天气数据按时间+坐标补全
      "future_simulation"  — 条件生成天气（Phase 3 实现占位）
      "correction"         — lookup 同时保留原始生成列（用于 calibration loss）
    """

    # OSM GraphML 路径（可指定特定年份路网）
    osm_graphml: Optional[str] = None

    # Open-Meteo CSV 路径（可指定多个站点）
    weather_csv: Optional[str] = None
    weather_csv_list: List[str] = field(default_factory=list)

    # 信号灯 GeoJSON（可选）
    signals_geojson: Optional[str] = None

    # 是否保存 OSM 匹配缓存
    osm_cache_npz: Optional[str] = None

    # future_simulation scenario（Phase 3 占位）
    weather_scenario: str = "normal"
    # 支持: normal / heavy_rain / snow / heat_wave / extreme_weather

    # 日期列、时间列名（在 df 中）
    date_col: str = "CRASH DATE"
    time_col: str = "CRASH TIME"

    # 天气匹配回退方式
    weather_fallback: str = "nearest_hour"
    # 支持: "nearest_hour" / "daily_mean"

    # OSM 属性回退（缺失时）
    osm_fallback: str = "median"


# ──────────────────────────────────────────────────────────────────────────────
# 2. WeatherLookup
# ──────────────────────────────────────────────────────────────────────────────

class WeatherLookup:
    """
    Open-Meteo 小时级天气数据查表。

    支持多 CSV 合并（多站点取均值）。
    索引策略：按 (date, hour) 精确匹配；缺失时取当日最近小时。
    """

    def __init__(self, csv_paths: List[str | Path], verbose: bool = True):
        self._df = self._load(csv_paths, verbose)
        # 构建 (date_str, hour) → row_index 快速索引
        self._index: Dict[Tuple[str, int], int] = {}
        for i, row in self._df.iterrows():
            key = (str(row["_date"]), int(row["_hour"]))
            self._index[key] = i
        if verbose:
            print(f"[WeatherLookup] 加载 {len(self._df):,} 小时记录，"
                  f"覆盖 {self._df['_date'].nunique()} 天")

    @classmethod
    def _load(cls, csv_paths: List[str | Path], verbose: bool) -> pd.DataFrame:
        dfs = []
        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                if verbose:
                    print(f"[WeatherLookup] ⚠ 文件不存在: {p}")
                continue
            try:
                # Open-Meteo CSV 第 1-2 行是 header meta，第 3 行开始是列名
                raw = pd.read_csv(p, skiprows=2, low_memory=False)
                # 标准化列名
                col_map: Dict[str, str] = {}
                for c in raw.columns:
                    cl = c.lower()
                    if "time" in cl and "time" not in col_map:
                        col_map[c] = "time"
                    elif "temperature" in cl:
                        col_map[c] = "TEMP_C"
                    elif "precipitation" in cl:
                        col_map[c] = "prcp"
                    elif "wind_speed" in cl or "wind speed" in cl:
                        col_map[c] = "WIND_SPEED_KMH"
                    elif "weather_code" in cl or "weather code" in cl:
                        col_map[c] = "_wmo_code"
                raw = raw.rename(columns=col_map)
                raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
                raw = raw.dropna(subset=["time"])
                raw["_date"] = raw["time"].dt.strftime("%Y-%m-%d")
                raw["_hour"] = raw["time"].dt.hour
                dfs.append(raw[["_date", "_hour", "TEMP_C", "prcp",
                                "WIND_SPEED_KMH", "_wmo_code"]])
            except Exception as e:
                if verbose:
                    print(f"[WeatherLookup] ⚠ 读取失败 {p.name}: {e}")

        if not dfs:
            # 返回空表
            return pd.DataFrame(columns=["_date", "_hour", "TEMP_C", "prcp",
                                          "WIND_SPEED_KMH", "_wmo_code"])

        if len(dfs) == 1:
            df = dfs[0].reset_index(drop=True)
        else:
            # 多站点：按 (date, hour) 合并取均值
            combined = pd.concat(dfs, ignore_index=True)
            df = combined.groupby(["_date", "_hour"], as_index=False).mean(numeric_only=True)
            df["_date"] = combined.groupby(["_date", "_hour"])["_date"].first().values
        return df.reset_index(drop=True)

    def lookup(
        self,
        date_series: pd.Series,
        time_series: pd.Series,
    ) -> pd.DataFrame:
        """
        根据 CRASH DATE 和 CRASH TIME 序列返回天气特征 DataFrame。

        输出列：TEMP_C, prcp, WIND_SPEED_KMH, WEATHER_CONDITION
        """
        n = len(date_series)
        results = {
            "TEMP_C":             np.full(n, np.nan),
            "prcp":               np.full(n, np.nan),
            "WIND_SPEED_KMH":     np.full(n, np.nan),
            "WEATHER_CONDITION":  np.full(n, "Unknown", dtype=object),
        }

        if self._df.empty:
            return pd.DataFrame(results)

        # 解析日期和小时
        dates = pd.to_datetime(date_series, errors="coerce")
        try:
            hours_raw = pd.to_datetime(
                date_series.astype(str) + " " + time_series.astype(str),
                errors="coerce"
            ).dt.hour
        except Exception:
            hours_raw = pd.Series(np.zeros(n, dtype=int))

        for i in range(n):
            if pd.isna(dates.iloc[i]):
                continue
            date_str = dates.iloc[i].strftime("%Y-%m-%d")
            hour     = int(hours_raw.iloc[i]) if pd.notna(hours_raw.iloc[i]) else 0

            row_idx = self._index.get((date_str, hour))
            if row_idx is None:
                # 回退：在同一天找最近小时
                for dh in range(1, 12):
                    for sign in (-1, 1):
                        h2 = hour + sign * dh
                        if 0 <= h2 <= 23:
                            row_idx = self._index.get((date_str, h2))
                            if row_idx is not None:
                                break
                    if row_idx is not None:
                        break

            if row_idx is None:
                continue

            row = self._df.iloc[row_idx]
            results["TEMP_C"][i]         = float(row.get("TEMP_C", np.nan))
            results["prcp"][i]           = float(row.get("prcp", np.nan))
            results["WIND_SPEED_KMH"][i] = float(row.get("WIND_SPEED_KMH", np.nan))
            results["WEATHER_CONDITION"][i] = _wmo_to_condition(
                row.get("_wmo_code", 0)
            )

        return pd.DataFrame(results)

    def lookup_df(
        self,
        df: pd.DataFrame,
        date_col: str = "CRASH DATE",
        time_col: str = "CRASH TIME",
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """对整个 DataFrame 做天气查表，填充天气列。"""
        weather_df = self.lookup(df[date_col], df[time_col])
        df = df.copy()
        for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION"]:
            if col not in df.columns or overwrite:
                df[col] = weather_df[col].values
            else:
                # 仅填充缺失值
                mask = df[col].isna() | (df[col] == "Unknown")
                df.loc[mask, col] = weather_df[col].values[mask]
        return df

    @property
    def coverage_summary(self) -> dict:
        if self._df.empty:
            return {"n_hours": 0, "n_days": 0}
        return {
            "n_hours": len(self._df),
            "n_days":  self._df["_date"].nunique(),
            "date_min": self._df["_date"].min(),
            "date_max": self._df["_date"].max(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. OSMContextLookup
# ──────────────────────────────────────────────────────────────────────────────

class OSMContextLookup:
    """
    OSM 路网属性查表（封装 prepare_2025_data.enrich_osm 逻辑）。

    功能：
    - 对合成事故坐标（已完成道路吸附）查 OSM 路网属性
    - 返回：DIST_TO_SIGNAL_M, HAS_TRAFFIC_SIGNAL, OSM_TYPE, OSM_ONEWAY, INFERRED_LANES
    - 支持 fallback median（缺失时用训练集中位数填充）
    """

    def __init__(
        self,
        graphml_path: str | Path,
        signals_path: Optional[str | Path] = None,
        verbose: bool = True,
    ):
        self.graphml_path  = Path(graphml_path)
        self.signals_path  = Path(signals_path) if signals_path else None
        self.verbose       = verbose
        self._G            = None
        self._G_proj       = None
        self._sig_tree     = None
        self._sig_coords   = None
        self._fallback_medians: Dict[str, float] = {}

    def _load_graph(self):
        """懒加载路网（首次使用时加载）。"""
        if self._G is not None:
            return

        import osmnx as ox
        from sklearn.neighbors import BallTree

        def _compat_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("yes", "true", "1", "on")

        if self.verbose:
            print(f"[OSMContextLookup] 加载路网: {self.graphml_path.name} ...")

        self._G = ox.load_graphml(
            str(self.graphml_path),
            edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
            graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
        )
        self._G_proj = ox.project_graph(self._G, to_crs="EPSG:32618")

        # 构建信号灯 BallTree
        sig_xy = []
        for _, attrs in self._G_proj.nodes(data=True):
            tags = attrs.get("tags", {})
            hw   = tags.get("highway", "") if isinstance(tags, dict) else str(tags)
            if "traffic_signals" in str(hw):
                try:
                    sig_xy.append((float(attrs["x"]), float(attrs["y"])))
                except (KeyError, ValueError):
                    pass

        # 也尝试独立信号灯文件
        if self.signals_path and self.signals_path.exists():
            try:
                import geopandas as gpd
                sig_gdf = gpd.read_file(str(self.signals_path)).to_crs("EPSG:32618")
                extra_xy = list(zip(sig_gdf.geometry.x, sig_gdf.geometry.y))
                sig_xy.extend(extra_xy)
            except Exception:
                pass

        if sig_xy:
            self._sig_coords = np.array(sig_xy, dtype=float)
            self._sig_tree   = BallTree(self._sig_coords, leaf_size=15)
            if self.verbose:
                print(f"[OSMContextLookup] 信号灯节点: {len(sig_xy):,}")
        else:
            if self.verbose:
                print("[OSMContextLookup] ⚠ 未找到信号灯节点")

    def set_fallback_medians(self, medians: Dict[str, float]):
        """设置各 OSM 属性的回退中位数（从训练集统计）。"""
        self._fallback_medians = medians

    def enrich(self, df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
        """
        对 df 中 LATITUDE/LONGITUDE 查 OSM 属性。

        修改/新增列：DIST_TO_SIGNAL_M, HAS_TRAFFIC_SIGNAL, OSM_TYPE, OSM_ONEWAY, INFERRED_LANES
        """
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point

        self._load_graph()
        df = df.copy()

        valid_mask = df["LATITUDE"].notna() & df["LONGITUDE"].notna()
        df_valid   = df[valid_mask].copy()

        if len(df_valid) == 0:
            return df

        # ── 信号灯距离 ─────────────────────────────────────────────────────
        dist_arr = np.full(len(df_valid), np.nan)
        has_sig  = np.zeros(len(df_valid), dtype=int)

        if self._sig_tree is not None:
            geometry = [Point(lon, lat)
                        for lon, lat in zip(df_valid["LONGITUDE"], df_valid["LATITUDE"])]
            gdf      = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
            gdf_proj = gdf.to_crs("EPSG:32618")
            pt_coords = np.column_stack([
                np.asarray(gdf_proj.geometry.x, dtype=float),
                np.asarray(gdf_proj.geometry.y, dtype=float),
            ])
            dists, _ = self._sig_tree.query(pt_coords, k=1)
            dist_arr = dists[:, 0]
            has_sig  = (dist_arr < 30).astype(int)

        df.loc[valid_mask, "DIST_TO_SIGNAL_M"]   = dist_arr
        df.loc[valid_mask, "HAS_TRAFFIC_SIGNAL"] = has_sig

        # ── 最近道路属性 ───────────────────────────────────────────────────
        if self.verbose:
            print(f"[OSMContextLookup] 匹配 {len(df_valid):,} 个坐标的 OSM 边属性...")

        ne_edges = ox.nearest_edges(
            self._G,
            X=np.asarray(df_valid["LONGITUDE"], dtype=float),
            Y=np.asarray(df_valid["LATITUDE"],  dtype=float),
        )

        osm_type_list, osm_lanes_list, osm_oneway_list = [], [], []
        for u, v, key in ne_edges:
            edge = self._G.get_edge_data(u, v, key) or {}

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
            if "motorway" in h or "trunk" in h: return 3
            if "primary"  in h: return 2
            return 1

        df.loc[valid_mask, "OSM_TYPE"]       = osm_type_list
        df.loc[valid_mask, "OSM_ONEWAY"]     = [int(b) for b in osm_oneway_list]
        df.loc[valid_mask, "INFERRED_LANES"] = [
            _infer_lanes(l, t) for l, t in zip(osm_lanes_list, osm_type_list)
        ]

        # ── 回退填充 ───────────────────────────────────────────────────────
        osm_num_cols = ["DIST_TO_SIGNAL_M", "INFERRED_LANES"]
        for col in osm_num_cols:
            if col in df.columns:
                fb = self._fallback_medians.get(col, np.nan)
                if not np.isnan(fb):
                    df[col] = df[col].fillna(fb)

        if self.verbose:
            print("[OSMContextLookup] OSM 属性补全完成")
        return df

    @staticmethod
    def compute_fallback_medians(train_df: pd.DataFrame) -> Dict[str, float]:
        """从训练集计算 OSM 属性的中位数，用于缺失回退。"""
        cols   = ["DIST_TO_SIGNAL_M", "INFERRED_LANES"]
        result = {}
        for c in cols:
            if c in train_df.columns:
                vals = pd.to_numeric(train_df[c], errors="coerce").dropna()
                if len(vals) > 0:
                    result[c] = float(vals.median())
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. ContextPipeline — 统一入口
# ──────────────────────────────────────────────────────────────────────────────

class ContextPipeline:
    """
    统一的上下文补全管线。

    根据 ContextConfig.context_mode 分发处理：
    - "historical_lookup"：天气查表 + OSM 查表
    - "future_simulation"：Phase 3 实现（当前占位，使用季节统计近似）
    - "correction"       ：lookup 同时保留原始列，计算 calibration 误差
    """

    def __init__(self, cfg: ContextConfig, verbose: bool = True):
        self.cfg     = cfg
        self.verbose = verbose
        self._weather: Optional[WeatherLookup]    = None
        self._osm:     Optional[OSMContextLookup] = None
        self._loaded  = False

    def _lazy_load(self):
        if self._loaded:
            return
        cfg = self.cfg

        # 天气
        csv_paths = []
        if cfg.weather_csv:
            csv_paths.append(cfg.weather_csv)
        csv_paths.extend(cfg.weather_csv_list)
        if csv_paths:
            self._weather = WeatherLookup(csv_paths, verbose=self.verbose)

        # OSM
        if cfg.osm_graphml and Path(cfg.osm_graphml).exists():
            self._osm = OSMContextLookup(
                cfg.osm_graphml,
                signals_path=cfg.signals_geojson,
                verbose=self.verbose,
            )

        self._loaded = True

    def set_osm_fallback_medians(self, medians: Dict[str, float]):
        """传入训练集统计的中位数（OSM 属性回退）。"""
        self._lazy_load()
        if self._osm:
            self._osm.set_fallback_medians(medians)

    def enrich(
        self,
        df: pd.DataFrame,
        year: Optional[int] = None,
        keep_original_cols: bool = False,
    ) -> pd.DataFrame:
        """
        对合成样本 DataFrame 补全上下文。

        参数
        ----
        df                  : 含 LATITUDE/LONGITUDE/CRASH DATE/CRASH TIME 的合成样本
        year                : 目标年份（future_simulation 模式使用）
        keep_original_cols  : correction 模式下保留原始扩散生成的天气列（后缀 _GEN）

        返回
        ----
        补全后的 DataFrame
        """
        self._lazy_load()
        mode = self.cfg.context_mode

        if mode == "historical_lookup":
            return self._enrich_historical(df)
        elif mode == "future_simulation":
            return self._enrich_future(df, year=year)
        elif mode == "correction":
            return self._enrich_correction(df)
        else:
            warnings.warn(f"[ContextPipeline] 未知 context_mode='{mode}'，回退到 historical_lookup")
            return self._enrich_historical(df)

    def _enrich_historical(self, df: pd.DataFrame) -> pd.DataFrame:
        """historical_lookup 模式：天气 + OSM 真实数据查表。"""
        date_col = self.cfg.date_col
        time_col = self.cfg.time_col

        # 天气查表
        if self._weather is not None:
            if date_col in df.columns and time_col in df.columns:
                df = self._weather.lookup_df(df, date_col=date_col, time_col=time_col)
            else:
                if self.verbose:
                    print(f"[ContextPipeline] ⚠ 未找到 {date_col}/{time_col}，跳过天气查表")

        # OSM 查表
        if self._osm is not None:
            df = self._osm.enrich(df)

        return df

    def _enrich_future(self, df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
        """
        future_simulation 模式（Phase 3 占位实现）。
        当前使用季节统计近似生成天气，路网用最新 OSM 数据。
        """
        if self.verbose:
            print("[ContextPipeline] future_simulation 模式（Phase 3 占位：季节统计近似）")

        # ── 季节→天气近似（临时实现，Phase 3 替换为条件生成器）────────────
        scenario = self.cfg.weather_scenario

        SEASON_WEATHER_APPROX = {
            "winter": {"TEMP_C": -1.0,  "prcp": 2.5,  "WIND_SPEED_KMH": 18.0, "WEATHER_CONDITION": "Snow"},
            "spring": {"TEMP_C": 13.0,  "prcp": 1.5,  "WIND_SPEED_KMH": 14.0, "WEATHER_CONDITION": "Partly Cloudy"},
            "summer": {"TEMP_C": 27.0,  "prcp": 1.2,  "WIND_SPEED_KMH": 12.0, "WEATHER_CONDITION": "Clear"},
            "autumn": {"TEMP_C": 12.0,  "prcp": 2.0,  "WIND_SPEED_KMH": 15.0, "WEATHER_CONDITION": "Partly Cloudy"},
        }
        SCENARIO_OVERRIDE = {
            "heavy_rain":    {"prcp": 20.0, "WEATHER_CONDITION": "Heavy Rain", "WIND_SPEED_KMH": 25.0},
            "snow":          {"TEMP_C": -5.0, "prcp": 10.0, "WEATHER_CONDITION": "Heavy Snow"},
            "heat_wave":     {"TEMP_C": 38.0, "WEATHER_CONDITION": "Clear", "prcp": 0.0},
            "extreme_weather":{"TEMP_C": -10.0,"prcp": 30.0,"WIND_SPEED_KMH": 60.0,"WEATHER_CONDITION": "Thunderstorm"},
        }

        df = df.copy()
        for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION"]:
            if col not in df.columns:
                df[col] = np.nan if col != "WEATHER_CONDITION" else "Unknown"

        # 按行填充季节近似
        if "SEASON" in df.columns:
            for season, vals in SEASON_WEATHER_APPROX.items():
                mask = df["SEASON"].str.lower() == season
                for col, val in vals.items():
                    df.loc[mask & df[col].isna(), col] = val
                    if col == "WEATHER_CONDITION":
                        df.loc[mask & (df[col] == "Unknown"), col] = val

        # Scenario 覆盖
        override = SCENARIO_OVERRIDE.get(scenario, {})
        for col, val in override.items():
            df[col] = val

        # OSM：用现有路网数据
        if self._osm is not None:
            df = self._osm.enrich(df)

        return df

    def _enrich_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        correction 模式：
        1. 保留原始生成天气列（重命名为 _GEN 后缀）
        2. 用真实数据查表覆盖
        3. 计算 calibration 误差统计
        """
        df = df.copy()
        gen_cols = ["TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION"]

        # 保留原始生成值
        for col in gen_cols:
            if col in df.columns:
                df[f"{col}_GEN"] = df[col].copy()

        # 真实查表覆盖
        df = self._enrich_historical(df)

        # 计算误差统计
        calib_stats: Dict[str, dict] = {}
        for col in ["TEMP_C", "prcp", "WIND_SPEED_KMH"]:
            gen_col  = f"{col}_GEN"
            if gen_col in df.columns and col in df.columns:
                gen_vals  = pd.to_numeric(df[gen_col], errors="coerce")
                real_vals = pd.to_numeric(df[col],     errors="coerce")
                valid     = gen_vals.notna() & real_vals.notna()
                if valid.sum() > 0:
                    diff = (gen_vals - real_vals)[valid]
                    calib_stats[col] = {
                        "mae":    float(diff.abs().mean()),
                        "bias":   float(diff.mean()),
                        "rmse":   float(np.sqrt((diff ** 2).mean())),
                    }

        df.attrs["calibration_stats"] = calib_stats
        if self.verbose and calib_stats:
            print("[ContextPipeline] Calibration 误差统计:")
            for col, stats in calib_stats.items():
                print(f"  {col}: MAE={stats['mae']:.3f}, bias={stats['bias']:.3f}")

        return df

    def summary(self) -> dict:
        """返回管线摘要信息。"""
        self._lazy_load()
        out = {"context_mode": self.cfg.context_mode}
        if self._weather:
            out["weather"] = self._weather.coverage_summary
        else:
            out["weather"] = None
        out["osm"] = {
            "graphml": str(self.cfg.osm_graphml),
            "loaded":  self._osm is not None and self._osm._G is not None,
        }
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 辅助：根据项目根目录和目标年份自动推断数据路径
# ──────────────────────────────────────────────────────────────────────────────

def auto_config(
    year: int = 2024,
    root: Optional[str | Path] = None,
    context_mode: str = "historical_lookup",
    weather_scenario: str = "normal",
) -> ContextConfig:
    """
    根据年份自动推断 OSM + 天气文件路径，返回 ContextConfig。

    文件约定：
      raw_data/osm/{year}/nyc_drive_graph.graphml
      raw_data/weather/{year}/open-meteo-*.csv
    """
    if root is None:
        root = Path(__file__).resolve().parent.parent
    root = Path(root)

    osm_path = root / "raw_data" / "osm" / str(year) / "nyc_drive_graph.graphml"
    if not osm_path.exists():
        # 回退到全局路径
        osm_path = root / "raw_data" / "osm" / "nyc_drive_graph.graphml"

    weather_dir  = root / "raw_data" / "weather" / str(year)
    weather_csvs = list(weather_dir.glob("open-meteo-*.csv")) if weather_dir.exists() else []
    weather_csv  = str(weather_csvs[0]) if weather_csvs else None

    return ContextConfig(
        context_mode     = context_mode,
        osm_graphml      = str(osm_path) if osm_path.exists() else None,
        weather_csv      = weather_csv,
        weather_scenario = weather_scenario,
    )
