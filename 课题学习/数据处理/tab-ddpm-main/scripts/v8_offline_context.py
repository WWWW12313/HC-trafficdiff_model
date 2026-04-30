"""
v8_offline_context.py

Offline context matching for v8 NYC crash synthesis.

This module avoids online API calls by matching generated anchors against
pre-downloaded local weather and OSM network data.

Note:
- This module does not call any LLM API.

Main functions:
1) match_weather_offline
2) match_osm_offline
3) impute_all_offline_contexts
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("v8.offline_context")

NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LON_MIN = -74.26
NYC_LON_MAX = -73.70


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging format for offline context pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a unified datetime column named `CRASH_TS` exists.

    Accepted inputs:
    - Existing `CRASH_TS`
    - `CRASH_DATE_TS` epoch seconds
    - (`CRASH DATE`, `CRASH TIME`) string columns
    """
    out = df.copy()

    if "CRASH_TS" in out.columns:
        out["CRASH_TS"] = pd.to_datetime(out["CRASH_TS"], errors="coerce")
        return out

    if "CRASH_DATE_TS" in out.columns:
        out["CRASH_TS"] = pd.to_datetime(out["CRASH_DATE_TS"], unit="s", errors="coerce")
        return out

    if "CRASH DATE" in out.columns and "CRASH TIME" in out.columns:
        out["CRASH_TS"] = pd.to_datetime(
            out["CRASH DATE"].astype(str).str.strip() + " " + out["CRASH TIME"].astype(str).str.strip(),
            errors="coerce",
        )
        return out

    raise ValueError("Cannot infer timestamp. Need one of: CRASH_TS, CRASH_DATE_TS, or (CRASH DATE + CRASH TIME).")


def match_weather_offline(
    df_generated: pd.DataFrame,
    weather_csv: str = "data/nyc_weather_2017.csv",
    tolerance_hours: int = 2,
) -> pd.DataFrame:
    """
    Match generated rows with offline weather by nearest timestamp.

    Uses pandas.merge_asof with tolerance window.

    Expected weather CSV columns:
    - timestamp (or CRASH_TS)
    - CTX_TEMP / temp
    - CTX_PRCP / prcp
    - CTX_COCO / coco
    """
    t0 = time.perf_counter()
    LOGGER.info("Weather offline matching started: weather_csv=%s", weather_csv)

    t_parse = time.perf_counter()
    out = _ensure_timestamp_column(df_generated)
    if out["CRASH_TS"].isna().all():
        raise ValueError("All generated timestamps are NaT after parsing.")
    LOGGER.info("Parsed generated timestamps: rows=%d valid=%d time=%.3fs", len(out), int(out["CRASH_TS"].notna().sum()), time.perf_counter() - t_parse)

    weather_path = Path(weather_csv)
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather CSV not found: {weather_csv}")

    t_load = time.perf_counter()
    w = pd.read_csv(weather_path)
    w.columns = w.columns.str.strip()
    LOGGER.info("Loaded weather CSV: rows=%d cols=%d time=%.3fs", len(w), len(w.columns), time.perf_counter() - t_load)

    ts_col_candidates = ["timestamp", "CRASH_TS", "time", "datetime"]
    ts_col: Optional[str] = next((c for c in ts_col_candidates if c in w.columns), None)
    if ts_col is None:
        raise ValueError("Weather CSV must contain one timestamp column: timestamp/CRASH_TS/time/datetime")

    w["CRASH_TS"] = pd.to_datetime(w[ts_col], errors="coerce")
    w = w.dropna(subset=["CRASH_TS"]).copy()

    col_temp = "CTX_TEMP" if "CTX_TEMP" in w.columns else ("temp" if "temp" in w.columns else None)
    col_prcp = "CTX_PRCP" if "CTX_PRCP" in w.columns else ("prcp" if "prcp" in w.columns else None)
    col_coco = "CTX_COCO" if "CTX_COCO" in w.columns else ("coco" if "coco" in w.columns else None)

    if col_temp is None or col_prcp is None or col_coco is None:
        raise ValueError("Weather CSV missing required columns for temp/prcp/coco mapping")

    w = w[["CRASH_TS", col_temp, col_prcp, col_coco]].copy()
    w = w.rename(columns={col_temp: "CTX_TEMP", col_prcp: "CTX_PRCP", col_coco: "CTX_COCO"})
    w = w.sort_values("CRASH_TS").drop_duplicates(subset=["CRASH_TS"], keep="first")

    out = out.sort_values("CRASH_TS").copy()
    t_join = time.perf_counter()
    merged = pd.merge_asof(
        out,
        w,
        on="CRASH_TS",
        direction="nearest",
        tolerance=pd.Timedelta(hours=tolerance_hours),
    )
    LOGGER.info("merge_asof finished: tolerance=%dh time=%.3fs", tolerance_hours, time.perf_counter() - t_join)

    merged["CTX_TEMP"] = pd.to_numeric(merged["CTX_TEMP"], errors="coerce")
    merged["CTX_PRCP"] = pd.to_numeric(merged["CTX_PRCP"], errors="coerce")
    merged["CTX_COCO"] = pd.to_numeric(merged["CTX_COCO"], errors="coerce")

    # Mirror commonly used weather columns.
    merged["TEMP_C"] = merged["CTX_TEMP"]
    merged["prcp"] = merged["CTX_PRCP"]
    merged["coco"] = merged["CTX_COCO"]
    merged["CTX_WEATHER_SOURCE"] = np.where(merged["CTX_TEMP"].notna(), "Meteostat_Hourly_Offline", "missing_weather")

    matched = int(merged["CTX_WEATHER_SOURCE"].eq("Meteostat_Hourly_Offline").sum())
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        "Weather offline matching done: rows=%d matched=%d missing=%d time=%.3fs",
        len(merged),
        matched,
        len(merged) - matched,
        elapsed,
    )
    return merged


def match_osm_offline(
    df_generated: pd.DataFrame,
    graphml_path: str = "data/nyc_drive_network.graphml",
) -> pd.DataFrame:
    """
    Match generated coordinates to offline OSM graph context.

    Steps:
    - clip coords to strict NYC bounds
    - load graph from local GraphML
    - nearest_nodes for intersection-level features
    - nearest_edges for road-level features (e.g., highway)
    """
    t0 = time.perf_counter()
    LOGGER.info("OSM offline matching started: graphml=%s", graphml_path)

    out = df_generated.copy()
    if "LATITUDE" not in out.columns or "LONGITUDE" not in out.columns:
        raise ValueError("df_generated must contain LATITUDE and LONGITUDE")

    out["LATITUDE"] = pd.to_numeric(out["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(out["LONGITUDE"], errors="coerce")

    # Keep row count stable and only match valid coordinates.
    valid_mask = out["LATITUDE"].notna() & out["LONGITUDE"].notna()
    clipped_lat = out.loc[valid_mask, "LATITUDE"].clip(lower=NYC_LAT_MIN, upper=NYC_LAT_MAX)
    clipped_lon = out.loc[valid_mask, "LONGITUDE"].clip(lower=NYC_LON_MIN, upper=NYC_LON_MAX)
    lat_clipped_cnt = int((clipped_lat != out.loc[valid_mask, "LATITUDE"]).sum())
    lon_clipped_cnt = int((clipped_lon != out.loc[valid_mask, "LONGITUDE"]).sum())
    out.loc[valid_mask, "LATITUDE"] = clipped_lat
    out.loc[valid_mask, "LONGITUDE"] = clipped_lon
    LOGGER.info(
        "Coordinate clipping done: valid=%d invalid=%d lat_clipped=%d lon_clipped=%d",
        int(valid_mask.sum()),
        int((~valid_mask).sum()),
        lat_clipped_cnt,
        lon_clipped_cnt,
    )

    graph_path = Path(graphml_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"GraphML not found: {graphml_path}")

    import osmnx as ox

    t_graph = time.perf_counter()
    g = ox.load_graphml(graph_path)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(g, nodes=True, edges=True)
    LOGGER.info("Graph loaded: nodes=%d edges=%d time=%.3fs", len(nodes_gdf), len(edges_gdf), time.perf_counter() - t_graph)

    xs = out.loc[valid_mask, "LONGITUDE"].to_numpy(dtype=float)
    ys = out.loc[valid_mask, "LATITUDE"].to_numpy(dtype=float)

    # Initialize defaults for all rows.
    out["CTX_IS_SIGNALIZED"] = 0
    out["CTX_HIGHWAY"] = "residential"
    out["CTX_ONEWAY"] = 0
    out["CTX_MAXSPEED"] = "25 mph"
    out["CTX_DIST_TO_INTERSECTION"] = np.nan

    if len(xs) == 0:
        out["CTX_OSM_SOURCE"] = "missing_osm"
        LOGGER.warning("No valid coordinates for OSM matching. Returning defaults.")
        return out

    t_nn = time.perf_counter()
    nearest_nodes = ox.distance.nearest_nodes(g, X=xs, Y=ys)
    nearest_edges = ox.distance.nearest_edges(g, X=xs, Y=ys)
    LOGGER.info("Nearest node/edge lookup done: rows=%d time=%.3fs", len(xs), time.perf_counter() - t_nn)

    # Node-derived context.
    node_series = nodes_gdf.reindex(nearest_nodes)
    node_highway = node_series.get("highway", pd.Series(index=node_series.index, dtype=object))
    if isinstance(node_highway, pd.Series):
        node_highway = node_highway.astype(str)
    else:
        node_highway = pd.Series(["" for _ in range(len(out))])

    signalized = node_highway.str.contains("traffic_signals", case=False, na=False).astype(int)

    # Edge-derived context.
    def _edge_highway_to_str(v: object) -> str:
        if isinstance(v, list):
            return str(v[0]) if v else "residential"
        if v is None:
            return "residential"
        if isinstance(v, float) and np.isnan(v):
            return "residential"
        return str(v)

    edge_rows = edges_gdf.reindex(nearest_edges)
    edge_highway = edge_rows.get("highway", pd.Series(index=edge_rows.index, dtype=object)).map(_edge_highway_to_str)

    out.loc[valid_mask, "CTX_IS_SIGNALIZED"] = signalized.to_numpy(dtype=int)
    out.loc[valid_mask, "CTX_HIGHWAY"] = edge_highway.to_numpy(dtype=object)
    out["CTX_OSM_SOURCE"] = np.where(valid_mask, "Geofabrik_Offline_GraphML", "missing_osm")

    elapsed = time.perf_counter() - t0
    LOGGER.info("OSM offline matching done: rows=%d time=%.3fs", len(out), elapsed)
    return out


def impute_all_offline_contexts(
    df_generated: pd.DataFrame,
    weather_csv: str = "data/nyc_weather_2017.csv",
    graphml_path: str = "data/nyc_drive_network.graphml",
    weather_tolerance_hours: int = 2,
) -> pd.DataFrame:
    """
    Sequential wrapper: weather offline matching then OSM offline matching.

    Returns fully enriched dataframe with CTX_* context columns.
    """
    t0 = time.perf_counter()
    LOGGER.info("Offline context imputation pipeline started")

    w = match_weather_offline(
        df_generated=df_generated,
        weather_csv=weather_csv,
        tolerance_hours=weather_tolerance_hours,
    )
    full = match_osm_offline(
        df_generated=w,
        graphml_path=graphml_path,
    )

    elapsed = time.perf_counter() - t0
    LOGGER.info("Offline context imputation pipeline finished: rows=%d time=%.3fs", len(full), elapsed)
    return full


if __name__ == "__main__":
    setup_logging()
    LOGGER.info("v8_offline_context module loaded. Use impute_all_offline_contexts(...) in pipeline scripts.")
