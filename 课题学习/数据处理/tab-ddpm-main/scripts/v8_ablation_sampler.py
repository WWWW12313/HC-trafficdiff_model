"""
v8_ablation_sampler.py

3-way ablation sampler for v8 neuro-symbolic study.
Modes:
- free: no post constraints
- hard: enforce logic + commonsense by deterministic overwrites
- soft: LLM-prior rejection sampling on commonsense (K candidates), then hard logic only

Outputs:
- exp/nyc_crash_v8_ablation/synthetic_{mode}.csv
- exp/nyc_crash_v8_ablation/synthetic_{mode}_meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree

LOGGER = logging.getLogger("v8.ablation.sampler")


LOGIC_RULES: Dict[str, Any] = {
    "multi_vehicle_total": {
        "vehicle_col_1": "VEHICLE TYPE CODE 1",
        "vehicle_col_2": "VEHICLE TYPE CODE 2",
        "total_col": "TOTAL_VEHICLES",
        "min_total_if_both_present": 2,
    },
    "casualty_sum": {
        "inj_total": "NUMBER OF PERSONS INJURED",
        "inj_parts": [
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
        ],
        "kill_total": "NUMBER OF PERSONS KILLED",
        "kill_parts": [
            "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST KILLED",
        ],
    },
}

COMMONSENSE_RULES: Dict[str, Any] = {
    "snowplow_weather_season": {
        "vehicle_cols": [
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
            "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5",
        ],
        "snow_keywords": ["snow plow", "snowplow", "plow"],
        "winter_months": [12, 1, 2],
        "snow_weather_codes": [15, 16],
        "max_temp_for_snow_vehicle": 2.0,
    }
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def infer_time_period(hour: int) -> int:
    if 7 <= hour <= 9:
        return 1
    if 10 <= hour <= 15:
        return 2
    if 16 <= hour <= 19:
        return 3
    return 0


def normalize_speed_tag(v: Any) -> str:
    if pd.isna(v):
        return "25 mph"
    t = str(v).strip()
    return t if t else "25 mph"


def vehicle_present(s: pd.Series) -> pd.Series:
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return ~txt.isin(["", "nan", "none", "null", "unknown", "unspecified"])


def deterministic_weather(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    month = out["MONTH"].astype(int).to_numpy(dtype=np.int32)
    lat = out["LATITUDE"].to_numpy(dtype=np.float64)
    lon = out["LONGITUDE"].to_numpy(dtype=np.float64)
    ts = out["CRASH_DATE_TS"].to_numpy(dtype=np.float64)
    tmin = out["CRASH_TIME_MIN"].to_numpy(dtype=np.float64)

    month_temp = np.array([-1.0, 0.5, 5.0, 10.0, 16.0, 22.0, 25.0, 24.0, 20.0, 14.0, 8.0, 2.0], dtype=np.float64)
    month_prcp_prob = np.array([0.24, 0.22, 0.26, 0.25, 0.27, 0.29, 0.31, 0.30, 0.28, 0.25, 0.24, 0.25], dtype=np.float64)

    def frac(x: np.ndarray) -> np.ndarray:
        return x - np.floor(x)

    def noise(offset: float) -> np.ndarray:
        key = lat * 12.9898 + lon * 78.233 + (ts / 3600.0) * 37.719 + offset + seed * 0.01
        return frac(np.sin(key) * 43758.5453)

    base_temp = month_temp[np.clip(month - 1, 0, 11)]
    lat_adj = (lat - 40.70) * (-4.0)
    lon_adj = (lon + 73.95) * 1.2
    diurnal = np.cos((tmin / 1440.0) * 2 * np.pi) * 2.0
    temp = base_temp + lat_adj + lon_adj + diurnal + (noise(1.3) - 0.5) * 1.8

    prcp_prob = month_prcp_prob[np.clip(month - 1, 0, 11)]
    n_prcp = noise(4.7)
    rain = n_prcp < prcp_prob
    prcp = np.where(rain, (prcp_prob - n_prcp + 1e-3) * 9.0, 0.0)

    wind = np.clip(8.0 + np.abs(lat_adj) * 1.2 + noise(9.1) * 14.0, 0.0, 45.0)

    coco = np.full(len(out), 1, dtype=np.int32)
    coco[(prcp > 0.0) & (temp > 1.5)] = 7
    coco[(prcp > 0.0) & (temp <= 1.5)] = 15
    cloudy = (~rain) & (noise(15.3) < 0.35)
    coco[cloudy] = 3

    out["CTX_TEMP"] = temp.astype(np.float32)
    out["CTX_PRCP"] = prcp.astype(np.float32)
    out["CTX_WSPD"] = wind.astype(np.float32)
    out["CTX_COCO"] = coco.astype(np.int32)

    # mirror common weather names
    out["TEMP_C"] = out["CTX_TEMP"]
    out["prcp"] = out["CTX_PRCP"]
    out["WIND_SPEED_KMH"] = out["CTX_WSPD"]
    out["coco"] = out["CTX_COCO"]
    return out


def enrich_weather_with_meteostat(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Overwrite weather fields from Meteostat using (lat, lon, datetime)."""
    out = df.copy()
    stats = {"weather_api_rows": 0, "weather_fallback_rows": len(out), "station_queries": 0}

    try:
        from meteostat import Hourly, Stations
    except Exception as e:
        LOGGER.warning("Meteostat unavailable, keep fallback weather: %s", e)
        out["CTX_WEATHER_SOURCE"] = "fallback"
        return out, stats

    dt = pd.to_datetime(out["CRASH DATE"].astype(str) + " " + out["CRASH TIME"].astype(str), errors="coerce")
    dt = dt.dt.floor("h")

    coord_key = list(zip(out["LATITUDE"].round(2), out["LONGITUDE"].round(2)))
    station_map: Dict[Tuple[float, float], Optional[str]] = {}

    for key in sorted(set(coord_key)):
        lat, lon = float(key[0]), float(key[1])
        try:
            station = Stations().nearby(lat, lon).fetch(1)
            sid = str(station.index[0]) if not station.empty else None
            station_map[key] = sid
            stats["station_queries"] += 1
        except Exception:
            station_map[key] = None

    cache: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

    api_mask = np.zeros(len(out), dtype=bool)
    for i in range(len(out)):
        key = coord_key[i]
        sid = station_map.get(key)
        t = dt.iloc[i]
        if sid is None or pd.isna(t):
            continue

        month_key = f"{t.year:04d}-{t.month:02d}"
        ck = (sid, month_key)
        if ck not in cache:
            try:
                start = datetime(t.year, t.month, 1)
                end = datetime(t.year, t.month, 28, 23)
                data = Hourly(sid, start, end).fetch()
                month_map: Dict[str, Dict[str, float]] = {}
                for ts, row in data.iterrows():
                    ts_key = pd.to_datetime(str(ts), errors="coerce")
                    if pd.isna(ts_key):
                        continue
                    key_str = ts_key.strftime("%Y-%m-%d-%H")
                    month_map[key_str] = {
                        "temp": float(row.get("temp", np.nan)),
                        "prcp": float(row.get("prcp", np.nan)),
                        "wspd": float(row.get("wspd", np.nan)),
                        "coco": float(row.get("coco", np.nan)),
                    }
                cache[ck] = month_map
            except Exception:
                cache[ck] = {}

        ts_key = t.strftime("%Y-%m-%d-%H")
        roww = cache.get(ck, {}).get(ts_key)
        if roww is None:
            continue

        temp = roww.get("temp", np.nan)
        prcp = roww.get("prcp", np.nan)
        wspd = roww.get("wspd", np.nan)
        coco = roww.get("coco", np.nan)
        if np.isnan(temp) or np.isnan(prcp) or np.isnan(wspd) or np.isnan(coco):
            continue

        out.at[i, "CTX_TEMP"] = float(temp)
        out.at[i, "CTX_PRCP"] = float(prcp)
        out.at[i, "CTX_WSPD"] = float(wspd)
        out.at[i, "CTX_COCO"] = int(coco)
        out.at[i, "TEMP_C"] = float(temp)
        out.at[i, "prcp"] = float(prcp)
        out.at[i, "WIND_SPEED_KMH"] = float(wspd)
        out.at[i, "coco"] = int(coco)
        stats["weather_api_rows"] += 1
        api_mask[i] = True

    stats["weather_fallback_rows"] = int(len(out) - stats["weather_api_rows"])
    out["CTX_WEATHER_SOURCE"] = np.where(api_mask, "api", "fallback")
    return out, stats


def _edge_highway(v: Any) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else "residential"
    if pd.isna(v):
        return "residential"
    return str(v)


def _normalize_oneway(v: Any) -> int:
    txt = str(v).strip().lower()
    return 1 if txt in {"1", "true", "yes", "y", "t"} else 0


def _estimate_grid_n(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> int:
    lat_km = max((lat_max - lat_min) * 111.0, 0.1)
    lon_km = max((lon_max - lon_min) * 85.0, 0.1)
    area_km2 = lat_km * lon_km
    # Keep total area unchanged, only split requests into smaller batches.
    return 3 if area_km2 > 600.0 else 2


def _build_tiles(lat_min: float, lat_max: float, lon_min: float, lon_max: float, grid_n: int) -> List[Tuple[float, float, float, float]]:
    lat_edges = np.linspace(lat_min, lat_max, grid_n + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_n + 1)
    tiles: List[Tuple[float, float, float, float]] = []
    for i in range(grid_n):
        for j in range(grid_n):
            south = float(lat_edges[i])
            north = float(lat_edges[i + 1])
            west = float(lon_edges[j])
            east = float(lon_edges[j + 1])
            tiles.append((south, north, west, east))
    return tiles


def _build_overpass_query(tile: Tuple[float, float, float, float], timeout_s: int = 180) -> str:
    south, north, west, east = tile
    return (
        f"[out:json][timeout:{timeout_s}];"
        "("
        f'way["highway"]({south},{west},{north},{east});'
        f'node["highway"="traffic_signals"]({south},{west},{north},{east});'
        f'relation["highway"]({south},{west},{north},{east});'
        ");"
        "out body;"
        ">;"
        "out skel qt;"
    )


def _request_overpass_tile(
    tile: Tuple[float, float, float, float],
    endpoints: List[str],
    retries: int = 3,
    retry_wait_s: int = 10,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    query = _build_overpass_query(tile, timeout_s=timeout_s)
    last_err: Optional[BaseException] = None

    for ep in endpoints:
        ep = ep.rstrip("/")
        url = f"{ep}/interpreter"
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    timeout=timeout_s + 30,
                    verify=False,
                )
                code = int(resp.status_code)
                if code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict) and "elements" in payload:
                        return payload
                    raise RuntimeError(f"invalid Overpass payload from {ep}")

                if code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"overpass_http_{code}")

                raise RuntimeError(f"overpass_http_{code}: {resp.text[:200]}")
            except BaseException as e:
                last_err = e
                LOGGER.warning(
                    "Overpass failed endpoint=%s attempt=%d/%d tile=%s err=%s",
                    ep,
                    attempt,
                    retries,
                    tile,
                    e,
                )
                if attempt < retries:
                    time.sleep(retry_wait_s)
                else:
                    break

    raise RuntimeError(f"All Overpass endpoints failed for tile={tile}: {last_err}")


def _merge_overpass_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for payload in payloads:
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        for e in elements:
            typ = e.get("type")
            eid = e.get("id")
            if typ is None or eid is None:
                continue
            key = (str(typ), int(eid))
            if key not in merged:
                merged[key] = e
    return {"elements": list(merged.values())}


def _extract_osm_structures(merged_payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    elements = merged_payload.get("elements", []) if isinstance(merged_payload, dict) else []
    node_rows: List[Dict[str, Any]] = []
    way_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []

    for e in elements:
        typ = str(e.get("type", ""))
        tags = e.get("tags", {}) if isinstance(e.get("tags", {}), dict) else {}
        if typ == "node":
            lat = e.get("lat")
            lon = e.get("lon")
            if lat is None or lon is None:
                continue
            node_rows.append(
                {
                    "id": int(e.get("id")),
                    "lat": float(lat),
                    "lon": float(lon),
                    "highway": tags.get("highway", ""),
                }
            )
        elif typ == "way":
            nds = e.get("nodes", []) if isinstance(e.get("nodes", []), list) else []
            if len(nds) < 2:
                continue
            way_rows.append(
                {
                    "id": int(e.get("id")),
                    "nodes": [int(x) for x in nds],
                    "highway": tags.get("highway", "residential"),
                    "maxspeed": tags.get("maxspeed", "25 mph"),
                    "oneway": tags.get("oneway", 0),
                }
            )
        elif typ == "relation":
            rel_rows.append({"id": int(e.get("id")), "tags": tags})

    node_df = pd.DataFrame(node_rows)
    way_df = pd.DataFrame(way_rows)
    rel_df = pd.DataFrame(rel_rows)
    return node_df, way_df, rel_df


def _build_edge_midpoints(node_df: pd.DataFrame, way_df: pd.DataFrame) -> pd.DataFrame:
    if node_df.empty or way_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "highway", "maxspeed", "oneway"])

    node_map: Dict[int, Tuple[float, float]] = {}
    for _, row in node_df.iterrows():
        id_val = row["id"]
        lat_val = row["lat"]
        lon_val = row["lon"]
        if pd.isna(id_val) or pd.isna(lat_val) or pd.isna(lon_val):
            continue
        try:
            nid = int(id_val)
            lat = float(lat_val)
            lon = float(lon_val)
        except Exception:
            continue
        node_map[nid] = (lat, lon)

    edge_rows: List[Dict[str, Any]] = []
    for _, row in way_df.iterrows():
        raw_nodes = row.get("nodes", [])
        nds: List[int] = []
        if isinstance(raw_nodes, list):
            for nid in raw_nodes:
                try:
                    nds.append(int(nid))
                except Exception:
                    continue
        if len(nds) < 2:
            continue
        hw = _edge_highway(row.get("highway", "residential"))
        ms = normalize_speed_tag(row.get("maxspeed", "25 mph"))
        ow = _normalize_oneway(row.get("oneway", 0))
        for a, b in zip(nds[:-1], nds[1:]):
            pa = node_map.get(int(a))
            pb = node_map.get(int(b))
            if pa is None or pb is None:
                continue
            mid_lat = (pa[0] + pb[0]) / 2.0
            mid_lon = (pa[1] + pb[1]) / 2.0
            edge_rows.append(
                {
                    "lat": mid_lat,
                    "lon": mid_lon,
                    "highway": hw,
                    "maxspeed": ms,
                    "oneway": ow,
                }
            )

    return pd.DataFrame(edge_rows)


def enrich_osm_with_overpass(df: pd.DataFrame, cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Overwrite road context fields from OSM via tiled Overpass downloads."""
    out = df.copy()
    stats = {
        "osm_api_rows": 0,
        "osm_fallback_rows": len(out),
        "signals_detected": 0,
        "osm_nodes_total": 0,
        "osm_ways_total": 0,
        "osm_relations_total": 0,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_path = cache_dir / "nyc_overpass_merged.json"

    merged_payload: Optional[Dict[str, Any]] = None
    if merged_path.exists():
        try:
            merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
        except Exception:
            merged_payload = None

    if merged_payload is None:
        lat_min = float(df["LATITUDE"].min())
        lat_max = float(df["LATITUDE"].max())
        lon_min = float(df["LONGITUDE"].min())
        lon_max = float(df["LONGITUDE"].max())
        margin = 0.01

        lat_min = max(40.4774, lat_min - margin)
        lat_max = min(40.9176, lat_max + margin)
        lon_min = max(-74.2591, lon_min - margin)
        lon_max = min(-73.7004, lon_max + margin)

        grid_n = _estimate_grid_n(lat_min, lat_max, lon_min, lon_max)
        tiles = _build_tiles(lat_min, lat_max, lon_min, lon_max, grid_n)
        LOGGER.info(
            "OSM tiled download start: bbox=(%.4f,%.4f,%.4f,%.4f), grid=%dx%d, tiles=%d",
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            grid_n,
            grid_n,
            len(tiles),
        )

        endpoints = [
            "https://overpass-api.de/api",
            "https://overpass.kumi.systems/api",
            "https://overpass.private.coffee/api",
        ]

        payloads: List[Dict[str, Any]] = []
        for idx, tile in enumerate(tiles, start=1):
            LOGGER.info("Downloading tile %d/%d: %s", idx, len(tiles), tile)
            try:
                payload = _request_overpass_tile(
                    tile=tile,
                    endpoints=endpoints,
                    retries=3,
                    retry_wait_s=10,
                    timeout_s=180,
                )
                payloads.append(payload)
            except BaseException as e:
                LOGGER.warning("Tile download failed after retries, tile=%s err=%s", tile, e)

            # Random cool-down to reduce API pressure.
            time.sleep(random.uniform(3.5, 6.5))

        if not payloads:
            LOGGER.warning("All tiled Overpass requests failed; fallback to donor OSM fields")
            out["CTX_OSM_SOURCE"] = "fallback"
            return out, stats

        merged_payload = _merge_overpass_payloads(payloads)
        merged_path.write_text(json.dumps(merged_payload, ensure_ascii=False), encoding="utf-8")

    if merged_payload is None:
        out["CTX_OSM_SOURCE"] = "fallback"
        return out, stats

    node_df, way_df, rel_df = _extract_osm_structures(merged_payload)
    edge_df = _build_edge_midpoints(node_df, way_df)

    stats["osm_nodes_total"] = int(len(node_df))
    stats["osm_ways_total"] = int(len(way_df))
    stats["osm_relations_total"] = int(len(rel_df))
    LOGGER.info(
        "Merged OSM summary: nodes=%d, ways=%d, relations=%d",
        len(node_df),
        len(way_df),
        len(rel_df),
    )

    api_mask = np.zeros(len(out), dtype=bool)
    if not edge_df.empty:
        edge_coords = np.radians(edge_df[["lat", "lon"]].to_numpy(dtype=np.float64))
        crash_coords = np.radians(out[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        tree = BallTree(edge_coords, metric="haversine")
        _, nn_idx = tree.query(crash_coords, k=1)
        nearest_idx = nn_idx.flatten()

        highways = edge_df.iloc[nearest_idx]["highway"].fillna("residential").astype(str).tolist()
        maxspeeds = edge_df.iloc[nearest_idx]["maxspeed"].fillna("25 mph").astype(str).tolist()
        oneways = edge_df.iloc[nearest_idx]["oneway"].fillna(0).astype(int).tolist()

        out["CTX_HIGHWAY"] = highways
        out["CTX_MAXSPEED"] = maxspeeds
        out["CTX_ONEWAY"] = oneways
        out["OSM_TYPE"] = out["CTX_HIGHWAY"]
        out["OSM_SPEED_TAG"] = out["CTX_MAXSPEED"]
        out["OSM_ONEWAY"] = out["CTX_ONEWAY"]
        stats["osm_api_rows"] = len(out)
        api_mask[:] = True

    # signal distance
    sig = node_df[node_df.get("highway", pd.Series(dtype=object)) == "traffic_signals"]
    if len(sig) > 0:
        sig_coords = np.radians(sig[["lat", "lon"]].to_numpy(dtype=np.float64))
        tree = BallTree(sig_coords, metric="haversine")
        coords = np.radians(out[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        d, _ = tree.query(coords, k=1)
        dist_m = d.flatten() * 6371000.0
        out["CTX_DIST_TO_INTERSECTION"] = dist_m
        out["CTX_IS_SIGNALIZED"] = (dist_m < 50.0).astype(int)
        out["DIST_TO_SIGNAL_M"] = out["CTX_DIST_TO_INTERSECTION"]
        out["HAS_TRAFFIC_SIGNAL"] = out["CTX_IS_SIGNALIZED"]
        stats["signals_detected"] = int((out["CTX_IS_SIGNALIZED"] > 0).sum())

    stats["osm_fallback_rows"] = int(len(out) - stats["osm_api_rows"])
    out["CTX_OSM_SOURCE"] = np.where(api_mask, "api", "fallback")
    return out, stats


class RootAnchorGenerator:
    def __init__(self, ref: pd.DataFrame, seed: int):
        self.ref = ref.copy()
        self.rng = np.random.default_rng(seed)

        self.ref["LATITUDE"] = safe_num(self.ref.get("LATITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref["LONGITUDE"] = safe_num(self.ref.get("LONGITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        dt = pd.to_datetime(
            self.ref["CRASH DATE"].astype(str).str.strip() + " " + self.ref["CRASH TIME"].astype(str).str.strip(),
            errors="coerce",
        )
        self.ref["_DT"] = dt
        self.ref = self.ref.dropna(subset=["LATITUDE", "LONGITUDE", "_DT"]).copy()

        self.gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=seed)
        self.gmm.fit(self.ref[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))

    def sample(self, n: int) -> pd.DataFrame:
        latlon, _ = self.gmm.sample(n)
        dt_ref = self.ref["_DT"].to_numpy()
        idx = self.rng.integers(0, len(dt_ref), size=n)
        dt = pd.Series(pd.to_datetime(dt_ref[idx]))
        dt = dt + pd.to_timedelta(self.rng.integers(-20, 21, size=n), unit="m")

        out = pd.DataFrame(
            {
                "LATITUDE": latlon[:, 0],
                "LONGITUDE": latlon[:, 1],
                "CRASH_DATE_TS": (dt.astype("int64") // 10**9).astype(np.int64),
                "CRASH_TIME_MIN": (dt.dt.hour * 60 + dt.dt.minute).astype(np.int32),
                "DAY_OF_WEEK": dt.dt.weekday.astype(np.int32),
                "CRASH_TIME_PERIOD": dt.dt.hour.map(infer_time_period).astype(np.int32),
                "MONTH": dt.dt.month.astype(np.int32),
                "CRASH DATE": dt.dt.strftime("%m/%d/%Y"),
                "CRASH TIME": dt.dt.strftime("%H:%M"),
            }
        )
        return out


class DonorGenerator:
    def __init__(self, ref: pd.DataFrame):
        self.ref = ref.copy()
        self.ref.columns = self.ref.columns.str.strip()
        self.ref["LATITUDE"] = safe_num(self.ref.get("LATITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref["LONGITUDE"] = safe_num(self.ref.get("LONGITUDE", pd.Series(np.nan, index=self.ref.index)), default=np.nan)
        self.ref = self.ref.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
        coords = np.radians(self.ref[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        self.tree = BallTree(coords, metric="haversine")

    def query_knn_indices(self, base_df: pd.DataFrame, k: int) -> np.ndarray:
        coords = np.radians(base_df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=np.float64))
        _, nn_idx = self.tree.query(coords, k=k)
        return nn_idx

    def materialize_from_indices(
        self,
        base_df: pd.DataFrame,
        donor_idx: np.ndarray,
        preserve_generated_weather: bool = True,
    ) -> pd.DataFrame:
        out = base_df.copy()
        donor = self.ref.iloc[donor_idx].reset_index(drop=True)

        protected = {
            "LATITUDE", "LONGITUDE", "CRASH DATE", "CRASH TIME", "CRASH_DATE_TS", "CRASH_TIME_MIN",
            "DAY_OF_WEEK", "CRASH_TIME_PERIOD", "MONTH",
        }
        if preserve_generated_weather:
            protected.update({"CTX_TEMP", "CTX_PRCP", "CTX_WSPD", "CTX_COCO", "TEMP_C", "prcp", "WIND_SPEED_KMH", "coco"})

        for c in donor.columns:
            if c in protected:
                continue
            out[c] = donor[c].to_numpy(copy=False)

        if "OSM_SPEED_TAG" in out.columns:
            out["OSM_SPEED_TAG"] = out["OSM_SPEED_TAG"].map(normalize_speed_tag)
        return out


def commonsense_penalty_vectorized(df: pd.DataFrame) -> np.ndarray:
    """Lower is better; 0 means all commonsense checks pass."""
    n = len(df)
    penalty = np.zeros(n, dtype=np.float32)

    rule = COMMONSENSE_RULES["snowplow_weather_season"]
    veh_cols = [c for c in rule["vehicle_cols"] if c in df.columns]
    if veh_cols:
        snow_any = np.zeros(n, dtype=bool)
        for c in veh_cols:
            v = df[c].fillna("").astype(str).str.lower()
            snow_any |= v.str.contains("snow plow|snowplow|plow", regex=True).to_numpy()

        month = pd.to_datetime(df.get("CRASH DATE", pd.Series(["2017-01-01"] * n)), errors="coerce").dt.month.fillna(1).astype(int)
        temp = safe_num(df.get("CTX_TEMP", pd.Series(20.0, index=df.index)), default=20.0)
        prcp = safe_num(df.get("CTX_PRCP", pd.Series(0.0, index=df.index)), default=0.0)
        coco = safe_num(df.get("CTX_COCO", pd.Series(1, index=df.index)), default=1).astype(int)

        winter = month.isin(rule["winter_months"]).to_numpy()
        snowy = coco.isin(rule["snow_weather_codes"]).to_numpy() | ((temp <= rule["max_temp_for_snow_vehicle"]) & (prcp > 0.0)).to_numpy()
        allowed = winter | snowy

        penalty += (snow_any & (~allowed)).astype(np.float32) * 1.0

    return penalty


class GeminiCommonsenseScorer:
    """Row-wise Gemini scorer returning penalty in [0, 100].

    Uses the latest `google-genai` SDK (`from google import genai`).
    """

    def __init__(
        self,
        model_name: str = "models/gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        sleep_s: float = 0.0,
        retries: int = 2,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.temperature = float(temperature)
        self.sleep_s = float(sleep_s)
        self.retries = int(retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self._client: Optional[Any] = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not found")

        # Latest SDK package: google-genai
        from google import genai  # type: ignore

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_prompt(self, row_data: Dict[str, Any]) -> str:
        return (
            "You are a strict NYC traffic commonsense judge. "
            "Evaluate whether this synthetic crash row violates basic real-world constraints.\n\n"
            f"ROW: {json.dumps(row_data, ensure_ascii=False)}\n\n"
            "Rules:\n"
            "1) Season vs vehicle: if CTX_TEMP > 15 and snow-plow-like vehicle appears, this is severe violation.\n"
            "2) Night vs daylight labels: if around 03:00 and marked full daylight, this is violation.\n"
            "3) Consider physically and contextually implausible combinations.\n\n"
            "Return JSON only:\n"
            '{"penalty_score": <int 0..100>, "reason": "<one sentence>"}'
        )

    def score_row(self, row_data: Dict[str, Any]) -> float:
        client = self._ensure_client()
        prompt = self._build_prompt(row_data)

        # Lazy import to keep non-gemini mode dependency-light.
        from google.genai import types  # type: ignore

        last_err: Optional[BaseException] = None
        max_tries = max(self.retries, 1)
        for attempt in range(max_tries):
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=self.temperature,
                    ),
                )
                text = getattr(resp, "text", "") or ""
                payload = json.loads(text)
                score = float(payload.get("penalty_score", 100))
                score = max(0.0, min(100.0, score))
                if self.sleep_s > 0:
                    time.sleep(self.sleep_s)
                return score
            except Exception as e:
                last_err = e
                # Low-quota friendly backoff to avoid hammering API on 429 spikes.
                if attempt < max_tries - 1:
                    delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** attempt))
                    delay += random.uniform(0.0, 0.3)
                    time.sleep(delay)
                continue

        LOGGER.warning("Gemini scoring failed, fallback score=100. err=%s", last_err)
        return 100.0

    def score_dataframe(self, df: pd.DataFrame, row_fields: Optional[List[str]] = None) -> np.ndarray:
        if df.empty:
            return np.zeros(0, dtype=np.float32)

        fields = row_fields or [
            "CRASH DATE",
            "CRASH TIME",
            "MONTH",
            "CRASH_TIME_MIN",
            "CTX_TEMP",
            "CTX_PRCP",
            "CTX_COCO",
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
            "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5",
            "TOTAL_VEHICLES",
        ]

        scores = np.zeros(len(df), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            row_data: Dict[str, Any] = {}
            for c in fields:
                if c in df.columns:
                    v = row[c]
                    if pd.isna(v):
                        row_data[c] = None
                    else:
                        row_data[c] = v.item() if hasattr(v, "item") else v
            scores[i] = float(self.score_row(row_data))
        return scores


def commonsense_penalty_hybrid(
    df: pd.DataFrame,
    penalty_backend: str,
    gemini_scorer: Optional[GeminiCommonsenseScorer],
    gemini_max_rows: int,
) -> np.ndarray:
    """Return penalty vector; rule backend by default, optional Gemini overlay."""
    base_pen = commonsense_penalty_vectorized(df)
    if penalty_backend != "gemini" or gemini_scorer is None:
        return base_pen

    n = len(df)
    if n == 0:
        return base_pen

    # Full-row Gemini scoring is expensive; cap rows for practical runs.
    m = min(max(gemini_max_rows, 0), n)
    if m <= 0:
        LOGGER.warning("Gemini backend requested but gemini_max_rows<=0, fallback to rule-only")
        return base_pen

    idx = np.arange(m, dtype=np.int64)
    gem_df = df.iloc[idx].copy()
    gem_scores = gemini_scorer.score_dataframe(gem_df)
    # Convert 0..100 to 0..1 and combine with rule penalty conservatively.
    gem_norm = (gem_scores / 100.0).astype(np.float32)
    out_pen = base_pen.copy()
    out_pen[idx] = np.maximum(out_pen[idx], gem_norm)

    if m < n:
        LOGGER.warning(
            "Gemini applied to first %d/%d rows only; remaining rows use rule penalty.",
            m,
            n,
        )
    return out_pen


def apply_hard_corrections(df: pd.DataFrame, fallback_vehicle_pool: Optional[pd.Series]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    stats = {"commonsense_rows": 0, "logic_rows": 0, "casualty_rows": 0}

    # commonsense hard fix: replace snowplow in forbidden condition
    rule = COMMONSENSE_RULES["snowplow_weather_season"]
    veh_cols = [c for c in rule["vehicle_cols"] if c in out.columns]
    if veh_cols:
        month = pd.to_datetime(out.get("CRASH DATE", pd.Series(["2017-01-01"] * len(out))), errors="coerce").dt.month.fillna(1).astype(int)
        temp = safe_num(out.get("CTX_TEMP", pd.Series(20.0, index=out.index)), default=20.0)
        prcp = safe_num(out.get("CTX_PRCP", pd.Series(0.0, index=out.index)), default=0.0)
        coco = safe_num(out.get("CTX_COCO", pd.Series(1, index=out.index)), default=1).astype(int)

        winter = month.isin(rule["winter_months"]).to_numpy()
        snowy = coco.isin(rule["snow_weather_codes"]).to_numpy() | ((temp <= rule["max_temp_for_snow_vehicle"]) & (prcp > 0.0)).to_numpy()
        allowed = winter | snowy

        pool = fallback_vehicle_pool.dropna().astype(str).tolist() if fallback_vehicle_pool is not None else ["Sedan", "SUV", "Taxi"]
        if not pool:
            pool = ["Sedan", "SUV"]

        row_fixed = np.zeros(len(out), dtype=bool)
        for c in veh_cols:
            s = out[c].fillna("").astype(str)
            bad = s.str.lower().str.contains("snow plow|snowplow|plow", regex=True).to_numpy() & (~allowed)
            if bad.any():
                out.loc[bad, c] = np.random.choice(pool, size=int(bad.sum()), replace=True)
                row_fixed |= bad
        stats["commonsense_rows"] = int(row_fixed.sum())

    # logic hard fix: total_vehicles
    lg = LOGIC_RULES["multi_vehicle_total"]
    c1 = lg["vehicle_col_1"] if lg["vehicle_col_1"] in out.columns else None
    c2 = lg["vehicle_col_2"] if lg["vehicle_col_2"] in out.columns else None
    if c1 and c2:
        if lg["total_col"] not in out.columns:
            out[lg["total_col"]] = 1
        both = vehicle_present(out[c1]).to_numpy() & vehicle_present(out[c2]).to_numpy()
        total = safe_num(out[lg["total_col"]], default=1.0).round().astype(int)
        bad = both & (total.to_numpy() < int(lg["min_total_if_both_present"]))
        if bad.any():
            total.loc[bad] = int(lg["min_total_if_both_present"])
            out[lg["total_col"]] = total.astype(int)
        stats["logic_rows"] = int(bad.sum())

    if "IS_MULTI_VEHICLE" in out.columns and "TOTAL_VEHICLES" in out.columns:
        out["IS_MULTI_VEHICLE"] = (safe_num(out["TOTAL_VEHICLES"], default=1.0) >= 2).astype(int)

    # logic hard fix: casualty sums
    cs = LOGIC_RULES["casualty_sum"]
    for c in cs["inj_parts"] + cs["kill_parts"]:
        if c not in out.columns:
            out[c] = 0

    inj_parts = [safe_num(out[c], default=0.0).clip(lower=0).round().astype(int) for c in cs["inj_parts"]]
    kill_parts = [safe_num(out[c], default=0.0).clip(lower=0).round().astype(int) for c in cs["kill_parts"]]
    inj_sum = inj_parts[0] + inj_parts[1] + inj_parts[2]
    kill_sum = kill_parts[0] + kill_parts[1] + kill_parts[2]

    old_inj = safe_num(out.get(cs["inj_total"], pd.Series(0, index=out.index)), default=0.0).round().astype(int)
    old_kill = safe_num(out.get(cs["kill_total"], pd.Series(0, index=out.index)), default=0.0).round().astype(int)
    bad_sum = (old_inj.to_numpy() != inj_sum.to_numpy()) | (old_kill.to_numpy() != kill_sum.to_numpy())

    out[cs["inj_total"]] = inj_sum.astype(int)
    out[cs["kill_total"]] = kill_sum.astype(int)
    stats["casualty_rows"] = int(bad_sum.sum())

    return out, stats


def apply_logic_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    stats = {"logic_rows": 0, "casualty_rows": 0}

    lg = LOGIC_RULES["multi_vehicle_total"]
    c1 = lg["vehicle_col_1"] if lg["vehicle_col_1"] in out.columns else None
    c2 = lg["vehicle_col_2"] if lg["vehicle_col_2"] in out.columns else None
    if c1 and c2:
        if lg["total_col"] not in out.columns:
            out[lg["total_col"]] = 1
        both = vehicle_present(out[c1]).to_numpy() & vehicle_present(out[c2]).to_numpy()
        total = safe_num(out[lg["total_col"]], default=1.0).round().astype(int)
        bad = both & (total.to_numpy() < int(lg["min_total_if_both_present"]))
        if bad.any():
            total.loc[bad] = int(lg["min_total_if_both_present"])
            out[lg["total_col"]] = total.astype(int)
        stats["logic_rows"] = int(bad.sum())

    if "IS_MULTI_VEHICLE" in out.columns and "TOTAL_VEHICLES" in out.columns:
        out["IS_MULTI_VEHICLE"] = (safe_num(out["TOTAL_VEHICLES"], default=1.0) >= 2).astype(int)

    cs = LOGIC_RULES["casualty_sum"]
    for c in cs["inj_parts"] + cs["kill_parts"]:
        if c not in out.columns:
            out[c] = 0

    inj_parts = [safe_num(out[c], default=0.0).clip(lower=0).round().astype(int) for c in cs["inj_parts"]]
    kill_parts = [safe_num(out[c], default=0.0).clip(lower=0).round().astype(int) for c in cs["kill_parts"]]
    inj_sum = inj_parts[0] + inj_parts[1] + inj_parts[2]
    kill_sum = kill_parts[0] + kill_parts[1] + kill_parts[2]

    old_inj = safe_num(out.get(cs["inj_total"], pd.Series(0, index=out.index)), default=0.0).round().astype(int)
    old_kill = safe_num(out.get(cs["kill_total"], pd.Series(0, index=out.index)), default=0.0).round().astype(int)
    bad_sum = (old_inj.to_numpy() != inj_sum.to_numpy()) | (old_kill.to_numpy() != kill_sum.to_numpy())

    out[cs["inj_total"]] = inj_sum.astype(int)
    out[cs["kill_total"]] = kill_sum.astype(int)
    stats["casualty_rows"] = int(bad_sum.sum())

    return out, stats


def choose_soft_candidates(
    base_df: pd.DataFrame,
    donor_gen: DonorGenerator,
    k: int,
    penalty_backend: str,
    gemini_scorer: Optional[GeminiCommonsenseScorer],
    gemini_max_rows: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate K donor candidates per row and keep min-penalty candidate."""
    nn_idx = donor_gen.query_knn_indices(base_df, k=k)
    n = len(base_df)

    best_pen = np.full(n, np.inf, dtype=np.float32)
    best_df: Optional[pd.DataFrame] = None
    best_choice = np.zeros(n, dtype=np.int32)

    for j in range(k):
        cand = donor_gen.materialize_from_indices(base_df, nn_idx[:, j])
        pen = commonsense_penalty_hybrid(
            cand,
            penalty_backend=penalty_backend,
            gemini_scorer=gemini_scorer,
            gemini_max_rows=gemini_max_rows,
        )

        take = pen < best_pen
        if best_df is None:
            best_df = cand.copy()
            best_pen = pen
            best_choice[:] = j
        else:
            assert best_df is not None
            if take.any():
                best_df.loc[take, :] = cand.loc[take, :].to_numpy(copy=False)
                best_pen[take] = pen[take]
                best_choice[take] = j

    assert best_df is not None
    stats = {
        "soft_avg_penalty": float(np.mean(best_pen)),
        "soft_nonzero_penalty_rate": float(np.mean(best_pen > 0.0)),
        "soft_choice_mean": float(np.mean(best_choice)),
    }
    return best_df, stats


@dataclass
class SamplerConfig:
    mode: str
    reference_csv: str
    output_dir: str
    n_samples: int
    seed: int
    k_candidates: int
    weather_source: str
    osm_source: str
    penalty_backend: str
    quota_profile: str
    gemini_model: str
    gemini_max_rows: int
    gemini_sleep_s: float
    gemini_temperature: float
    gemini_retries: int
    gemini_backoff_base_s: float
    gemini_backoff_cap_s: float


def apply_quota_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Apply reproducible low-quota defaults unless user already overrides explicitly."""
    if args.quota_profile != "low":
        return args

    if args.weather_source == "api":
        args.weather_source = "donor"
    if args.osm_source == "api":
        args.osm_source = "donor"
    if args.gemini_max_rows == 200:
        args.gemini_max_rows = 60
    if abs(args.gemini_sleep_s - 0.0) < 1e-12:
        args.gemini_sleep_s = 1.5
    if abs(args.gemini_temperature - 0.1) < 1e-12:
        args.gemini_temperature = 0.0
    if args.gemini_retries == 2:
        args.gemini_retries = 4
    if abs(args.gemini_backoff_base_s - 1.0) < 1e-12:
        args.gemini_backoff_base_s = 2.0
    if abs(args.gemini_backoff_cap_s - 30.0) < 1e-12:
        args.gemini_backoff_cap_s = 45.0

    return args


def run(cfg: SamplerConfig) -> None:
    if cfg.mode not in {"free", "hard", "soft"}:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    LOGGER.info("Loading reference CSV: %s", cfg.reference_csv)
    ref = pd.read_csv(cfg.reference_csv)
    ref.columns = ref.columns.str.strip()

    LOGGER.info("Stage A: root generation")
    root_gen = RootAnchorGenerator(ref, seed=cfg.seed)
    base = root_gen.sample(cfg.n_samples)

    LOGGER.info("Stage B: weather context (%s)", cfg.weather_source)
    base = deterministic_weather(base, seed=cfg.seed)
    weather_stats: Dict[str, int] = {"weather_api_rows": 0, "weather_fallback_rows": len(base), "station_queries": 0}
    if cfg.weather_source == "api":
        LOGGER.warning("weather_source=api is disabled for reproducibility; force using donor weather context")
        cfg.weather_source = "donor"

    if cfg.weather_source == "donor":
        base["CTX_WEATHER_SOURCE"] = "donor_pending"
    else:
        base["CTX_WEATHER_SOURCE"] = "deterministic"

    donor = DonorGenerator(ref)
    fallback_pool = ref["VEHICLE TYPE CODE 1"] if "VEHICLE TYPE CODE 1" in ref.columns else None

    meta: Dict[str, Any] = {
        "mode": cfg.mode,
        "n_samples": int(cfg.n_samples),
        "k_candidates": int(cfg.k_candidates),
        "weather_source": cfg.weather_source,
        "osm_source": cfg.osm_source,
        "penalty_backend": cfg.penalty_backend,
        "quota_profile": cfg.quota_profile,
        "gemini_config": {
            "model": cfg.gemini_model,
            "max_rows": int(cfg.gemini_max_rows),
            "sleep_s": float(cfg.gemini_sleep_s),
            "temperature": float(cfg.gemini_temperature),
            "retries": int(cfg.gemini_retries),
            "backoff_base_s": float(cfg.gemini_backoff_base_s),
            "backoff_cap_s": float(cfg.gemini_backoff_cap_s),
        },
        "weather_stats": weather_stats,
    }

    if cfg.mode == "free":
        LOGGER.info("Mode FREE: no corrections")
        nn = donor.query_knn_indices(base, k=1).flatten()
        out = donor.materialize_from_indices(base, nn, preserve_generated_weather=(cfg.weather_source != "donor"))
        meta["correction_rate"] = 0.0

    elif cfg.mode == "hard":
        LOGGER.info("Mode HARD: hard commonsense + hard logic")
        nn = donor.query_knn_indices(base, k=1).flatten()
        tmp = donor.materialize_from_indices(base, nn, preserve_generated_weather=(cfg.weather_source != "donor"))
        out, stats = apply_hard_corrections(tmp, fallback_pool)
        corrected_rows = max(stats["commonsense_rows"], stats["logic_rows"], stats["casualty_rows"])
        meta["hard_stats"] = stats
        meta["correction_rate"] = float(corrected_rows / max(len(out), 1))

    else:
        LOGGER.info("Mode SOFT: rejection sampling by commonsense prior, then hard logic only")
        gem_scorer: Optional[GeminiCommonsenseScorer] = None
        if cfg.penalty_backend == "gemini":
            try:
                gem_scorer = GeminiCommonsenseScorer(
                    model_name=cfg.gemini_model,
                    temperature=cfg.gemini_temperature,
                    sleep_s=cfg.gemini_sleep_s,
                    retries=cfg.gemini_retries,
                    backoff_base_s=cfg.gemini_backoff_base_s,
                    backoff_cap_s=cfg.gemini_backoff_cap_s,
                )
                # Probe model init early for clearer logs.
                gem_scorer._ensure_client()
                LOGGER.info("Gemini penalty backend enabled: model=%s", cfg.gemini_model)
            except Exception as e:
                LOGGER.warning("Gemini backend unavailable, fallback to rule penalty: %s", e)
                gem_scorer = None

        tmp, soft_stats = choose_soft_candidates(
            base,
            donor,
            k=cfg.k_candidates,
            penalty_backend=cfg.penalty_backend,
            gemini_scorer=gem_scorer,
            gemini_max_rows=cfg.gemini_max_rows,
        )
        out, logic_stats = apply_logic_only(tmp)
        corrected_rows = max(logic_stats["logic_rows"], logic_stats["casualty_rows"])
        meta["soft_stats"] = soft_stats
        meta["logic_stats"] = logic_stats
        meta["correction_rate"] = float(corrected_rows / max(len(out), 1))

    # OSM API enrichment is applied after candidate selection so donor columns do not overwrite it.
    osm_stats: Dict[str, int] = {"osm_api_rows": 0, "osm_fallback_rows": len(out), "signals_detected": 0}
    if cfg.osm_source == "api":
        LOGGER.warning("osm_source=api is disabled for reproducibility; force using donor OSM context")
        cfg.osm_source = "donor"

    if cfg.osm_source == "api":
        out["CTX_OSM_SOURCE"] = "donor"
    else:
        out["CTX_OSM_SOURCE"] = "donor"
    meta["osm_stats"] = osm_stats

    # keep IS_* at tail
    is_cols = [c for c in out.columns if c.startswith("IS_")]
    other_cols = [c for c in out.columns if not c.startswith("IS_")]
    out = out[other_cols + is_cols]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"synthetic_{cfg.mode}.csv"
    out_meta = out_dir / f"synthetic_{cfg.mode}_meta.json"

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved: %s", out_csv.as_posix())
    LOGGER.info("Saved: %s", out_meta.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="v8 ablation sampler")
    parser.add_argument("--mode", type=str, required=True, choices=["free", "hard", "soft"])
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--output_dir", type=str, default="exp/nyc_crash_v8_ablation")
    parser.add_argument("--n_samples", type=int, default=159992)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_candidates", type=int, default=3)
    parser.add_argument("--weather_source", type=str, default="donor", choices=["api", "donor", "deterministic"])
    parser.add_argument("--osm_source", type=str, default="donor", choices=["api", "donor"])
    parser.add_argument("--penalty_backend", type=str, default="rule", choices=["rule", "gemini"])
    parser.add_argument("--quota_profile", type=str, default="none", choices=["none", "low"])
    parser.add_argument("--gemini_model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--gemini_max_rows", type=int, default=200)
    parser.add_argument("--gemini_sleep_s", type=float, default=0.0)
    parser.add_argument("--gemini_temperature", type=float, default=0.1)
    parser.add_argument("--gemini_retries", type=int, default=2)
    parser.add_argument("--gemini_backoff_base_s", type=float, default=1.0)
    parser.add_argument("--gemini_backoff_cap_s", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args = apply_quota_profile(args)

    setup_logging(verbose=args.verbose)
    cfg = SamplerConfig(
        mode=args.mode,
        reference_csv=args.reference_csv,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        k_candidates=args.k_candidates,
        weather_source=args.weather_source,
        osm_source=args.osm_source,
        penalty_backend=args.penalty_backend,
        quota_profile=args.quota_profile,
        gemini_model=args.gemini_model,
        gemini_max_rows=args.gemini_max_rows,
        gemini_sleep_s=args.gemini_sleep_s,
        gemini_temperature=args.gemini_temperature,
        gemini_retries=args.gemini_retries,
        gemini_backoff_base_s=args.gemini_backoff_base_s,
        gemini_backoff_cap_s=args.gemini_backoff_cap_s,
    )
    run(cfg)


if __name__ == "__main__":
    main()
