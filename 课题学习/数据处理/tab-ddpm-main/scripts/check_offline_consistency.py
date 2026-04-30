"""
Phase 0 offline consistency checker.

What this script does:
1) Read raw NYC 2017 crash CSV.
2) Rebuild offline context features using local OSM PBF + Open-Meteo weather CSV.
3) Join with nyc_2017_pristine_v8.csv and compute discrepancy rates.
4) If discrepancy is acceptable, write v9_offline_pristine.csv as new offline source.

Usage example:
C:/Users/Admin/anaconda3/envs/crashgen/python.exe scripts/check_offline_consistency.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pyrosm import OSM
from sklearn.neighbors import BallTree

LOGGER = logging.getLogger("phase0.offline_consistency")
EARTH_RADIUS_M = 6371000.0


@dataclass
class Config:
    raw_csv: Path
    pbf_path: Path
    weather_csv: Path
    pristine_csv: Path
    report_json: Path
    remapped_csv: Path
    v9_output_csv: Path
    join_key: str = "COLLISION_ID"
    signal_threshold_m: float = 20.0
    weather_rounding: str = "floor"
    accept_rate: float = 0.35
    force_overwrite: bool = False
    sample_n: int = 0


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return _repo_root().parent


def _load_csv_strip(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _build_crash_dt(df: pd.DataFrame) -> pd.Series:
    if "CRASH DATE" not in df.columns or "CRASH TIME" not in df.columns:
        raise ValueError("raw/pristine csv must include CRASH DATE and CRASH TIME")
    return pd.to_datetime(
        df["CRASH DATE"].astype(str).str.strip() + " " + df["CRASH TIME"].astype(str).str.strip(),
        errors="coerce",
    )


def _safe_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return txt.isin(["1", "true", "yes", "y", "t"])


def _haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return EARTH_RADIUS_M * c


def _parse_speed_tag_to_mph(v: object) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    txt = str(v).strip().lower()
    if not txt or txt in {"nan", "none", "n/a", "na", "unknown"}:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _format_speed_tag_from_mph(v: object) -> str:
    mph = _parse_speed_tag_to_mph(v)
    if mph is None:
        return "N/A"
    return f"{int(round(float(mph)))} mph"


def _highway_default_mph(osm_type: object) -> float:
    t = str(osm_type).strip().lower()
    # NYC fallback defaults; residential/unknown roads use 25 mph by default.
    if "motorway" in t:
        return 50.0
    if "trunk" in t:
        return 35.0
    if "primary" in t:
        return 30.0
    if "secondary" in t:
        return 25.0
    if "tertiary" in t:
        return 25.0
    if "living_street" in t:
        return 15.0
    return 25.0


def _resolve_primary_speed_tag(row: pd.Series) -> str:
    for key in [
        "OSM_SPEED_TAG_RAW",
        "OSM_MAXSPEED_FWD_RAW",
        "OSM_MAXSPEED_BWD_RAW",
        "OSM_SOURCE_MAXSPEED_RAW",
    ]:
        if key not in row.index:
            continue
        txt = _extract_first_text(row.get(key), default="N/A")
        if txt != "N/A":
            return txt
    return "N/A"


def _apply_speed_limit_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "OSM_SPEED_TAG" not in out.columns:
        out["OSM_SPEED_TAG"] = "N/A"

    # Step 1) Parse speed from resolved OSM tags.
    parsed = out["OSM_SPEED_TAG"].map(_parse_speed_tag_to_mph)
    out["REAL_SPEED_LIMIT"] = pd.to_numeric(pd.Series(parsed, index=out.index), errors="coerce")
    out["SPEED_SOURCE"] = np.where(out["REAL_SPEED_LIMIT"].notna(), "osm_tag", "unknown")

    # Step 2) Highway-level median from valid tag-based speeds.
    if "OSM_TYPE" in out.columns:
        med = out.loc[out["REAL_SPEED_LIMIT"].notna()].groupby("OSM_TYPE")["REAL_SPEED_LIMIT"].median()
        missing = out["REAL_SPEED_LIMIT"].isna()
        if missing.any() and not med.empty:
            mapped = out.loc[missing, "OSM_TYPE"].map(med)
            use_idx = mapped[mapped.notna()].index
            out.loc[use_idx, "REAL_SPEED_LIMIT"] = mapped.loc[use_idx].astype(float)
            out.loc[use_idx, "SPEED_SOURCE"] = "highway_median"

    # Step 3) NYC defaults by road class.
    missing = out["REAL_SPEED_LIMIT"].isna()
    if missing.any():
        out.loc[missing, "REAL_SPEED_LIMIT"] = out.loc[missing, "OSM_TYPE"].map(_highway_default_mph).astype(float)
        out.loc[missing, "SPEED_SOURCE"] = "nyc_default"

    # Keep compatibility: standardized string form for downstream v8-style fields.
    out["OSM_SPEED_TAG"] = out["REAL_SPEED_LIMIT"].map(_format_speed_tag_from_mph)
    return out


def _coco_to_weather_text(v: object) -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "Unknown"
        code_raw = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(code_raw):
            return "Unknown"
        code = int(float(code_raw))
    except Exception:
        return "Unknown"

    # Lightweight WMO-style labels for compatibility fields.
    if code in {0, 1}:
        return "Clear"
    if code in {2, 3}:
        return "Cloudy"
    if code in {45, 48}:
        return "Fog"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if code in {95, 96, 99}:
        return "Thunderstorm"
    return "Other"


def align_to_pristine_schema(remapped_df: pd.DataFrame, pristine_csv: Path) -> pd.DataFrame:
    """Return a v8-compatible dataframe with column order aligned to pristine csv."""
    out = remapped_df.copy()

    # Remove internal columns from remapping process.
    drop_internal = ["__CRASH_DT", "__HOUR_KEY"]
    out = out.drop(columns=[c for c in drop_internal if c in out.columns], errors="ignore")

    # Fill compatibility weather text columns if absent.
    if "WEATHER_CONDITION" not in out.columns:
        out["WEATHER_CONDITION"] = out.get("coco", pd.Series(index=out.index)).map(_coco_to_weather_text)
    if "REAL_WEATHER" not in out.columns:
        out["REAL_WEATHER"] = out["WEATHER_CONDITION"]

    if pristine_csv.exists():
        p = pd.read_csv(pristine_csv, nrows=1)
        pristine_cols = [str(c).strip() for c in p.columns]
        for c in pristine_cols:
            if c not in out.columns:
                out[c] = np.nan
        # Keep strict v8 column order for full compatibility.
        out = out[pristine_cols]

    return out


def _extract_first_text(v: object, default: str = "N/A") -> str:
    if isinstance(v, list):
        if not v:
            return default
        vv = v[0]
    else:
        vv = v
    if vv is None:
        return default
    if isinstance(vv, float) and np.isnan(vv):
        return default
    txt = str(vv).strip()
    return txt if txt else default


def load_openmeteo_hourly(weather_csv: Path, weather_rounding: str) -> pd.DataFrame:
    raw = pd.read_csv(weather_csv)
    # Some Open-Meteo files include metadata lines before the hourly table.
    if "time" not in raw.columns:
        raw = pd.read_csv(weather_csv, skiprows=3)

    need = [
        "time",
        "temperature_2m (\u00b0C)",
        "precipitation (mm)",
        "wind_speed_10m (km/h)",
        "weather_code (wmo code)",
    ]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError(f"weather csv missing required columns: {miss}")

    out = pd.DataFrame()
    dt = pd.to_datetime(raw["time"], errors="coerce")
    if weather_rounding == "round":
        hour_key = dt.dt.round("h")
    else:
        hour_key = dt.dt.floor("h")

    out["__HOUR_KEY"] = hour_key
    out["TEMP_C"] = pd.to_numeric(raw["temperature_2m (\u00b0C)"], errors="coerce")
    out["prcp"] = pd.to_numeric(raw["precipitation (mm)"], errors="coerce")
    out["WIND_SPEED_KMH"] = pd.to_numeric(raw["wind_speed_10m (km/h)"], errors="coerce")
    out["coco"] = pd.to_numeric(raw["weather_code (wmo code)"], errors="coerce")

    out = out.dropna(subset=["__HOUR_KEY"]).copy()
    out = out.drop_duplicates(subset=["__HOUR_KEY"], keep="first").sort_values("__HOUR_KEY")

    out["CTX_TEMP"] = out["TEMP_C"]
    out["CTX_PRCP"] = out["prcp"]
    out["CTX_WSPD"] = out["WIND_SPEED_KMH"]
    out["CTX_COCO"] = out["coco"]
    out["CTX_WEATHER_SOURCE"] = "OpenMeteo_Offline"

    LOGGER.info("weather hourly loaded: rows=%d", len(out))
    return out


def map_offline_osm(raw_df: pd.DataFrame, pbf_path: Path, signal_threshold_m: float) -> pd.DataFrame:
    req = ["LATITUDE", "LONGITUDE"]
    for c in req:
        if c not in raw_df.columns:
            raise ValueError(f"raw csv missing required column: {c}")

    out = raw_df.copy()
    out["LATITUDE"] = pd.to_numeric(out["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(out["LONGITUDE"], errors="coerce")

    valid = out["LATITUDE"].notna() & out["LONGITUDE"].notna()
    out["DIST_TO_SIGNAL_M"] = np.nan
    out["HAS_TRAFFIC_SIGNAL"] = 0
    out["OSM_TYPE"] = "N/A"
    out["OSM_SPEED_TAG"] = "N/A"
    out["OSM_ONEWAY"] = 0

    if int(valid.sum()) == 0:
        LOGGER.warning("no valid coordinates for OSM mapping")
        return out

    osm = OSM(str(pbf_path))
    net = osm.get_network(network_type="driving", nodes=True)
    if net is None:
        raise RuntimeError("no driving network extracted from PBF")
    nodes, edges = net
    if edges is None or len(edges) == 0:
        raise RuntimeError("no driving edges extracted from PBF")

    edges = edges.copy()
    if "geometry" not in edges.columns:
        raise RuntimeError("edges missing geometry")

    # Nearest edge by midpoint in metric CRS, then convert back to lat/lon.
    edges_metric = edges.to_crs("EPSG:2263")
    mids = edges_metric.geometry.centroid.to_crs("EPSG:4326")
    edge_lat = mids.y.to_numpy(dtype=float)
    edge_lon = mids.x.to_numpy(dtype=float)
    edge_latlon_rad = np.radians(np.column_stack([edge_lat, edge_lon]))
    edge_tree = BallTree(edge_latlon_rad, metric="haversine")

    q_lat = out.loc[valid, "LATITUDE"].to_numpy(dtype=float)
    q_lon = out.loc[valid, "LONGITUDE"].to_numpy(dtype=float)
    q_latlon_rad = np.radians(np.column_stack([q_lat, q_lon]))

    _, idx_edge = edge_tree.query(q_latlon_rad, k=1)
    edge_idx = idx_edge[:, 0]
    matched_edges = edges.iloc[edge_idx]

    out.loc[valid, "OSM_TYPE"] = matched_edges.get("highway", pd.Series(index=matched_edges.index)).map(lambda v: _extract_first_text(v, "N/A")).to_numpy()
    out.loc[valid, "OSM_SPEED_TAG_RAW"] = matched_edges.get("maxspeed", pd.Series(index=matched_edges.index)).map(lambda v: _extract_first_text(v, "N/A")).to_numpy()
    out.loc[valid, "OSM_MAXSPEED_FWD_RAW"] = matched_edges.get("maxspeed:forward", pd.Series(index=matched_edges.index)).map(lambda v: _extract_first_text(v, "N/A")).to_numpy()
    out.loc[valid, "OSM_MAXSPEED_BWD_RAW"] = matched_edges.get("maxspeed:backward", pd.Series(index=matched_edges.index)).map(lambda v: _extract_first_text(v, "N/A")).to_numpy()
    out.loc[valid, "OSM_SOURCE_MAXSPEED_RAW"] = matched_edges.get("source:maxspeed", pd.Series(index=matched_edges.index)).map(lambda v: _extract_first_text(v, "N/A")).to_numpy()
    out.loc[valid, "OSM_SPEED_TAG"] = out.loc[valid, ["OSM_SPEED_TAG_RAW", "OSM_MAXSPEED_FWD_RAW", "OSM_MAXSPEED_BWD_RAW", "OSM_SOURCE_MAXSPEED_RAW"]].apply(_resolve_primary_speed_tag, axis=1).to_numpy()
    out.loc[valid, "OSM_ONEWAY"] = matched_edges.get("oneway", pd.Series(index=matched_edges.index)).map(lambda v: 1 if _normalize_bool_series(pd.Series([v])).iloc[0] else 0).to_numpy()

    # Traffic signals from POI first, fallback to node highway tag.
    signal_lat = np.array([], dtype=float)
    signal_lon = np.array([], dtype=float)
    try:
        signals = osm.get_pois(custom_filter={"highway": ["traffic_signals"]})
        if signals is not None and len(signals) > 0 and "geometry" in signals.columns:
            sg = signals.geometry
            signal_lat = sg.y.to_numpy(dtype=float)
            signal_lon = sg.x.to_numpy(dtype=float)
    except Exception:
        pass

    if signal_lat.size == 0 and nodes is not None and len(nodes) > 0:
        nd = nodes.copy()
        if "highway" in nd.columns and "geometry" in nd.columns:
            mask = nd["highway"].astype(str).str.lower().eq("traffic_signals")
            if mask.any():
                ng = nd.loc[mask, "geometry"]
                signal_lat = ng.y.to_numpy(dtype=float)
                signal_lon = ng.x.to_numpy(dtype=float)

    if signal_lat.size > 0:
        sig_tree = BallTree(np.radians(np.column_stack([signal_lat, signal_lon])), metric="haversine")
        dist_rad, _ = sig_tree.query(q_latlon_rad, k=1)
        dist_m = (dist_rad[:, 0] * EARTH_RADIUS_M).astype(float)
        out.loc[valid, "DIST_TO_SIGNAL_M"] = dist_m
        out.loc[valid, "HAS_TRAFFIC_SIGNAL"] = (dist_m <= float(signal_threshold_m)).astype(int)
    else:
        LOGGER.warning("no traffic signal points extracted from PBF")

    return out


def enrich_offline_context(raw_df: pd.DataFrame, pbf_path: Path, weather_csv: Path, signal_threshold_m: float, weather_rounding: str) -> pd.DataFrame:
    out = raw_df.copy()
    out["__CRASH_DT"] = _build_crash_dt(out)
    if weather_rounding == "round":
        out["__HOUR_KEY"] = out["__CRASH_DT"].dt.round("h")
    else:
        out["__HOUR_KEY"] = out["__CRASH_DT"].dt.floor("h")

    weather = load_openmeteo_hourly(weather_csv=weather_csv, weather_rounding=weather_rounding)
    out = out.merge(weather, on="__HOUR_KEY", how="left")

    out = map_offline_osm(out, pbf_path=pbf_path, signal_threshold_m=signal_threshold_m)

    # Derived fields to align with pristine schema style.
    out["CRASH_FULL_TIME"] = out["__CRASH_DT"]
    out = _apply_speed_limit_hierarchy(out)
    out["OSM_LANES_TAG"] = "N/A"
    out["HAS_DIVIDER"] = 0
    out["INFERRED_LANES"] = np.where(out["OSM_TYPE"].astype(str).str.lower().isin(["motorway", "trunk"]), 3, 2)

    veh_cols = [c for c in ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"] if c in out.columns]
    if veh_cols:
        present = np.zeros(len(out), dtype=int)
        for c in veh_cols:
            present += (~_safe_str(out[c]).str.lower().isin(["", "nan", "none", "null", "unknown", "unspecified"]))
        out["TOTAL_VEHICLES"] = np.maximum(present, 1)
    else:
        out["TOTAL_VEHICLES"] = 1
    out["IS_MULTI_VEHICLE"] = (pd.to_numeric(out["TOTAL_VEHICLES"], errors="coerce").fillna(1) >= 2).astype(int)

    return out


def _series_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def _compare_numeric(a: pd.Series, b: pd.Series, atol: float) -> Tuple[int, int]:
    aa = _series_numeric(a)
    bb = _series_numeric(b)
    both_nan = aa.isna() & bb.isna()
    comparable = ~(both_nan)
    diff = (aa - bb).abs()
    mismatch = comparable & (~(diff <= atol))
    mismatch = mismatch.fillna(True)
    return int(mismatch.sum()), int(comparable.sum())


def _compare_bool(a: pd.Series, b: pd.Series) -> Tuple[int, int]:
    aa = _normalize_bool_series(a)
    bb = _normalize_bool_series(b)
    return int((aa != bb).sum()), len(aa)


def _compare_speed_tag(a: pd.Series, b: pd.Series, mph_tol: float = 5.0) -> Tuple[int, int]:
    am = a.map(_parse_speed_tag_to_mph)
    bm = b.map(_parse_speed_tag_to_mph)
    ams = pd.Series(am, index=a.index, dtype=float)
    bms = pd.Series(bm, index=b.index, dtype=float)

    both_num = ams.notna() & bms.notna()
    both_nan = ams.isna() & bms.isna()

    mism_num = both_num & ((ams - bms).abs() > mph_tol)
    a_txt = _norm_text(a)
    b_txt = _norm_text(b)
    mism_txt = (~both_num) & (~both_nan) & (a_txt != b_txt)

    mismatch = int(mism_num.sum() + mism_txt.sum())
    comparable = int((~both_nan).sum())
    return mismatch, comparable


def compute_discrepancy(remapped: pd.DataFrame, pristine: pd.DataFrame, join_key: str) -> Dict[str, object]:
    if join_key not in remapped.columns or join_key not in pristine.columns:
        raise ValueError(f"join_key not found in both dataframes: {join_key}")

    left = remapped.copy()
    right = pristine.copy()

    # Keep first row for duplicate join keys to make comparison deterministic.
    l_dup = int(left.duplicated(subset=[join_key]).sum())
    r_dup = int(right.duplicated(subset=[join_key]).sum())
    if l_dup > 0:
        left = left.drop_duplicates(subset=[join_key], keep="first")
    if r_dup > 0:
        right = right.drop_duplicates(subset=[join_key], keep="first")

    joined = left.merge(right, on=join_key, how="inner", suffixes=("_offline", "_v8"))

    col_specs = {
        "TEMP_C": ("num", 1.0),
        "prcp": ("num", 0.5),
        "WIND_SPEED_KMH": ("num", 2.0),
        "coco": ("num", 0.0),
        "DIST_TO_SIGNAL_M": ("num", 30.0),
        "HAS_TRAFFIC_SIGNAL": ("bool", 0.0),
        "OSM_TYPE": ("text", 0.0),
        "OSM_SPEED_TAG": ("speed", 5.0),
        "OSM_ONEWAY": ("bool", 0.0),
    }

    feature_report: Dict[str, Dict[str, float | str]] = {}
    mismatch_total = 0
    comparable_total = 0

    for col, (kind, tol) in col_specs.items():
        a_col = f"{col}_offline"
        b_col = f"{col}_v8"
        if a_col not in joined.columns or b_col not in joined.columns:
            continue

        a = joined[a_col]
        b = joined[b_col]

        if kind == "num":
            mism, comp = _compare_numeric(a, b, atol=float(tol))
        elif kind == "bool":
            mism, comp = _compare_bool(a, b)
        elif kind == "speed":
            mism, comp = _compare_speed_tag(a, b, mph_tol=float(tol))
        else:
            aa = _norm_text(a)
            bb = _norm_text(b)
            both_empty = aa.eq("") & bb.eq("")
            comp = int((~both_empty).sum())
            mism = int(((aa != bb) & (~both_empty)).sum())

        rate = float(mism / comp) if comp > 0 else 0.0
        feature_report[col] = {
            "mismatch_count": float(mism),
            "comparable_count": float(comp),
            "discrepancy_rate": rate,
            "tolerance": float(tol),
            "kind": kind,
        }
        mismatch_total += mism
        comparable_total += comp

    overall_rate = float(mismatch_total / comparable_total) if comparable_total > 0 else 1.0

    return {
        "join_key": join_key,
        "rows_offline": int(len(left)),
        "rows_pristine": int(len(right)),
        "rows_joined": int(len(joined)),
        "offline_duplicate_join_keys": l_dup,
        "pristine_duplicate_join_keys": r_dup,
        "overall_discrepancy_rate": overall_rate,
        "overall_mismatch_cells": int(mismatch_total),
        "overall_comparable_cells": int(comparable_total),
        "feature_discrepancy": feature_report,
    }


def write_v9_if_pass(remapped_df: pd.DataFrame, report: Dict[str, object], cfg: Config) -> bool:
    overall = float(cast(float, report.get("overall_discrepancy_rate", 1.0)))
    passed = overall <= float(cfg.accept_rate)

    if (not passed) and (not cfg.force_overwrite):
        LOGGER.warning(
            "discrepancy %.6f > accept_rate %.6f, skip v9 overwrite (set --force_overwrite to override)",
            overall,
            cfg.accept_rate,
        )
        return False

    v9_df = align_to_pristine_schema(remapped_df, cfg.pristine_csv)
    cfg.v9_output_csv.parent.mkdir(parents=True, exist_ok=True)
    v9_df.to_csv(cfg.v9_output_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("v9 offline pristine written: %s", cfg.v9_output_csv.as_posix())
    return True


def run(cfg: Config) -> Dict[str, object]:
    LOGGER.info("loading raw csv: %s", cfg.raw_csv.as_posix())
    raw = _load_csv_strip(cfg.raw_csv)
    if cfg.sample_n and cfg.sample_n > 0:
        raw = raw.head(int(cfg.sample_n)).copy()
        LOGGER.info("sample mode enabled: sample_n=%d", int(cfg.sample_n))

    LOGGER.info("re-mapping offline context from pbf+weather")
    remapped = enrich_offline_context(
        raw_df=raw,
        pbf_path=cfg.pbf_path,
        weather_csv=cfg.weather_csv,
        signal_threshold_m=cfg.signal_threshold_m,
        weather_rounding=cfg.weather_rounding,
    )

    cfg.remapped_csv.parent.mkdir(parents=True, exist_ok=True)
    remapped.to_csv(cfg.remapped_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("saved remapped offline csv: %s", cfg.remapped_csv.as_posix())

    LOGGER.info("loading pristine v8 csv: %s", cfg.pristine_csv.as_posix())
    pristine = _load_csv_strip(cfg.pristine_csv)

    report = compute_discrepancy(remapped=remapped, pristine=pristine, join_key=cfg.join_key)
    report["accept_rate_threshold"] = float(cfg.accept_rate)
    overall_rate = float(cast(float, report.get("overall_discrepancy_rate", 1.0)))
    report["passed"] = bool(overall_rate <= float(cfg.accept_rate))

    wrote_v9 = write_v9_if_pass(remapped_df=remapped, report=report, cfg=cfg)
    report["v9_written"] = wrote_v9
    report["v9_output_csv"] = cfg.v9_output_csv.as_posix()
    report["remapped_csv"] = cfg.remapped_csv.as_posix()

    cfg.report_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("overall discrepancy rate: %.6f", overall_rate)
    LOGGER.info("report saved: %s", cfg.report_json.as_posix())

    return report


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    project = _project_root()

    parser = argparse.ArgumentParser(description="Phase 0 offline consistency checker")
    parser.add_argument(
        "--raw_csv",
        type=str,
        default=str(project / "原始数据集" / "nyc_accidents_2017.csv"),
    )
    parser.add_argument(
        "--pbf_path",
        type=str,
        default=str(repo / "osmdata" / "new-york-180101-internal.osm.pbf"),
    )
    parser.add_argument(
        "--weather_csv",
        type=str,
        default=str(repo / "weather" / "open-meteo-40.74N74.04W51m.csv"),
    )
    parser.add_argument(
        "--pristine_csv",
        type=str,
        default=str(repo / "nyc_2017_pristine_v8.csv"),
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=str(repo / "exp" / "phase0" / "offline_consistency_report.json"),
    )
    parser.add_argument(
        "--remapped_csv",
        type=str,
        default=str(repo / "exp" / "phase0" / "offline_remapped_2017.csv"),
    )
    parser.add_argument(
        "--v9_output_csv",
        type=str,
        default=str(repo / "v9_offline_pristine.csv"),
    )
    parser.add_argument("--join_key", type=str, default="COLLISION_ID")
    parser.add_argument("--signal_threshold_m", type=float, default=20.0)
    parser.add_argument("--weather_rounding", type=str, choices=["floor", "round"], default="floor")
    parser.add_argument("--accept_rate", type=float, default=0.35)
    parser.add_argument("--force_overwrite", action="store_true")
    parser.add_argument("--sample_n", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    cfg = Config(
        raw_csv=Path(args.raw_csv),
        pbf_path=Path(args.pbf_path),
        weather_csv=Path(args.weather_csv),
        pristine_csv=Path(args.pristine_csv),
        report_json=Path(args.report_json),
        remapped_csv=Path(args.remapped_csv),
        v9_output_csv=Path(args.v9_output_csv),
        join_key=args.join_key,
        signal_threshold_m=float(args.signal_threshold_m),
        weather_rounding=args.weather_rounding,
        accept_rate=float(args.accept_rate),
        force_overwrite=bool(args.force_overwrite),
        sample_n=int(args.sample_n),
    )

    run(cfg)


if __name__ == "__main__":
    main()
