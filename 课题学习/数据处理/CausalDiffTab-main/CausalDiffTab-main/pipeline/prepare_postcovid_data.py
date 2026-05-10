from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data"


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _time_period(hour: int) -> str:
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _contains_any(text: str, keys: List[str]) -> bool:
    t = str(text).lower()
    return any(k in t for k in keys)


def _vehicle_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = [
        "VEHICLE TYPE CODE 1",
        "VEHICLE TYPE CODE 2",
        "VEHICLE TYPE CODE 3",
        "VEHICLE TYPE CODE 4",
        "VEHICLE TYPE CODE 5",
    ]
    merged = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    return {
        "is_sedan": merged.str.contains("sedan|4 dr", regex=True),
        "is_suv": merged.str.contains("suv|sport utility", regex=True),
        "is_taxi": merged.str.contains("taxi|cab", regex=True),
        "is_truck": merged.str.contains("truck|tractor", regex=True),
        "is_pickup": merged.str.contains("pickup", regex=True),
        "is_bus": merged.str.contains("bus", regex=True),
        "is_van": merged.str.contains("van", regex=True),
        "is_motorcycle": merged.str.contains("motorcycle|scooter", regex=True),
        "is_bicycle": merged.str.contains("bicycle|bike", regex=True),
        "is_emergency": merged.str.contains("ambulance|fire|police", regex=True),
    }


def _factor_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = [
        "CONTRIBUTING FACTOR VEHICLE 1",
        "CONTRIBUTING FACTOR VEHICLE 2",
        "CONTRIBUTING FACTOR VEHICLE 3",
        "CONTRIBUTING FACTOR VEHICLE 4",
        "CONTRIBUTING FACTOR VEHICLE 5",
    ]
    merged = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    return {
        "is_distracted": merged.str.contains("distraction|inattention", regex=True),
        "is_speeding": merged.str.contains("speed", regex=True),
        "is_failure_to_yield": merged.str.contains("yield|right.of.way", regex=True),
        "is_following_too_closely": merged.str.contains("following too closely", regex=True),
        "is_drunk_driving": merged.str.contains("alcohol|drugs|intoxicated", regex=True),
        "is_fatigue": merged.str.contains("fatigue|fell asleep", regex=True),
        "is_view_obstructed": merged.str.contains("view obstructed|visibility", regex=True),
        "is_vehicle_defect": merged.str.contains("defective|brake|tire|steering", regex=True),
        "is_backing_unsafely": merged.str.contains("backing unsafely", regex=True),
        "is_pedestrian_related": merged.str.contains("pedestrian", regex=True),
        "is_inexperience": merged.str.contains("inexperience", regex=True),
        "is_pavement_slippery": merged.str.contains("pavement slippery", regex=True),
    }


def _make_output(
    raw_df: pd.DataFrame,
    train_ref: pd.DataFrame,
    n_sample: int,
    seed: int,
    graphml_path: Optional[Path] = None,
    signals_path: Optional[Path] = None,
    weather_paths: Optional[List[Path]] = None,
) -> pd.DataFrame:
    if len(raw_df) > n_sample:
        raw_df = raw_df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    else:
        raw_df = raw_df.reset_index(drop=True)

    crash_dt = pd.to_datetime(raw_df["CRASH DATE"], errors="coerce")
    crash_tm = pd.to_datetime(raw_df["CRASH TIME"], format="%H:%M", errors="coerce")
    hour = crash_tm.dt.hour.fillna(12).astype(int)

    out = pd.DataFrame(index=raw_df.index)
    out["LATITUDE"] = pd.to_numeric(raw_df["LATITUDE"], errors="coerce")
    out["LONGITUDE"] = pd.to_numeric(raw_df["LONGITUDE"], errors="coerce")
    ang = 2.0 * np.pi * (hour.astype(float) / 24.0)
    out["CRASH_TIME_SIN"] = np.sin(ang)
    out["CRASH_TIME_COS"] = np.cos(ang)
    out["SEASON"] = crash_dt.dt.month.fillna(1).astype(int).map(_season_from_month)
    out["DAY_OF_WEEK"] = crash_dt.dt.dayofweek.fillna(0).astype(int)
    out["TIME_PERIOD"] = hour.map(_time_period)

    veh_flags = _vehicle_flags(raw_df)
    fac_flags = _factor_flags(raw_df)
    for k, v in {**veh_flags, **fac_flags}.items():
        out[k] = v.astype(int)

    out["NUMBER_OF_PEDESTRIANS_INJURED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF PEDESTRIANS INJURED"], errors="coerce").fillna(0) > 0
    ).astype(int)
    out["NUMBER_OF_PEDESTRIANS_KILLED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF PEDESTRIANS KILLED"], errors="coerce").fillna(0) > 0
    ).astype(int)
    out["NUMBER_OF_CYCLIST_INJURED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF CYCLIST INJURED"], errors="coerce").fillna(0) > 0
    ).astype(int)
    out["NUMBER_OF_CYCLIST_KILLED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF CYCLIST KILLED"], errors="coerce").fillna(0) > 0
    ).astype(int)
    out["NUMBER_OF_MOTORIST_INJURED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF MOTORIST INJURED"], errors="coerce").fillna(0) > 0
    ).astype(int)
    out["NUMBER_OF_MOTORIST_KILLED_BIN"] = (
        pd.to_numeric(raw_df["NUMBER OF MOTORIST KILLED"], errors="coerce").fillna(0) > 0
    ).astype(int)

    vcols = [
        "VEHICLE TYPE CODE 1",
        "VEHICLE TYPE CODE 2",
        "VEHICLE TYPE CODE 3",
        "VEHICLE TYPE CODE 4",
        "VEHICLE TYPE CODE 5",
    ]
    total_veh = raw_df[vcols].notna().sum(axis=1).clip(lower=1, upper=5)
    out["TOTAL_VEHICLES"] = total_veh.astype(int)
    out["IS_MULTI_VEHICLE"] = (total_veh > 1).astype(int)

    out["NUMBER OF PERSONS INJURED"] = pd.to_numeric(
        raw_df["NUMBER OF PERSONS INJURED"], errors="coerce"
    ).fillna(0.0)

    # Level-B：优先使用本地离线文件做真实空间/时间匹配（修复静态问题）
    # 若无本地文件则 fallback 回 train_ref median/mode（兼容旧行为）
    _LEVEL_B = [
        "TEMP_C", "prcp", "WIND_SPEED_KMH",
        "DIST_TO_SIGNAL_M", "INFERRED_LANES",
        "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY",
        "coco", "WEATHER_CONDITION", "OSM_TYPE",
    ]

    osm_done     = False
    weather_done = False

    # OSM 真实空间匹配
    if graphml_path is not None and graphml_path.exists():
        try:
            from pipeline.prepare_2025_data import enrich_osm
            out = enrich_osm(out, graphml_path, signals_path)
            osm_done = True
        except Exception as e:
            print(f"  [OSM] ⚠ 路网匹配失败（{e}），回退 median")

    # 天气真实时间匹配
    if weather_paths:
        valid_wpaths = [p for p in weather_paths if p.exists()]
        if valid_wpaths:
            try:
                # 临时加回日期时间列，供 enrich_weather 使用
                out["CRASH DATE"] = raw_df["CRASH DATE"].values
                out["CRASH TIME"] = raw_df["CRASH TIME"].values
                from pipeline.prepare_2025_data import enrich_weather
                out = enrich_weather(out, valid_wpaths)
                out.drop(columns=["CRASH DATE", "CRASH TIME"], inplace=True, errors="ignore")
                weather_done = True
            except Exception as e:
                out.drop(columns=["CRASH DATE", "CRASH TIME"], inplace=True, errors="ignore")
                print(f"  [Weather] ⚠ 天气匹配失败（{e}），回退 median")

    # Fallback：仍缺失的 Level-B 列 → train_ref median/mode
    for c in _LEVEL_B:
        if c in out.columns:
            continue
        if c not in train_ref.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_ref[c]):
            out[c] = pd.to_numeric(train_ref[c], errors="coerce").median()
        else:
            mode = train_ref[c].mode(dropna=True)
            out[c] = mode.iloc[0] if len(mode) else train_ref[c].dropna().iloc[0]

    if not osm_done:
        print("  ⚠ OSM 使用静态 median（无本地 graphml）")
    if not weather_done:
        print("  ⚠ 天气使用静态 median（无本地 weather CSV）")

    # Keep the exact schema used by the evaluator.
    keep_cols = [c for c in train_ref.columns if c in out.columns]
    out = out.reindex(columns=keep_cols)
    for c in train_ref.columns:
        if c not in out.columns:
            out[c] = train_ref[c].iloc[0]
    out = out[train_ref.columns]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare post-COVID (2022/2023) transfer test data")
    parser.add_argument(
        "--raw_csv",
        default=str(RAW_DIR / "crash" / "Motor_Vehicle_Collisions_-_Crashes_20250929.csv"),
    )
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--n_sample", type=int, default=5000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument(
        "--osm_graphml",
        default=str(RAW_DIR / "osm" / "nyc_drive_graph.graphml"),
        help="本地路网 GraphML（可选，有则做真实空间匹配）",
    )
    parser.add_argument(
        "--osm_signals",
        default=str(RAW_DIR / "osm" / "nyc_traffic_signals.geojson"),
        help="信号灯 GeoJSON（可选）",
    )
    parser.add_argument(
        "--weather_csvs", nargs="*",
        default=[
            str(RAW_DIR / "weather" / "72503.csv.gz"),
            str(RAW_DIR / "weather" / "74486.csv.gz"),
        ],
        help="Meteostat bulk CSV（支持 .csv.gz）",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    if not raw_path.is_file():
        raise SystemExit(f"Raw CSV not found: {raw_path}")

    train_ref_path = ROOT / "data" / "nyc_crash" / "train.csv"
    if not train_ref_path.is_file():
        raise SystemExit(f"Reference train not found: {train_ref_path}")

    raw = pd.read_csv(raw_path, low_memory=False)
    raw["CRASH DATE"] = pd.to_datetime(raw["CRASH DATE"], errors="coerce")
    train_ref = pd.read_csv(train_ref_path)

    graphml_path  = Path(args.osm_graphml)
    signals_path  = Path(args.osm_signals)
    weather_paths = [Path(p) for p in (args.weather_csvs or [])]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for y in args.years:
        dfx = raw[raw["CRASH DATE"].dt.year == int(y)].copy()
        dfx = dfx[
            dfx["LATITUDE"].between(40.4, 40.95, inclusive="both")
            & dfx["LONGITUDE"].between(-74.3, -73.7, inclusive="both")
        ]
        out_df = _make_output(
            dfx, train_ref,
            n_sample=int(args.n_sample), seed=int(args.seed),
            graphml_path=graphml_path  if graphml_path.exists()  else None,
            signals_path=signals_path  if signals_path.exists()  else None,
            weather_paths=weather_paths,
        )

        latest = out_dir / f"postcovid_test_{y}_n{args.n_sample}.csv"
        stamped = out_dir / f"postcovid_test_{y}_n{args.n_sample}_{ts}.csv"
        out_df.to_csv(latest, index=False)
        out_df.to_csv(stamped, index=False)
        outputs.append({"year": int(y), "n_rows": int(len(out_df)), "latest": str(latest), "stamped": str(stamped)})

    report = {
        "generated_at": ts,
        "years": [int(y) for y in args.years],
        "n_sample_per_year": int(args.n_sample),
        "seed": int(args.seed),
        "output_files": outputs,
        "fill_strategy": {
            "level_a": "derived from raw columns",
            "level_b_osm":     "real_spatial"  if graphml_path.exists() else "static_median",
            "level_b_weather": "real_temporal" if any(p.exists() for p in weather_paths) else "static_median",
        },
    }
    report_json_latest = out_dir / "postcovid_prep_report_latest.json"
    report_json_stamp = out_dir / f"postcovid_prep_report_{ts}.json"
    report_md_latest = out_dir / "postcovid_prep_report_latest.md"
    report_md_stamp = out_dir / f"postcovid_prep_report_{ts}.md"

    report_json_latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_json_stamp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Post-COVID Migration Test - Data Preparation Report",
        "",
        f"- generated_at: `{ts}`",
        f"- target_years: `{args.years}`",
        f"- n_sample_per_year: `{args.n_sample}`",
        "",
        "## Output Files",
        "",
        "| year | n_rows | latest | stamped |",
        "| --- | --- | --- | --- |",
    ]
    for item in outputs:
        md.append(
            f"| {item['year']} | {item['n_rows']} | `{item['latest']}` | `{item['stamped']}` |"
        )
    md.append("")
    md.append("## Notes")
    md.append("- Weather/road context columns are filled from train-set medians/modes to keep schema compatibility.")
    md.append("- This setup focuses transfer shift on temporal + behavior patterns from post-COVID years.")

    md_text = "\n".join(md)
    report_md_latest.write_text(md_text, encoding="utf-8")
    report_md_stamp.write_text(md_text, encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
