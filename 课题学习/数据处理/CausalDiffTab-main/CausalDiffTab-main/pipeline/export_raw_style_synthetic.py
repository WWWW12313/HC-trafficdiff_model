"""Export synthetic crash samples in raw NYC crash + API supplement style.

The diffusion/evaluation CSVs intentionally keep engineered training columns
such as ``is_sedan`` and ``CRASH_TIME_SIN``.  This script creates a delivery
table that looks like the pre-training table: original NYC crash columns plus
API/context enrichment columns, with engineered ``is_*`` columns restored into
vehicle and contributing-factor text fields.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_CRASH_COLUMNS = [
    "CRASH DATE",
    "CRASH TIME",
    "BOROUGH",
    "ZIP CODE",
    "LATITUDE",
    "LONGITUDE",
    "LOCATION",
    "ON STREET NAME",
    "CROSS STREET NAME",
    "OFF STREET NAME",
    "NUMBER OF PERSONS INJURED",
    "NUMBER OF PERSONS KILLED",
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
    "CONTRIBUTING FACTOR VEHICLE 1",
    "CONTRIBUTING FACTOR VEHICLE 2",
    "CONTRIBUTING FACTOR VEHICLE 3",
    "CONTRIBUTING FACTOR VEHICLE 4",
    "CONTRIBUTING FACTOR VEHICLE 5",
    "COLLISION_ID",
    "VEHICLE TYPE CODE 1",
    "VEHICLE TYPE CODE 2",
    "VEHICLE TYPE CODE 3",
    "VEHICLE TYPE CODE 4",
    "VEHICLE TYPE CODE 5",
]

API_SUPPLEMENT_COLUMNS = [
    "CRASH_FULL_TIME",
    "TEMP_C",
    "prcp",
    "WIND_SPEED_KMH",
    "coco",
    "WEATHER_CONDITION",
    "REAL_WEATHER",
    "DIST_TO_SIGNAL_M",
    "HAS_TRAFFIC_SIGNAL",
    "OSM_TYPE",
    "OSM_SPEED_TAG",
    "OSM_LANES_TAG",
    "OSM_ONEWAY",
    "REAL_SPEED_LIMIT",
    "HAS_DIVIDER",
    "INFERRED_LANES",
    "TOTAL_VEHICLES",
    "IS_MULTI_VEHICLE",
]

VEHICLE_FLAG_TO_LABEL = [
    ("is_bus", "Bus"),
    ("is_truck", "Box Truck"),
    ("is_taxi", "Taxi"),
    ("is_motorcycle", "Motorcycle"),
    ("is_bicycle", "Bicycle"),
    ("is_suv", "Station Wagon/Sport Utility Vehicle"),
    ("is_sedan", "Sedan"),
    ("is_other_vehicle", "Other"),
]

FACTOR_FLAG_TO_LABEL = [
    ("is_distracted", "Driver Inattention/Distraction"),
    ("is_speeding", "Unsafe Speed"),
    ("is_failure_to_yield", "Failure to Yield Right-of-Way"),
    ("is_following_too_closely", "Following Too Closely"),
    ("is_drunk_driving", "Alcohol Involvement"),
    ("is_fatigue", "Fell Asleep"),
    ("is_view_obstructed", "View Obstructed/Limited"),
    ("is_vehicle_defect", "Other Vehicular"),
    ("is_backing_unsafely", "Backing Unsafely"),
    ("is_pedestrian_related", "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion"),
    ("is_inexperience", "Driver Inexperience"),
    ("is_pavement_slippery", "Pavement Slippery"),
]


def _latest_raw_crash_csv() -> Path:
    candidates = sorted((ROOT / "raw_data" / "crash").glob("Motor_Vehicle_Collisions_-_Crashes_*.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError("raw_data/crash 下没有 Motor_Vehicle_Collisions 原始 CSV")
    return candidates[0]


def _truthy(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0) > 0.5


def _pick_dates(source: pd.DataFrame, raw_csv: Path, target_year: int, seed: int) -> pd.Series:
    usecols = ["CRASH DATE"]
    raw_dates = pd.read_csv(raw_csv, usecols=usecols, low_memory=False)
    dt = pd.to_datetime(raw_dates["CRASH DATE"], errors="coerce").dropna()
    dt = dt[dt.dt.year == int(target_year)]
    if len(dt) == 0:
        base = pd.Timestamp(f"{int(target_year)}-01-01")
        return pd.Series(
            [(base + pd.Timedelta(days=i % 366)).strftime("%m/%d/%Y") for i in range(len(source))],
            index=source.index,
        )

    months = dt.dt.month
    season = pd.Series(np.where(months.isin([12, 1, 2]), "winter", ""), index=dt.index)
    season = season.mask(months.isin([3, 4, 5]), "spring")
    season = season.mask(months.isin([6, 7, 8]), "summer")
    season = season.mask(months.isin([9, 10, 11]), "autumn")
    dow = dt.dt.dayofweek.astype(int)

    by_key: dict[tuple[str, int], list[pd.Timestamp]] = {}
    by_season: dict[str, list[pd.Timestamp]] = {}
    for date_value, season_value, dow_value in zip(dt.tolist(), season.tolist(), dow.tolist()):
        by_key.setdefault((str(season_value), int(dow_value)), []).append(date_value)
        by_season.setdefault(str(season_value), []).append(date_value)

    rng = np.random.RandomState(seed)
    src_season = source["SEASON"].astype(str).str.lower() if "SEASON" in source.columns else pd.Series("", index=source.index)
    src_dow = pd.to_numeric(source["DAY_OF_WEEK"], errors="coerce").fillna(0).astype(int) if "DAY_OF_WEEK" in source.columns else pd.Series(0, index=source.index)
    all_dates = dt.tolist()
    picked = []
    for season_value, dow_value in zip(src_season.tolist(), src_dow.tolist()):
        candidates = by_key.get((str(season_value), int(dow_value))) or by_season.get(str(season_value)) or all_dates
        picked.append(candidates[int(rng.randint(0, len(candidates)))])
    return pd.Series(pd.to_datetime(picked).strftime("%m/%d/%Y"), index=source.index)


def _restore_vehicle_codes(source: pd.DataFrame, out: pd.DataFrame) -> None:
    labels_by_row: list[list[str]] = []
    for _, row in source.iterrows():
        labels = []
        for flag_col, label in VEHICLE_FLAG_TO_LABEL:
            if flag_col in source.columns and pd.to_numeric(row.get(flag_col), errors="coerce") > 0.5:
                labels.append(label)
        if not labels:
            labels = ["Sedan"]
        labels_by_row.append(labels[:5])

    for i in range(1, 6):
        col = f"VEHICLE TYPE CODE {i}"
        out[col] = [labels[i - 1] if len(labels) >= i else "" for labels in labels_by_row]


def _restore_factor_codes(source: pd.DataFrame, out: pd.DataFrame) -> None:
    labels_by_row: list[list[str]] = []
    for _, row in source.iterrows():
        labels = []
        for flag_col, label in FACTOR_FLAG_TO_LABEL:
            if flag_col in source.columns and pd.to_numeric(row.get(flag_col), errors="coerce") > 0.5:
                labels.append(label)
        if not labels:
            labels = ["Unspecified"]
        labels_by_row.append(labels[:5])

    for i in range(1, 6):
        col = f"CONTRIBUTING FACTOR VEHICLE {i}"
        out[col] = [labels[i - 1] if len(labels) >= i else "" for labels in labels_by_row]


def _restore_injury_columns(source: pd.DataFrame, out: pd.DataFrame) -> None:
    persons_injured = pd.to_numeric(source.get("NUMBER OF PERSONS INJURED", 0), errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    ped_injured = pd.to_numeric(source.get("NUMBER_OF_PEDESTRIANS_INJURED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    cyclist_injured = pd.to_numeric(source.get("NUMBER_OF_CYCLIST_INJURED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    motorist_injured = (persons_injured - ped_injured - cyclist_injured).clip(lower=0)
    motorist_bin = pd.to_numeric(source.get("NUMBER_OF_MOTORIST_INJURED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    motorist_injured = motorist_injured.where(motorist_injured > 0, motorist_bin)

    ped_killed = pd.to_numeric(source.get("NUMBER_OF_PEDESTRIANS_KILLED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    cyclist_killed = pd.to_numeric(source.get("NUMBER_OF_CYCLIST_KILLED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)
    motorist_killed = pd.to_numeric(source.get("NUMBER_OF_MOTORIST_KILLED_BIN", 0), errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    out["NUMBER OF PERSONS INJURED"] = persons_injured
    out["NUMBER OF PEDESTRIANS INJURED"] = ped_injured
    out["NUMBER OF CYCLIST INJURED"] = cyclist_injured
    out["NUMBER OF MOTORIST INJURED"] = motorist_injured.astype(int)
    out["NUMBER OF PEDESTRIANS KILLED"] = ped_killed
    out["NUMBER OF CYCLIST KILLED"] = cyclist_killed
    out["NUMBER OF MOTORIST KILLED"] = motorist_killed
    out["NUMBER OF PERSONS KILLED"] = (ped_killed + cyclist_killed + motorist_killed).astype(int)


def _restore_real_speed_limit(source: pd.DataFrame) -> pd.Series:
    speed = pd.to_numeric(source.get("REAL_SPEED_LIMIT", np.nan), errors="coerce")
    if speed.notna().any() and speed.dropna().median() >= 5:
        return speed.round(1)

    scaler_path = ROOT / "data" / "processed" / "continuous_scaler.pkl"
    if scaler_path.is_file() and speed.notna().any():
        try:
            with open(scaler_path, "rb") as f:
                scaler_data = pickle.load(f)
            columns = list(scaler_data["columns"])
            col_idx = columns.index("REAL_SPEED_LIMIT")
            values = np.zeros((len(source), len(columns)), dtype=float)
            values[:, col_idx] = speed.fillna(float(speed.dropna().median())).to_numpy(dtype=float)
            restored = scaler_data["scaler"].inverse_transform(pd.DataFrame(values, columns=columns))[:, col_idx]
            return pd.Series(restored, index=source.index).clip(lower=0).round(1)
        except Exception:
            pass

    return pd.Series(25.0, index=source.index)


def _fill_location_text_fields(source: pd.DataFrame, out: pd.DataFrame, raw_csv: Path, target_year: int) -> None:
    usecols = [
        "CRASH DATE",
        "LATITUDE",
        "LONGITUDE",
        "BOROUGH",
        "ZIP CODE",
        "ON STREET NAME",
        "CROSS STREET NAME",
        "OFF STREET NAME",
    ]
    try:
        raw = pd.read_csv(raw_csv, usecols=usecols, low_memory=False)
    except Exception:
        return

    raw_dt = pd.to_datetime(raw["CRASH DATE"], errors="coerce")
    raw_lat = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    raw_lon = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    mask = (
        (raw_dt.dt.year == int(target_year))
        & raw_lat.between(40.45, 41.15)
        & raw_lon.between(-74.30, -73.65)
    )
    raw = raw.loc[mask].copy()
    if raw.empty:
        return

    try:
        from sklearn.neighbors import BallTree
    except Exception:
        return

    raw_lat = pd.to_numeric(raw["LATITUDE"], errors="coerce")
    raw_lon = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    source_lat = pd.to_numeric(source.get("LATITUDE", np.nan), errors="coerce")
    source_lon = pd.to_numeric(source.get("LONGITUDE", np.nan), errors="coerce")
    source_mask = source_lat.notna() & source_lon.notna()
    if not source_mask.any():
        return

    raw_coords = np.radians(np.column_stack([raw_lat.to_numpy(dtype=float), raw_lon.to_numpy(dtype=float)]))
    source_coords = np.radians(np.column_stack([
        source_lat.loc[source_mask].to_numpy(dtype=float),
        source_lon.loc[source_mask].to_numpy(dtype=float),
    ]))
    tree = BallTree(raw_coords, metric="haversine")
    _, indices = tree.query(source_coords, k=1)
    nearest = raw.iloc[indices[:, 0]].reset_index(drop=True)
    target_index = out.index[source_mask]
    for col in ["BOROUGH", "ZIP CODE", "ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME"]:
        out[col] = out[col].astype(object)
        out.loc[target_index, col] = nearest[col].fillna("").astype(str).to_numpy()


def export_raw_style(input_csv: Path, output_csv: Path, target_year: int, raw_csv: Path | None, seed: int) -> pd.DataFrame:
    source = pd.read_csv(input_csv, low_memory=False)
    raw_csv = raw_csv or _latest_raw_crash_csv()

    out = pd.DataFrame(index=source.index)
    for col in RAW_CRASH_COLUMNS + API_SUPPLEMENT_COLUMNS:
        out[col] = source[col] if col in source.columns else ""

    out["CRASH DATE"] = _pick_dates(source, raw_csv, target_year, seed)
    if "CRASH TIME" not in source.columns or source["CRASH TIME"].isna().all():
        out["CRASH TIME"] = "00:00"
    else:
        out["CRASH TIME"] = source["CRASH TIME"].astype(str)

    if "LATITUDE" in source.columns and "LONGITUDE" in source.columns:
        lat = pd.to_numeric(source["LATITUDE"], errors="coerce")
        lon = pd.to_numeric(source["LONGITUDE"], errors="coerce")
        out["LOCATION"] = [f"({a:.6f}, {b:.6f})" if pd.notna(a) and pd.notna(b) else "" for a, b in zip(lat, lon)]

    _fill_location_text_fields(source, out, raw_csv, target_year)

    crash_dt = pd.to_datetime(out["CRASH DATE"].astype(str) + " " + out["CRASH TIME"].astype(str), errors="coerce")
    out["CRASH_FULL_TIME"] = crash_dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    _restore_injury_columns(source, out)
    _restore_factor_codes(source, out)
    _restore_vehicle_codes(source, out)

    out["REAL_SPEED_LIMIT"] = _restore_real_speed_limit(source)
    if "WEATHER_CONDITION" in out.columns:
        out["REAL_WEATHER"] = out["WEATHER_CONDITION"]
    if "OSM_ONEWAY" in out.columns:
        oneway = pd.to_numeric(out["OSM_ONEWAY"], errors="coerce").fillna(0) > 0.5
        out["OSM_ONEWAY"] = oneway.map({True: True, False: False})
    if "TOTAL_VEHICLES" in source.columns:
        total_vehicles = pd.to_numeric(source["TOTAL_VEHICLES"], errors="coerce").fillna(1).round().astype(int).clip(1, 5)
        out["TOTAL_VEHICLES"] = total_vehicles
        out["IS_MULTI_VEHICLE"] = (total_vehicles > 1).astype(int)

    start_id = int(target_year) * 10_000_000
    out["COLLISION_ID"] = np.arange(start_id, start_id + len(out), dtype=np.int64)

    final_cols = RAW_CRASH_COLUMNS + [c for c in API_SUPPLEMENT_COLUMNS if c in out.columns]
    out = out[final_cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export synthetic data as raw crash + API supplement table")
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--target_year", type=int, default=2024)
    parser.add_argument("--raw_csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = export_raw_style(args.input_csv, args.output_csv, args.target_year, args.raw_csv, args.seed)
    print(f"[write] {args.output_csv} ({len(out)} rows, {out.shape[1]} cols)")
    print(f"[check] CRASH DATE years: {sorted(pd.to_datetime(out['CRASH DATE'], errors='coerce').dt.year.dropna().astype(int).unique().tolist())}")
    hidden = [c for c in out.columns if c.startswith("is_") or c in {"CRASH_TIME_SIN", "CRASH_TIME_COS", "SEASON", "DAY_OF_WEEK", "TIME_PERIOD"}]
    print(f"[check] engineered columns in export: {hidden}")


if __name__ == "__main__":
    main()