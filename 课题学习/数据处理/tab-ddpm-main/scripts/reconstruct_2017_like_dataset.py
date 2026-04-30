"""
Reconstruct a source-like 2017 crash dataset from v3 synthetic output.

Goals:
1. Generate CRASH DATE / CRASH TIME from CRASH_SEASON + DAY_OF_WEEK + CRASH_TIME_PERIOD.
2. Backfill CONTRIBUTING FACTOR VEHICLE 1-5 from CAUSE_* and IS_* signals.
3. Keep IS_* columns at the end for consistency checks.

Input:
  exp/nyc_crash_v3/causal_m4_v6_catY_full/synthetic_complete.csv

Output:
  synthetic_2017_like.csv
  synthetic_2017_like_report.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


CAUSE_MAP = {
    "CAUSE_001": "Driver Inattention/Distraction",
    "CAUSE_002": "Following Too Closely",
    "CAUSE_003": "Failure to Yield Right-of-Way",
    "CAUSE_004": "Passing or Lane Usage Improper",
    "CAUSE_005": "Unspecified",
}


IS_TO_FACTOR = {
    "IS_ALCOHOL_INVOLVED": "Alcohol Involvement",
    "IS_SPEEDING": "Unsafe Speed",
    "IS_DISTRACTED": "Driver Inattention/Distraction",
    "IS_FOLLOWING_TOO_CLOSE": "Following Too Closely",
    "IS_FAILURE_TO_YIELD": "Failure to Yield Right-of-Way",
    "IS_IMPROPER_LANE_USE": "Passing or Lane Usage Improper",
    "IS_BACKING_UNSAFE": "Backing Unsafely",
    "IS_IMPROPER_TURNING": "Turning Improperly",
    "IS_TRAFFIC_SIGNAL_VIOLATION": "Traffic Control Disregarded",
    "IS_INEXPERIENCED_DRIVER": "Driver Inexperience",
    "IS_FATIGUED": "Fatigued/Drowsy",
    "IS_POOR_ROAD_CONDITION": "Pavement Slippery",
    "IS_VISION_OBSCURED": "View Obstructed/Limited",
    "IS_PEDESTRIAN_CYCLIST_ERROR": "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion",
    "IS_AGGRESSIVE_DRIVING": "Aggressive Driving/Road Rage",
    "IS_VEHICLE_DEFECT": "Brakes Defective",
    "IS_OVERSIZED_VEHICLE": "Oversized Vehicle",
    "IS_OTHER_VEHICULAR": "Other Vehicular",
    "IS_ANIMAL_RELATED": "Animals Action",
    "IS_DRIVERLESS": "Driverless/Runaway Vehicle",
    "IS_UNSPECIFIED": "Unspecified",
    "IS_NONE_INVOLVED": "None",
}


def month_to_season(month: int) -> int:
    # Keep the same logic used in prepare_data_v3.py: ((month % 12) // 3)
    return (month % 12) // 3


def build_date_pool_2017() -> dict[tuple[int, int], list[dt.date]]:
    start = dt.date(2017, 1, 1)
    end = dt.date(2017, 12, 31)
    one_day = dt.timedelta(days=1)

    pool: dict[tuple[int, int], list[dt.date]] = {}
    cur = start
    while cur <= end:
        key = (month_to_season(cur.month), cur.weekday())
        pool.setdefault(key, []).append(cur)
        cur += one_day
    return pool


def sample_time_from_period(period: int, rng: np.random.Generator) -> tuple[int, int]:
    if period == 1:
        hour = int(rng.integers(7, 10))
    elif period == 2:
        hour = int(rng.integers(10, 16))
    elif period == 3:
        hour = int(rng.integers(16, 20))
    else:
        # Night: 0-6 and 20-23
        if rng.random() < 0.6:
            hour = int(rng.integers(0, 7))
        else:
            hour = int(rng.integers(20, 24))
    minute = int(rng.integers(0, 60))
    return hour, minute


def to_int_safe(v, default=0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def get_primary_cause(row: pd.Series) -> str:
    for c in ["CAUSE_001", "CAUSE_002", "CAUSE_003", "CAUSE_004", "CAUSE_005"]:
        if c in row and to_int_safe(row[c], 0) == 1:
            return CAUSE_MAP[c]
    return "Unspecified"


def get_is_causes(row: pd.Series) -> list[str]:
    factors: list[str] = []
    for col, label in IS_TO_FACTOR.items():
        if col in row and to_int_safe(row[col], 0) == 1:
            factors.append(label)
    return factors


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def reconstruct(input_csv: Path, output_csv: Path, report_txt: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(input_csv)

    pool = build_date_pool_2017()

    crash_dates = []
    crash_times = []
    full_times = []

    factor_cols = {f"CONTRIBUTING FACTOR VEHICLE {i}": [] for i in range(1, 6)}

    for _, row in df.iterrows():
        season = to_int_safe(row.get("CRASH_SEASON", 0), 0)
        dow = to_int_safe(row.get("DAY_OF_WEEK", 0), 0)
        period = to_int_safe(row.get("CRASH_TIME_PERIOD", 0), 0)

        candidates = pool.get((season, dow), [])
        if not candidates:
            # fallback: any date in the same season
            candidates = [
                d for (s, _), arr in pool.items() if s == season for d in arr
            ]
        chosen_date = candidates[int(rng.integers(0, len(candidates)))]

        hh, mm = sample_time_from_period(period, rng)
        chosen_dt = dt.datetime(chosen_date.year, chosen_date.month, chosen_date.day, hh, mm)

        crash_dates.append(chosen_dt.strftime("%m/%d/%Y"))
        crash_times.append(chosen_dt.strftime("%H:%M"))
        full_times.append(chosen_dt.strftime("%Y-%m-%d %H:%M:%S"))

        primary = get_primary_cause(row)
        from_is = get_is_causes(row)
        all_factors = unique_keep_order([primary] + from_is)
        if not all_factors:
            all_factors = ["Unspecified"]

        total_veh = to_int_safe(row.get("TOTAL_VEHICLES", 1), 1)
        n_slots = min(max(total_veh, 1), 5)

        # Keep at least one cause; remaining slots follow TOTAL_VEHICLES.
        assigned = all_factors[:n_slots]
        while len(assigned) < n_slots:
            assigned.append("Unspecified")

        padded = assigned + [""] * (5 - len(assigned))
        for i in range(1, 6):
            factor_cols[f"CONTRIBUTING FACTOR VEHICLE {i}"].append(padded[i - 1])

    df_out = df.copy()
    df_out.insert(0, "CRASH DATE", crash_dates)
    df_out.insert(1, "CRASH TIME", crash_times)
    df_out.insert(2, "CRASH_FULL_TIME", full_times)

    for i in range(1, 6):
        df_out[f"CONTRIBUTING FACTOR VEHICLE {i}"] = factor_cols[f"CONTRIBUTING FACTOR VEHICLE {i}"]

    is_cols = [c for c in df_out.columns if c.startswith("IS_")]
    non_is_cols = [c for c in df_out.columns if not c.startswith("IS_")]
    df_out = df_out[non_is_cols + is_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Build a compact report for quick inspection.
    dt_series = pd.to_datetime(df_out["CRASH DATE"] + " " + df_out["CRASH TIME"], errors="coerce")
    season_check = ((dt_series.dt.month % 12) // 3).fillna(-1).astype(int)
    dow_check = dt_series.dt.weekday.fillna(-1).astype(int)
    period_check = pd.Series(0, index=df_out.index)
    hour = dt_series.dt.hour.fillna(0).astype(int)
    period_check[(hour >= 7) & (hour <= 9)] = 1
    period_check[(hour >= 10) & (hour <= 15)] = 2
    period_check[(hour >= 16) & (hour <= 19)] = 3

    season_match = (season_check == pd.to_numeric(df_out["CRASH_SEASON"], errors="coerce").fillna(-1).astype(int)).mean() * 100
    dow_match = (dow_check == pd.to_numeric(df_out["DAY_OF_WEEK"], errors="coerce").fillna(-1).astype(int)).mean() * 100
    period_match = (period_check == pd.to_numeric(df_out["CRASH_TIME_PERIOD"], errors="coerce").fillna(-1).astype(int)).mean() * 100

    lines = []
    lines.append("Reconstructed 2017-like Dataset Report")
    lines.append(f"Input: {input_csv}")
    lines.append(f"Output: {output_csv}")
    lines.append(f"Rows: {len(df_out)}, Cols: {df_out.shape[1]}")
    lines.append("")
    lines.append("[Date/Time Backfill Consistency]")
    lines.append(f"CRASH_SEASON match: {season_match:.2f}%")
    lines.append(f"DAY_OF_WEEK match: {dow_match:.2f}%")
    lines.append(f"CRASH_TIME_PERIOD match: {period_match:.2f}%")
    lines.append("")
    lines.append("[Contributing Factor Preview]")
    for i in range(1, 6):
        vc = df_out[f"CONTRIBUTING FACTOR VEHICLE {i}"].value_counts(dropna=False).head(5)
        lines.append(f"Top-{i} factor values:")
        lines.append(vc.to_string())
        lines.append("")
    lines.append("[Sample Rows x10]")
    show_cols = [
        "CRASH DATE", "CRASH TIME", "LATITUDE", "LONGITUDE",
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
        "TOTAL_VEHICLES", "IS_MULTI_VEHICLE", "NUMBER OF PERSONS INJURED",
    ]
    show_cols = [c for c in show_cols if c in df_out.columns]
    lines.append(df_out[show_cols].head(10).to_string(index=False))

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct source-like 2017 dataset from synthetic data")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="exp/nyc_crash_v3/causal_m4_v6_catY_full/synthetic_complete.csv",
        help="Input synthetic complete CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="exp/nyc_crash_v3/causal_m4_v6_catY_full/synthetic_2017_like.csv",
        help="Output reconstructed CSV",
    )
    parser.add_argument(
        "--report_txt",
        type=str,
        default="exp/nyc_crash_v3/causal_m4_v6_catY_full/synthetic_2017_like_report.txt",
        help="Output text report",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reconstruct(Path(args.input_csv), Path(args.output_csv), Path(args.report_txt), seed=args.seed)
    print("✅ Reconstructed dataset saved:", args.output_csv)
    print("✅ Report saved:", args.report_txt)


if __name__ == "__main__":
    main()
