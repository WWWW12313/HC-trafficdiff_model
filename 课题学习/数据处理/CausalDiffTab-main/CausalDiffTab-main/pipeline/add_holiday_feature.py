"""
Add HOLIDAY feature to NYC crash data CSV files.

Maps crash dates to US federal holiday flags and weekday proximity flags.
Requires 'holidays' package: pip install holidays

Usage:
  python pipeline/add_holiday_feature.py \
      --input data/nyc_crash/train.csv \
      --output data/nyc_crash/train_with_holiday.csv \
      --date_col CRASH_DATE \
      --year 2017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CDT_ROOT = Path(__file__).resolve().parent.parent

# US Federal Holidays (MM-DD format for date-agnostic lookup)
_FIXED_HOLIDAYS = {
    "01-01",  # New Year's Day
    "07-04",  # Independence Day
    "11-11",  # Veterans Day
    "12-25",  # Christmas Day
    "12-24",  # Christmas Eve (observed)
    "12-31",  # New Year's Eve (observed)
}

# Floating holidays by rule (applied when 'holidays' package not available)
def _is_near_holiday_by_season(row: pd.Series) -> int:
    """Heuristic: map SEASON + DAY_OF_WEEK to approximate holiday likelihood.

    Returns 1 if likely holiday period, 0 otherwise.
    This is a fallback when actual dates are unavailable.
    """
    season = str(row.get("SEASON", "")).lower()
    dow = str(row.get("DAY_OF_WEEK", "")).lower()

    # Weekend flag (most common holiday-adjacent days)
    is_weekend = dow in {"0", "6", "saturday", "sunday", "sat", "sun"}

    # Holiday seasons: late December / early January → winter
    if season == "winter" and is_weekend:
        return 1
    # Summer long weekends (Memorial Day/July 4th/Labor Day)
    if season == "summer" and is_weekend:
        return 1
    # Autumn (Thanksgiving)
    if season == "autumn" and is_weekend:
        return 1
    return 0


def add_holiday_from_date(df: pd.DataFrame, date_col: str, year: Optional[int] = None) -> pd.DataFrame:
    """Add HOLIDAY column (0/1) using actual date column.

    Args:
        df: DataFrame with a date column.
        date_col: Column name containing date strings.
        year: Hint year for lookup (uses actual dates if parseable).

    Returns:
        DataFrame with new HOLIDAY column added.
    """
    try:
        import holidays as hol_lib
        use_lib = True
    except ImportError:
        print("[warn] 'holidays' package not found. Using fixed-date fallback.")
        use_lib = False

    df = df.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mmdd = dates.dt.strftime("%m-%d").fillna("")

    if use_lib:
        target_years = [year] if year else list(dates.dt.year.dropna().astype(int).unique())
        us_holidays = set()
        for y in target_years:
            us_holidays.update(hol_lib.country_holidays("US", years=y).keys())
        df["HOLIDAY"] = dates.dt.date.apply(lambda d: int(d in us_holidays) if pd.notna(d) else 0)
    else:
        df["HOLIDAY"] = mmdd.apply(lambda x: int(x in _FIXED_HOLIDAYS))

    df["IS_WEEKEND"] = ((dates.dt.dayofweek >= 5)).astype(int)
    df["IS_HOLIDAY_OR_WEEKEND"] = ((df["HOLIDAY"] == 1) | (df["IS_WEEKEND"] == 1)).astype(int)

    print(f"[holiday] Holiday rate: {float(df['HOLIDAY'].mean()):.4f}")
    print(f"[holiday] Weekend rate: {float(df['IS_WEEKEND'].mean()):.4f}")
    print(f"[holiday] Holiday-or-weekend rate: {float(df['IS_HOLIDAY_OR_WEEKEND'].mean()):.4f}")
    return df


def add_holiday_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: add HOLIDAY proxy from SEASON + DAY_OF_WEEK columns (no date needed).

    Useful when only the processed feature CSV is available.
    """
    df = df.copy()
    df["HOLIDAY"] = df.apply(_is_near_holiday_by_season, axis=1)
    rate = float(df["HOLIDAY"].mean())
    print(f"[holiday_proxy] Rate (heuristic season+weekend): {rate:.4f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add HOLIDAY feature to NYC crash CSV")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--date_col",
        type=str,
        default="CRASH DATE",
        help="Date column name (if available)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year hint for holiday lookup",
    )
    parser.add_argument(
        "--proxy_mode",
        action="store_true",
        help="Use SEASON+DAY_OF_WEEK proxy instead of real date column",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        inp = CDT_ROOT / args.input
    out = Path(args.output)
    if not out.is_absolute():
        out = CDT_ROOT / args.output

    print(f"[load] {inp}")
    df = pd.read_csv(inp, low_memory=False)
    print(f"[load] shape={df.shape}")

    if args.proxy_mode or args.date_col not in df.columns:
        print("[mode] Using proxy mode (SEASON + DAY_OF_WEEK)")
        df = add_holiday_proxy(df)
    else:
        df = add_holiday_from_date(df, args.date_col, args.year)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[save] {out} (shape={df.shape})")

    # Update info.json hint
    info_path = CDT_ROOT / "data" / "nyc_crash" / "info.json"
    if info_path.is_file():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        cat_names = list(info.get("cat_col_names", []))
        added = []
        for col in ["HOLIDAY", "IS_WEEKEND", "IS_HOLIDAY_OR_WEEKEND"]:
            if col in df.columns and col not in cat_names:
                cat_names.append(col)
                added.append(col)
        if added:
            print(f"[info.json] Add to cat_col_names: {added}")
            print("[info.json] NOTE: Manually update info.json and re-train to use these features.")


if __name__ == "__main__":
    main()
