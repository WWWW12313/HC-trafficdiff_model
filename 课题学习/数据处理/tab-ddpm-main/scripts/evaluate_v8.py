"""
evaluate_v8.py

v8 evaluation:
- Reuses v7 evaluation outputs.
- Adds Commonsense_Violation_Rate for 3 strict rules:
  1) Spatio-temporal deterministic weather linkage proxy
  2) Snowplow seasonal/weather guard
  3) Multi-vehicle math constraint (TOTAL_VEHICLES >= 2)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_v7 import evaluate as evaluate_v7  # type: ignore


def safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _vehicle_present(s: pd.Series) -> pd.Series:
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return (~txt.isin(["", "nan", "none", "null", "unknown", "unspecified"]))


def commonsense_violation_rate(df: pd.DataFrame) -> Dict[str, float]:
    n = max(len(df), 1)

    # Rule 1 proxy: weather must exist and be numerically valid
    weather_cols = ["CTX_TEMP", "CTX_PRCP", "CTX_WSPD", "CTX_COCO"]
    miss = pd.Series(False, index=df.index)
    for c in weather_cols:
        if c not in df.columns:
            miss = pd.Series(True, index=df.index)
            break
        miss = miss | safe_num(df[c], default=np.nan).isna()
    r1 = float(miss.mean())

    # Rule 2: snowplow guard
    veh_cols = [c for c in ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"] if c in df.columns]
    if not veh_cols:
        r2 = 0.0
    else:
        month = pd.to_datetime(df.get("CRASH DATE", pd.Series(["2017-01-01"] * len(df))), errors="coerce").dt.month.fillna(1).astype(int)
        temp = safe_num(df.get("CTX_TEMP", pd.Series(20.0, index=df.index)), default=20.0)
        prcp = safe_num(df.get("CTX_PRCP", pd.Series(0.0, index=df.index)), default=0.0)
        coco = safe_num(df.get("CTX_COCO", pd.Series(1, index=df.index)), default=1).astype(int)

        winter = month.isin([12, 1, 2])
        snowy = coco.isin([15, 16]) | ((temp <= 2.0) & (prcp > 0))
        allowed = winter | snowy

        snow_any = pd.Series(False, index=df.index)
        for c in veh_cols:
            v = df[c].fillna("").astype(str).str.lower()
            snow_any = snow_any | v.str.contains("snow plow|snowplow|plow", regex=True)

        r2 = float((snow_any & (~allowed)).mean())

    # Rule 3: TOTAL_VEHICLES math
    c1 = "VEHICLE TYPE CODE 1" if "VEHICLE TYPE CODE 1" in df.columns else None
    c2 = "VEHICLE TYPE CODE 2" if "VEHICLE TYPE CODE 2" in df.columns else None
    if c1 is None or c2 is None or "TOTAL_VEHICLES" not in df.columns:
        r3 = 0.0
    else:
        both = _vehicle_present(df[c1]) & _vehicle_present(df[c2])
        total = safe_num(df["TOTAL_VEHICLES"], default=1.0)
        r3 = float((both & (total < 2)).mean())

    overall = float((r1 + r2 + r3) / 3.0)
    return {
        "rule1_weather_determinism_proxy": r1,
        "rule2_snowplow_guard": r2,
        "rule3_multi_vehicle_math": r3,
        "Commonsense_Violation_Rate": overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v8 synthetic dataset")
    parser.add_argument("--real_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--syn_csv", type=str, default="exp/nyc_crash_v8/synthetic_2017_v8.csv")
    parser.add_argument("--out_json", type=str, default="exp/nyc_crash_v8/eval_v8.json")
    args = parser.parse_args()

    # First generate base v7-compatible metrics
    evaluate_v7(args.real_csv, args.syn_csv, args.out_json)

    # Then append commonsense metrics
    out_path = Path(args.out_json)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    syn_df = pd.read_csv(args.syn_csv)
    payload["commonsense"] = commonsense_violation_rate(syn_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== v8 evaluation done ===")
    print(json.dumps(payload["commonsense"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
