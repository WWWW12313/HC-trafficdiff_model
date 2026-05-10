from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.postprocess_samples import (  # noqa: E402
    DEFAULT_TARGET_CONDITION_COLS,
    _causal_target_resample,
    _causal_weather_resample,
)


DEFAULT_FILES = [
    "baseline_tabddpm_full.csv",
    "ablation_no_causal_full.csv",
    "ablation_no_hierarchy_full.csv",
    "ours_full_model_full.csv",
    "ours_stage2_causal_full.csv",
]


def calibrate_one(
    input_csv: Path,
    output_csv: Path,
    weather_reference_csv: Path | None,
    target_reference_csv: Path | None,
    weather_condition_cols: list[str],
    target_condition_cols: list[str],
    weather_min_bucket_size: int,
    target_min_bucket_size: int,
    seed: int,
) -> dict:
    df = pd.read_csv(input_csv, low_memory=False)
    report: dict[str, object] = {"input": str(input_csv), "output": str(output_csv), "rows": len(df)}

    if weather_reference_csv is not None:
        df, weather_stats = _causal_weather_resample(
            df,
            reference_csv=str(weather_reference_csv),
            condition_cols=weather_condition_cols,
            min_bucket_size=weather_min_bucket_size,
            seed=seed,
        )
        report["weather"] = weather_stats

    if target_reference_csv is not None:
        before_mean = pd.to_numeric(df["NUMBER OF PERSONS INJURED"], errors="coerce").mean()
        df, target_stats = _causal_target_resample(
            df,
            reference_csv=str(target_reference_csv),
            target_col="NUMBER OF PERSONS INJURED",
            condition_cols=target_condition_cols,
            min_bucket_size=target_min_bucket_size,
            seed=seed,
            enforce_injury_lower_bound=True,
        )
        after_mean = pd.to_numeric(df["NUMBER OF PERSONS INJURED"], errors="coerce").mean()
        target_stats["before_mean"] = float(before_mean)
        target_stats["after_mean"] = float(after_mean)
        report["target"] = target_stats

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply causal weather resampling and source-domain target calibration to synthetic CSVs."
    )
    parser.add_argument("--input_dir", default=str(ROOT / "results" / "synthetic_2024_source_support_repair"))
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "synthetic_2024_source_support_repair_causal_calibrated"))
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--weather_reference_csv", default=str(ROOT / "data" / "nyc_crash_2025" / "test.csv"))
    parser.add_argument("--target_reference_csv", default=str(ROOT / "data" / "nyc_crash_2024" / "train.csv"))
    parser.add_argument("--weather_condition_cols", default="SEASON,TIME_PERIOD")
    parser.add_argument("--target_condition_cols", default=",".join(DEFAULT_TARGET_CONDITION_COLS))
    parser.add_argument("--weather_min_bucket_size", type=int, default=30)
    parser.add_argument("--target_min_bucket_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_weather", action="store_true")
    parser.add_argument("--no_target", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    weather_reference_csv = None if args.no_weather else Path(args.weather_reference_csv)
    target_reference_csv = None if args.no_target else Path(args.target_reference_csv)
    weather_condition_cols = [c.strip() for c in args.weather_condition_cols.split(",") if c.strip()]
    target_condition_cols = [c.strip() for c in args.target_condition_cols.split(",") if c.strip()]

    reports = []
    for name in args.files:
        input_csv = input_dir / name
        if not input_csv.exists():
            print(f"[skip] missing: {input_csv}")
            continue
        output_csv = output_dir / name
        report = calibrate_one(
            input_csv=input_csv,
            output_csv=output_csv,
            weather_reference_csv=weather_reference_csv,
            target_reference_csv=target_reference_csv,
            weather_condition_cols=weather_condition_cols,
            target_condition_cols=target_condition_cols,
            weather_min_bucket_size=args.weather_min_bucket_size,
            target_min_bucket_size=args.target_min_bucket_size,
            seed=args.seed,
        )
        reports.append(report)
        target_report = report.get("target", {})
        weather_report = report.get("weather", {})
        print(
            f"[ok] {name}: "
            f"weather_exact={weather_report.get('exact', 'NA')} "
            f"target_mean={target_report.get('before_mean', 'NA')}->{target_report.get('after_mean', 'NA')}"
        )

    pd.DataFrame(reports).to_json(output_dir / "calibration_report.json", orient="records", indent=2, force_ascii=False)
    print(f"[done] wrote {len(reports)} files to {output_dir}")


if __name__ == "__main__":
    main()