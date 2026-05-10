from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.road_snap import enrich_road_context
from src.sample_conditional import repair_stage1_spatial_support


DEFAULT_FILES = [
    "baseline_tabddpm_full.csv",
    "ablation_no_causal_full.csv",
    "ablation_no_hierarchy_full.csv",
    "ours_full_model_full.csv",
]


def align_one(
    input_csv: Path,
    output_csv: Path,
    reference_csv: Path,
    graphml_path: Path,
    signals_path: Path | None,
    seed: int,
) -> None:
    df = pd.read_csv(input_csv, low_memory=False)
    df = repair_stage1_spatial_support(df, reference_csv=str(reference_csv), seed=seed)
    df = enrich_road_context(
        df,
        graphml_path=graphml_path,
        signals_path=signals_path,
        columns={
            "DIST_TO_SIGNAL_M",
            "HAS_TRAFFIC_SIGNAL",
            "OSM_TYPE",
            "OSM_ONEWAY",
            "INFERRED_LANES",
            "REAL_SPEED_LIMIT",
            "HAS_DIVIDER",
        },
        overwrite=True,
        verbose=True,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[saved] {output_csv} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=ROOT / "results" / "synthetic_2024_source")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "synthetic_2024_source_support_repair")
    parser.add_argument("--reference_csv", type=Path, default=ROOT / "data" / "nyc_crash_2024" / "train.csv")
    parser.add_argument("--graphml", type=Path, default=ROOT / "raw_data" / "osm" / "2024" / "nyc_drive_graph.graphml")
    parser.add_argument("--signals", type=Path, default=ROOT / "raw_data" / "osm" / "2024" / "nyc_traffic_signals.geojson")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_stage2_support_repair",
        action="store_true",
        help="Copy the existing ours_stage2_causal support-repair output into output_dir for joint evaluation.",
    )
    args = parser.parse_args()

    signals_path = args.signals if args.signals.exists() else None
    for filename in args.files:
        input_csv = args.input_dir / filename
        if not input_csv.exists():
            print(f"[skip] missing input: {input_csv}")
            continue
        print(f"\n[align] {filename}")
        align_one(
            input_csv=input_csv,
            output_csv=args.output_dir / filename,
            reference_csv=args.reference_csv,
            graphml_path=args.graphml,
            signals_path=signals_path,
            seed=args.seed,
        )

    if args.copy_stage2_support_repair:
        src = ROOT / "results" / "synthetic_2024_stage2_causal_unified_road_support_repair" / "ours_stage2_causal_full.csv"
        dst = args.output_dir / "ours_stage2_causal_full.csv"
        if src.exists():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[copied] {src} -> {dst}")
        else:
            print(f"[skip] missing support-repair ours_stage2 output: {src}")


if __name__ == "__main__":
    main()