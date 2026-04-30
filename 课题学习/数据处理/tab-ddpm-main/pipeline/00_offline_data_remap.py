from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.check_offline_consistency import Config, run, setup_logging


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_phase0_config(cfg: Dict[str, Any], repo_root: Path) -> Config:
    paths = cfg["paths"]
    offline = cfg["offline_remap"]

    def abs_path(v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else repo_root / p

    return Config(
        raw_csv=abs_path(paths["raw_csv"]),
        pbf_path=abs_path(paths["pbf_path"]),
        weather_csv=abs_path(paths["weather_csv"]),
        pristine_csv=abs_path(paths["pristine_v8_csv"]),
        report_json=abs_path(offline["report_json"]),
        remapped_csv=abs_path(offline["remapped_csv"]),
        v9_output_csv=abs_path(offline["v9_output_csv"]),
        join_key=str(offline.get("join_key", "COLLISION_ID")),
        signal_threshold_m=float(offline.get("signal_threshold_m", 20.0)),
        weather_rounding=str(offline.get("weather_rounding", "floor")),
        accept_rate=float(offline.get("accept_rate", 0.35)),
        force_overwrite=bool(offline.get("force_overwrite", False)),
        sample_n=int(offline.get("sample_n", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 00: offline remap and consistency check")
    parser.add_argument("--config", type=str, default="configs/v9_experiment.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(repo_root / args.config)

    setup_logging(verbose=args.verbose)
    phase0_cfg = build_phase0_config(cfg, repo_root=repo_root)
    report = run(phase0_cfg)

    print("[pipeline/00] done")
    print(f"overall_discrepancy_rate={report.get('overall_discrepancy_rate')}")
    print(f"v9_written={report.get('v9_written')}")


if __name__ == "__main__":
    main()
