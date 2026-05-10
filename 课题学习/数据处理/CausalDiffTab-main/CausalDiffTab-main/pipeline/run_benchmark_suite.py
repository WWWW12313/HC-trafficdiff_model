"""
Unified benchmark suite runner for CausalDiffTab.

What it does:
1) Run internal models (baseline/ablations/ours) via run_all_experiments.py
2) Run external baselines (CTGAN/TVAE/SMOTE) via run_external_baselines.py
3) Run unified evaluation via evaluate_all.py
4) Package key files and artifacts into a timestamped bundle directory

Examples:
  python pipeline/run_benchmark_suite.py --tier balanced
  python pipeline/run_benchmark_suite.py --tier full --skip_internal_train
  python pipeline/run_benchmark_suite.py --tier full --only_package
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)

INTERNAL_MODELS = [
    "baseline_tabddpm",
    "ablation_no_causal",
    "ablation_no_hierarchy",
    "ours_full_model",
]


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _run_internal(tier: str, device: str, skip_internal_train: bool) -> None:
    for model in INTERNAL_MODELS:
        cmd = [
            str(PYTHON),
            str(ROOT / "pipeline" / "run_all_experiments.py"),
            "--model",
            model,
            "--tier",
            tier,
            "--device",
            device,
        ]
        if skip_internal_train:
            cmd.append("--skip_train")
        _run(cmd)


def _run_external(tier: str) -> None:
    cmd = [
        str(PYTHON),
        str(ROOT / "pipeline" / "run_external_baselines.py"),
        "--tier",
        tier,
    ]
    _run(cmd)


def _run_eval(tier: str, task_type: str | None = None) -> None:
    cmd = [
        str(PYTHON),
        str(ROOT / "pipeline" / "evaluate_all.py"),
        "--real_test",
        "synthetic/nyc_crash/test.csv",
        "--file_glob",
        f"*_{tier}.csv",
    ]
    if task_type:
        cmd.extend(["--task_type", task_type])
    _run(cmd)


def _package_bundle(tier: str, include_glob: str = "*.csv") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = ROOT / "results" / "bundles" / f"benchmark_{tier}_{ts}"
    bundle.mkdir(parents=True, exist_ok=True)

    copied = []

    key_files = [
        ROOT / "src" / "train_hierarchical.py",
        ROOT / "src" / "postprocess_samples.py",
        ROOT / "src" / "sample_conditional.py",
        ROOT / "pipeline" / "run_all_experiments.py",
        ROOT / "pipeline" / "run_external_baselines.py",
        ROOT / "pipeline" / "evaluate_all.py",
        ROOT / "pipeline" / "benchmark_evaluator.py",
        ROOT / "pipeline" / "run_benchmark_suite.py",
        ROOT / "data" / "nyc_crash" / "info.json",
        ROOT / "data" / "processed" / "column_groups.json",
        ROOT / "data" / "processed" / "continuous_scaler.pkl",
    ]
    for p in (ROOT / "configs" / "experiments").glob("*.yaml"):
        key_files.append(p)

    for src in key_files:
        rel = src.relative_to(ROOT)
        dst = bundle / rel
        if _copy_if_exists(src, dst):
            copied.append(str(rel))

    syn_dir = ROOT / "results" / "synthetic"
    for src in syn_dir.glob(f"*_{tier}.csv"):
        rel = src.relative_to(ROOT)
        dst = bundle / rel
        if _copy_if_exists(src, dst):
            copied.append(str(rel))

    for extra in [
        ROOT / "results" / "eval_report_latest.json",
        ROOT / "results" / "eval_report_latest.md",
    ]:
        rel = extra.relative_to(ROOT)
        dst = bundle / rel
        if _copy_if_exists(extra, dst):
            copied.append(str(rel))

    manifest = {
        "created_at": datetime.now().isoformat(),
        "tier": tier,
        "python": str(PYTHON),
        "copied_files": copied,
    }
    with open(bundle / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[bundle] {bundle}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark suite and package key artifacts")
    parser.add_argument("--tier", choices=["quick", "balanced", "full"], default="balanced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_internal_train", action="store_true")
    parser.add_argument("--skip_internal", action="store_true")
    parser.add_argument("--skip_external", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--only_package", action="store_true")
    parser.add_argument(
        "--task_type",
        choices=["regression", "classification"],
        default=None,
        help="透传到 evaluate_all.py，用于覆盖 info.json 中的任务类型",
    )
    args = parser.parse_args()

    if args.only_package:
        _package_bundle(args.tier)
        return

    if not args.skip_internal:
        _run_internal(args.tier, args.device, args.skip_internal_train)

    if not args.skip_external:
        _run_external(args.tier)

    if not args.skip_eval:
        _run_eval(args.tier, args.task_type)

    _package_bundle(args.tier)


if __name__ == "__main__":
    main()
