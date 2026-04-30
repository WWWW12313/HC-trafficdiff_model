from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click pipeline: train -> evaluate -> output")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v9.csv")
    parser.add_argument("--profiles", type=str, default="quick,balanced,full")
    parser.add_argument("--output_root", type=str, default="exp/nonbaseline_suite_v9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_gemini_soft", action="store_true")
    parser.add_argument("--gemini_model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--run_v9_llm", action="store_true")
    parser.add_argument("--llm_mode", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    py = sys.executable

    train_state = Path(args.output_root) / "train_state.json"
    eval_state = Path(args.output_root) / "eval_state.json"
    report_md = Path(args.output_root) / "nonbaseline_unified_report.md"

    train_cmd = [
        py,
        "scripts/nonbaseline_train.py",
        "--repo_root",
        str(root),
        "--reference_csv",
        args.reference_csv,
        "--profiles",
        args.profiles,
        "--output_root",
        args.output_root,
        "--seed",
        str(args.seed),
        "--gemini_model",
        args.gemini_model,
        "--llm_mode",
        args.llm_mode,
        "--llm_model",
        args.llm_model,
    ]
    if args.enable_gemini_soft:
        train_cmd.append("--enable_gemini_soft")
    if args.run_v9_llm:
        train_cmd.append("--run_v9_llm")

    eval_cmd = [
        py,
        "scripts/nonbaseline_evaluate.py",
        "--repo_root",
        str(root),
        "--train_state",
        str(train_state),
        "--reference_csv",
        args.reference_csv,
        "--output_eval_state",
        str(eval_state),
    ]

    output_cmd = [
        py,
        "scripts/nonbaseline_output.py",
        "--eval_state",
        str(eval_state),
        "--out_md",
        str(report_md),
    ]

    run_cmd(train_cmd, root)
    run_cmd(eval_cmd, root)
    run_cmd(output_cmd, root)

    print("All done.")
    print(f"train_state: {train_state.as_posix()}")
    print(f"eval_state: {eval_state.as_posix()}")
    print(f"report: {report_md.as_posix()}")


if __name__ == "__main__":
    main()
