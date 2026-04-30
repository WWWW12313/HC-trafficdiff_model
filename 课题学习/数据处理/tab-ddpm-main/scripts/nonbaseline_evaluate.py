from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.repo_root).resolve()
    py = sys.executable

    state_path = Path(args.train_state)
    payload = json.loads(state_path.read_text(encoding="utf-8"))

    reference_csv = payload.get("reference_csv", args.reference_csv)
    results: Dict[str, Any] = {"reference_csv": reference_csv, "profiles": {}}

    profiles = payload.get("profiles", {})
    for profile, info in profiles.items():
        v7_syn_csv = info["v7_synthetic_csv"]
        v7_model_dir = Path(info["v7_model_dir"])
        v8_dir = Path(info["v8_dir"])

        v7_eval_json = v7_model_dir / f"eval_v7_{profile}.json"
        run_cmd(
            [
                py,
                "scripts/evaluate_v7.py",
                "--real_csv",
                reference_csv,
                "--syn_csv",
                str(v7_syn_csv),
                "--out_json",
                str(v7_eval_json),
            ],
            cwd=root,
        )

        v8_eval_json = v8_dir / "v8_ablation_metrics.json"
        v8_eval_md = v8_dir / "v8_ablation_report.md"
        run_cmd(
            [
                py,
                "scripts/v8_ablation_eval.py",
                "--real_csv",
                reference_csv,
                "--base_dir",
                str(v8_dir),
                "--out_md",
                str(v8_eval_md),
                "--out_json",
                str(v8_eval_json),
            ],
            cwd=root,
        )

        results["profiles"][profile] = {
            "v7_eval_json": str(v7_eval_json),
            "v8_eval_json": str(v8_eval_json),
            "v8_eval_md": str(v8_eval_md),
        }

    eval_state_path = Path(args.output_eval_state)
    eval_state_path.parent.mkdir(parents=True, exist_ok=True)
    eval_state_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved eval state: {eval_state_path.as_posix()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-baseline suite outputs")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--train_state", type=str, default="exp/nonbaseline_suite_v9/train_state.json")
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v9.csv")
    parser.add_argument("--output_eval_state", type=str, default="exp/nonbaseline_suite_v9/eval_state.json")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
