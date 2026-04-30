from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "v7_mode": "quick",
        "v8_n_samples": 20000,
        "v8_k_candidates": 2,
        "gemini_max_rows": 30,
        "v8_weather_source": "donor",
        "v8_osm_source": "donor",
    },
    "balanced": {
        "v7_mode": "balanced",
        "v8_n_samples": 80000,
        "v8_k_candidates": 3,
        "gemini_max_rows": 80,
        "v8_weather_source": "donor",
        "v8_osm_source": "donor",
    },
    "full": {
        "v7_mode": "full",
        "v8_n_samples": 159992,
        "v8_k_candidates": 4,
        "gemini_max_rows": 150,
        "v8_weather_source": "donor",
        "v8_osm_source": "donor",
    },
}


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.repo_root).resolve()
    py = sys.executable

    selected_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in selected_profiles:
        if p not in PROFILES:
            raise ValueError(f"Unknown profile: {p}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Shared v7 data prep (v9 based) once.
    v7_data_dir = Path(args.v7_data_dir)
    run_cmd(
        [
            py,
            "scripts/prepare_data_v7.py",
            "--input_csv",
            args.reference_csv,
            "--output_dir",
            str(v7_data_dir),
            "--seed",
            str(args.seed),
        ],
        cwd=root,
    )

    runs: Dict[str, Any] = {}
    for profile in selected_profiles:
        cfg = PROFILES[profile]
        prof_dir = out_root / profile
        prof_dir.mkdir(parents=True, exist_ok=True)

        # v7 train + sample
        v7_model_dir = prof_dir / "v7"
        v7_model_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                py,
                "train_causal_v7.py",
                "--data",
                str(v7_data_dir),
                "--output",
                str(v7_model_dir),
                "--mode",
                cfg["v7_mode"],
            ],
            cwd=root,
        )
        v7_syn_csv = v7_model_dir / "synthetic_2017_v7.csv"
        run_cmd(
            [
                py,
                "scripts/sample_two_stage_v7.py",
                "--model_dir",
                str(v7_model_dir),
                "--data_dir",
                str(v7_data_dir),
                "--reference_csv",
                args.reference_csv,
                "--output_csv",
                str(v7_syn_csv),
                "--n_samples",
                str(cfg["v8_n_samples"]),
                "--seed",
                str(args.seed),
            ],
            cwd=root,
        )

        # v8 ablation sampling
        v8_dir = prof_dir / "v8_ablation"
        v8_dir.mkdir(parents=True, exist_ok=True)

        for mode in ["free", "hard", "soft"]:
            cmd = [
                py,
                "scripts/v8_ablation_sampler.py",
                "--mode",
                mode,
                "--reference_csv",
                args.reference_csv,
                "--output_dir",
                str(v8_dir),
                "--n_samples",
                str(cfg["v8_n_samples"]),
                "--k_candidates",
                str(cfg["v8_k_candidates"]),
                "--seed",
                str(args.seed),
                "--weather_source",
                str(cfg["v8_weather_source"]),
                "--osm_source",
                str(cfg["v8_osm_source"]),
            ]

            # Soft mode can optionally use low-quota Gemini.
            if mode == "soft" and args.enable_gemini_soft:
                cmd.extend(
                    [
                        "--penalty_backend",
                        "gemini",
                        "--quota_profile",
                        "low",
                        "--gemini_model",
                        args.gemini_model,
                        "--gemini_max_rows",
                        str(cfg["gemini_max_rows"]),
                    ]
                )
            run_cmd(cmd, cwd=root)

        runs[profile] = {
            "profile": profile,
            "v7_model_dir": str(v7_model_dir),
            "v7_synthetic_csv": str(v7_syn_csv),
            "v8_dir": str(v8_dir),
            "n_samples": int(cfg["v8_n_samples"]),
        }

    llm_v9 = None
    if args.run_v9_llm:
        llm_dir = out_root / "v9_llm"
        llm_dir.mkdir(parents=True, exist_ok=True)
        out_json = llm_dir / "physics_dag_v9.json"
        run_cmd(
            [
                py,
                "scripts/llm_physics_dag_v9.py",
                "--input_csv",
                args.reference_csv,
                "--llm_mode",
                args.llm_mode,
                "--model",
                args.llm_model,
                "--output_json",
                str(out_json),
                "--output_prompt",
                str(llm_dir / "physics_dag_prompt_v9.txt"),
            ],
            cwd=root,
        )
        llm_v9 = {"output_json": str(out_json), "llm_mode": args.llm_mode, "model": args.llm_model}

    payload = {
        "reference_csv": args.reference_csv,
        "seed": args.seed,
        "profiles": runs,
        "llm_v9": llm_v9,
    }
    state_path = out_root / "train_state.json"
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved train state: {state_path.as_posix()}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train non-baseline suite for v7/v8 and optional v9 LLM DAG")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--reference_csv", type=str, default="nyc_2017_pristine_v9.csv")
    parser.add_argument("--v7_data_dir", type=str, default="data/nyc_crash_v7_v9")
    parser.add_argument("--profiles", type=str, default="quick,balanced,full")
    parser.add_argument("--output_root", type=str, default="exp/nonbaseline_suite_v9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_gemini_soft", action="store_true")
    parser.add_argument("--gemini_model", type=str, default="models/gemini-2.5-flash")
    parser.add_argument("--run_v9_llm", action="store_true")
    parser.add_argument("--llm_mode", type=str, default="mock", choices=["mock", "openai"])
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    main()
