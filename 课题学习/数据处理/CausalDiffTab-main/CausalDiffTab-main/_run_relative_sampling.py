#!/usr/bin/env python
"""Run relative guidance sampling for macro_sparse_anneal_v2 on 2025 data."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabdiff"))

from src.sample_conditional import run_sampling

if __name__ == "__main__":
    print("[run] Starting relative guidance sampling...")
    run_sampling(
        ckpt_dir="ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2",
        data_dir="data/nyc_crash_2025_v2",
        condition_csv=None,
        condition_train_indices="results/_cache/eval_uniform_5k.txt",
        num_samples=5000,
        batch_size=500,
        device="cuda:0",
        output_csv="results/synthetic/transfer_2025_macro_sparse_relative_g05_uniform5k.csv",
        do_postprocess=True,
        impute_resample_rounds=1,
        impute_condition="x_0",
        impute_stage="stage3",
        causal_guidance_scale=0.0,
        macro_guidance_scale=0.5,
        macro_guidance_mode="relative",
        macro_guidance_adaptive_drift_threshold=2.0,
    )
    print("[run] Relative guidance sampling complete.")
