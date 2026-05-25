#!/bin/bash
set -e
MODELS=(
  "exp_c_macro_full_v1"
  "macro_sparse_anneal_v2"
  "ablation_no_causal_v2"
  "ablation_no_hierarchy_v2"
  "baseline_tabddpm_v2"
)
IDX="results/_cache/eval_uniform_5k.txt"

for model in "${MODELS[@]}"; do
  echo "========================================"
  echo "[transfer] Sampling $model on 2025 data"
  echo "========================================"
  conda run -n crashgen python src/sample_conditional.py \
    --ckpt_dir "ckpt/nyc_crash_2024_v2/stage3_full_full_${model}" \
    --data_dir "data/nyc_crash_2025_v2" \
    --condition_train_indices "$IDX" \
    --num_samples 5000 --batch_size 500 --device cuda:0 \
    --output_csv "results/synthetic/transfer_2025_${model}_uniform5k.csv"
done

echo "========================================"
echo "[transfer] Sampling Ours + guidance=0.5 on 2025 data"
echo "========================================"
conda run -n crashgen python src/sample_conditional.py \
  --ckpt_dir "ckpt/nyc_crash_2024_v2/stage3_full_full_exp_c_macro_full_v1" \
  --data_dir "data/nyc_crash_2025_v2" \
  --condition_train_indices "$IDX" \
  --num_samples 5000 --batch_size 500 --device cuda:0 \
  --output_csv "results/synthetic/transfer_2025_ours_guide0p5_uniform5k.csv" \
  --macro_guidance_scale 0.5

echo "All transfer sampling complete!"
