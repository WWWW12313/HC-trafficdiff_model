#!/bin/bash
set -e

# Evaluate robust guidance experiments on 2025 transfer
# Usage: bash scripts/evaluate_robust_guidance.sh

REAL="data/nyc_crash_2025_v2/test.csv"

echo "============================================"
echo "Evaluating Robust Guidance Experiments"
echo "============================================"

# Baseline (no guidance) - already evaluated, just for reference
# echo "[eval] Baseline (no guidance)"
# python pipeline/evaluate_macro_relations.py --real "$REAL" --synthetic results/synthetic/transfer_2025_macro_sparse_anneal_v2_uniform5k_physical.csv

# Adaptive guidance
if [ -f "results/synthetic/transfer_2025_macro_sparse_adaptive_g05_uniform5k_physical.csv" ]; then
    echo "[eval] Adaptive guidance (scale=0.5)"
    conda run -n crashgen python pipeline/evaluate_macro_relations.py \
        --real "$REAL" \
        --synthetic results/synthetic/transfer_2025_macro_sparse_adaptive_g05_uniform5k_physical.csv \
        --output_json results/macro_relation_report_transfer_2025_adaptive_g05.json
fi

# Relative guidance
if [ -f "results/synthetic/transfer_2025_macro_sparse_relative_g05_uniform5k_physical.csv" ]; then
    echo "[eval] Relative guidance (scale=0.5)"
    conda run -n crashgen python pipeline/evaluate_macro_relations.py \
        --real "$REAL" \
        --synthetic results/synthetic/transfer_2025_macro_sparse_relative_g05_uniform5k_physical.csv \
        --output_json results/macro_relation_report_transfer_2025_relative_g05.json
fi

# Annealed guidance
if [ -f "results/synthetic/transfer_2025_macro_sparse_annealed_g05_uniform5k_physical.csv" ]; then
    echo "[eval] Annealed guidance (scale=0.5)"
    conda run -n crashgen python pipeline/evaluate_macro_relations.py \
        --real "$REAL" \
        --synthetic results/synthetic/transfer_2025_macro_sparse_annealed_g05_uniform5k_physical.csv \
        --output_json results/macro_relation_report_transfer_2025_annealed_g05.json
fi

echo "============================================"
echo "Macro relation evaluation complete!"
echo "============================================"
