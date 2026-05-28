import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tabdiff')
from src.sample_conditional import run_sampling

run_sampling(
    ckpt_dir="ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2_ft2025",
    data_dir="data/nyc_crash_2025_v2",
    condition_train_indices="results/_cache/eval_uniform_5k.txt",
    num_samples=5000, batch_size=500, device="cuda:0",
    output_csv="results/synthetic/transfer_2025_macro_sparse_anneal_v2_ft2025_uniform5k.csv",
    macro_guidance_scale=0.0,
)
