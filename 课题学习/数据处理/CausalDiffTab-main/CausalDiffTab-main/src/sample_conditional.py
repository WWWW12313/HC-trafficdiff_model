"""
Hierarchical CausalDiffTab - 完整采样器
========================================
支持两种模式:
  1. unconditional: 直接从 Stage 3 模型生成全量特征 (用于评估)
  2. conditional:   DDPM Inpainting, 锁定 Stage 1/2 条件, 只生成 Stage 3

Usage:
  # 无条件采样 (评估用)
  python src/sample_conditional.py \
    --ckpt_dir ckpt/nyc_crash/stage3_full_full \
    --num_samples 5000 --device cuda:0

  # 条件采样：训练集行索引（张量空间与 TabDiffDataset 一致），重采样 y + Stage3 分类
  python src/sample_conditional.py \
    --ckpt_dir ckpt/nyc_crash/stage3_full_full \
    --condition_train_indices "0,1,2" \
    --num_samples 100 --device cuda:0

  # 外部 CSV 条件：需与 TabDiff 前向编码对齐后方可接入；当前仍会回退无条件并提示
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

import src as cdt_src
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from utils_train import TabDiffDataset


# ============================================================
# 1. 模型加载
# ============================================================

def load_model(
    ckpt_dir: str,
    data_dir: str,
    device: str = "cuda:0",
    ckpt_name: str = None,
) -> tuple:
    """
    从 checkpoint 目录加载训练好的扩散模型。

    Returns:
        diffusion: UnifiedCtimeDiffusion
        dataset: TabDiffDataset (用于 inverse transform)
        info: dict
    """
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        raw_config = pickle.load(f)

    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    dataset = TabDiffDataset(
        os.path.basename(data_dir), data_dir, info,
        y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    d_numerical = dataset.d_numerical
    categories = dataset.categories

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (
        (categories + 1).tolist() if len(categories) > 0 else []
    )

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
        causal_weight_max=1.0,
        causal_warmup_steps=1,
    )
    diffusion.to(device)

    if ckpt_name is None:
        pt_files = [f for f in os.listdir(ckpt_dir)
                     if f.startswith("best_model_") and f.endswith(".pt")]
        if pt_files:
            ckpt_name = sorted(pt_files)[0]
        else:
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            ckpt_name = sorted(pt_files)[-1] if pt_files else None

    if ckpt_name is None:
        raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(f"[load] Loading checkpoint: {ckpt_name}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])

    diffusion.eval()
    print(f"[load] Model loaded: d_numerical={d_numerical}, "
          f"categories={list(categories)[:5]}... ({len(categories)} total)")

    return diffusion, dataset, info


# ============================================================
# 2. 无条件采样
# ============================================================

def sample_unconditional(
    diffusion: UnifiedCtimeDiffusion,
    dataset: TabDiffDataset,
    info: dict,
    num_samples: int = 5000,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    无条件采样: 使用模型直接生成全量特征, 经 inverse transform 得到 DataFrame。
    """
    print(f"[sample] Generating {num_samples} samples (batch={batch_size})...")

    all_samples = []
    remaining = num_samples

    while remaining > 0:
        b = min(remaining, batch_size)
        syn_tensor = diffusion.sample(b)
        all_samples.append(syn_tensor)
        remaining -= b
        print(f"  generated {num_samples - remaining}/{num_samples}")

    syn_data = torch.cat(all_samples, dim=0)[:num_samples]

    syn_df = tensor_to_dataframe(syn_data, dataset, info)
    return syn_df


def tensor_to_dataframe(
    syn_data: torch.Tensor,
    dataset: TabDiffDataset,
    info: dict,
) -> pd.DataFrame:
    """
    将模型输出张量转换为 DataFrame, 应用 inverse transforms。
    复刻 Trainer.sample_synthetic 的逻辑。
    """
    from tabdiff.trainer import split_num_cat_target, recover_data

    num_inverse = dataset.num_inverse
    int_inverse = dataset.int_inverse
    cat_inverse = dataset.cat_inverse

    arr = syn_data.detach().float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        arr, info, num_inverse, int_inverse, cat_inverse,
    )

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(k): v for k, v in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    return syn_df


# ============================================================
# 3. 条件 Inpainting 采样
# ============================================================

def build_stage_indices(info: dict, column_groups: dict) -> Dict[str, List[int]]:
    """
    构建各 Stage 特征在模型内部张量中的列索引。
    regression 模式下 y prepend 到 num 部分, 故 num 有 +1 offset。
    """
    num_col_names = info["num_col_names"]
    cat_col_names = info["cat_col_names"]
    cat_sizes = info["cat_sizes"]

    s1_cont = set(column_groups.get("stage1_continuous", []))
    s1_cat = set(column_groups.get("stage1_categorical", []))
    s2_cont = set(column_groups.get("stage2_continuous", []))
    s2_cat = set(column_groups.get("stage2_categorical", []))

    is_regression = info.get("task_type", "regression") == "regression"
    y_offset = 1 if is_regression else 0

    def num_indices(target_set):
        return [i + y_offset for i, c in enumerate(num_col_names) if c in target_set]

    cat_offsets = []
    cur = 0
    for s in cat_sizes:
        cat_offsets.append(cur)
        cur += s + 1

    def cat_indices(target_set):
        idx = []
        for i, c in enumerate(cat_col_names):
            if c in target_set:
                start = cat_offsets[i]
                end = start + cat_sizes[i] + 1
                idx.extend(range(start, end))
        return idx

    cond_cont = s1_cont | s2_cont
    cond_cat = s1_cat | s2_cat

    return {
        "cond_num_idx": num_indices(cond_cont),
        "free_num_idx": num_indices(set(num_col_names) - cond_cont),
        "cond_cat_idx": cat_indices(cond_cat),
        "free_cat_idx": cat_indices(set(cat_col_names) - cond_cat),
    }


def build_stage3_impute_masks(info: dict, column_groups: dict):
    """
    用于 diffusion.sample_impute：num_mask_idx / cat_mask_idx 为需要 **重新生成** 的维度。
    与 make_dataset(regression) 一致：X_num 第 0 列为目标 y（经 TabDiff 侧 Quantile），
    其余列为 num_col_names 顺序的连续特征。
    """
    cat_col_names = info["cat_col_names"]
    s3 = set(column_groups.get("stage3_categorical", []))
    cat_mask_idx = [i for i, c in enumerate(cat_col_names) if c in s3]
    num_mask_idx = [0]
    return num_mask_idx, cat_mask_idx


def _reset_impute_state(diffusion: UnifiedCtimeDiffusion) -> None:
    diffusion.w_num = 0.0
    diffusion.w_cat = 0.0
    diffusion.num_mask_idx = []
    diffusion.cat_mask_idx = []


def parse_train_indices(spec: str) -> List[int]:
    """逗号分隔，或指向每行一个整数的文本文件。"""
    p = Path(spec)
    if p.is_file():
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [int(x.strip()) for x in lines if x.strip()]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def sample_impute_stage3(
    diffusion: UnifiedCtimeDiffusion,
    dataset: TabDiffDataset,
    info: dict,
    column_groups: dict,
    train_row_indices: List[int],
    resample_rounds: int = 1,
    impute_condition: str = "x_0",
    w_num: float = 0.0,
    w_cat: float = 0.0,
) -> torch.Tensor:
    """
    以训练集中指定行的 Stage1+2 为条件，仅对目标 y 与 Stage3 分类列做 sample_impute。
    条件行必须与 TabDiffDataset 内部张量一致（已含 TabDiff Quantile + Ordinal 编码）。
    """
    device = diffusion.device
    d_n = dataset.d_numerical
    n = len(dataset)
    idx = [i % n for i in train_row_indices]
    X = dataset.X[idx].to(device).float()
    x_num = X[:, :d_n].clone()
    x_cat = X[:, d_n:].long()

    num_mask_idx, cat_mask_idx = build_stage3_impute_masks(info, column_groups)

    x_train_num = dataset.X[:, :d_n].float()
    avg_m = x_train_num[:, num_mask_idx].mean(dim=0).to(device)
    for k, ji in enumerate(num_mask_idx):
        x_num[:, ji] = avg_m[k]

    mi = diffusion.mask_index
    for j in cat_mask_idx:
        x_cat[:, j] = mi[j]

    try:
        out = diffusion.sample_impute(
            x_num,
            x_cat,
            num_mask_idx,
            cat_mask_idx,
            resample_rounds,
            impute_condition,
            w_num,
            w_cat,
        )
    finally:
        _reset_impute_state(diffusion)
    return out


@torch.no_grad()
def sample_conditional(
    diffusion: UnifiedCtimeDiffusion,
    condition_x0_num: torch.Tensor,
    cond_num_idx: List[int],
    free_num_idx: List[int],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    DDPM Inpainting 条件采样:
    反向去噪过程中, 每步对条件列 (Stage 1/2) 做前向加噪替换,
    仅让 Stage 3 列自由去噪。
    """
    b = condition_x0_num.shape[0]
    dtype = torch.float32
    num_timesteps = diffusion.num_timesteps

    t = torch.linspace(0, 1, num_timesteps, dtype=dtype, device=device)
    t = t[:, None]

    sigma_num_cur = diffusion.num_schedule.total_noise(t)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(t)
    sigma_num_next = torch.zeros_like(sigma_num_cur)
    sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur)
    sigma_cat_next[1:] = sigma_cat_cur[:-1]

    t_hat = t
    sigma_num_hat = sigma_num_cur
    sigma_cat_hat = sigma_cat_cur

    z_norm = torch.randn(
        (b, diffusion.num_numerical_features), device=device
    ) * sigma_num_cur[-1]

    has_cat = len(diffusion.num_classes) > 0
    z_cat = torch.zeros((b, 0), device=device).float()
    if has_cat:
        z_cat = diffusion._sample_masked_prior(b, len(diffusion.num_classes))

    pbar = tqdm(reversed(range(num_timesteps)), total=num_timesteps,
                desc="Conditional Inpainting")

    for i in pbar:
        z_norm, z_cat, q_xs = diffusion.edm_update(
            z_norm, z_cat, i,
            t[i], t[i - 1] if i > 0 else None, t_hat[i],
            sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i],
            sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
        )

        if i > 0 and cond_num_idx:
            sigma_cond = sigma_num_next[i]
            eps_num = torch.randn_like(condition_x0_num)
            x_cond_noisy = condition_x0_num + sigma_cond * eps_num
            z_norm[:, cond_num_idx] = x_cond_noisy[:, cond_num_idx]
        elif i == 0 and cond_num_idx:
            z_norm[:, cond_num_idx] = condition_x0_num[:, cond_num_idx]

    sample = torch.cat([z_norm, z_cat], dim=1).cpu()
    return sample


# ============================================================
# 4. 端到端管线
# ============================================================

def run_sampling(
    ckpt_dir: str,
    data_dir: str = None,
    condition_csv: str = None,
    condition_train_indices: str = None,
    num_samples: int = 5000,
    batch_size: int = 500,
    device: str = "cuda:0",
    output_csv: str = None,
    do_postprocess: bool = True,
    impute_resample_rounds: int = 1,
    impute_condition: str = "x_0",
    road_graphml: str = None,
    road_signals: str = None,
    snap_max_dist_m: float = 300.0,
    recompute_osm_after_snap: bool = True,
):
    """
    完整采样管线:
    - 无附加条件: 无条件 sample()
    - condition_train_indices: 训练集行号（与 TabDiffDataset 一致），对 y + Stage3 做 sample_impute
    - condition_csv: 预留；未与 TabDiff 前向编码对齐前仍回退无条件（见下方说明）
    - do_postprocess: 自动调用后处理还原物理值
    """
    data_dir = data_dir or str(CDT_ROOT / "data" / "nyc_crash")

    if condition_train_indices:
        mode_s = "impute_stage3 (train row indices)"
    elif condition_csv is not None:
        mode_s = "condition_csv -> fallback unconditional"
    else:
        mode_s = "unconditional"
    print("=" * 60)
    print("Hierarchical CausalDiffTab - Sampler")
    print(f"  Mode: {mode_s}")
    print(f"  Samples: {num_samples}")
    print(f"  Device: {device}")
    print("=" * 60)

    diffusion, dataset, info = load_model(ckpt_dir, data_dir, device)

    column_groups_json = str(CDT_ROOT / "data" / "processed" / "column_groups.json")
    with open(column_groups_json, "r", encoding="utf-8") as f:
        groups = json.load(f)

    if condition_train_indices:
        base_idx = parse_train_indices(condition_train_indices)
        if not base_idx:
            raise ValueError("condition_train_indices 解析结果为空")
        expanded = [base_idx[i % len(base_idx)] for i in range(num_samples)]
        print(f"[impute] Stage3 sample_impute, base indices={base_idx}, total rows={num_samples}, "
              f"batch_size={batch_size}")
        chunks = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            sub = expanded[start:end]
            t_out = sample_impute_stage3(
                diffusion,
                dataset,
                info,
                groups,
                sub,
                resample_rounds=impute_resample_rounds,
                impute_condition=impute_condition,
            )
            chunks.append(t_out)
        syn_tensor = torch.cat(chunks, dim=0)
        syn_df = tensor_to_dataframe(syn_tensor, dataset, info)
    elif condition_csv is not None:
        stage_idx = build_stage_indices(info, groups)
        print(f"[indices] cond_num: {len(stage_idx['cond_num_idx'])} dims, "
              f"free_num: {len(stage_idx['free_num_idx'])} dims")
        print(f"[indices] cond_cat: {len(stage_idx['cond_cat_idx'])} dims, "
              f"free_cat: {len(stage_idx['free_cat_idx'])} dims")
        print("[warn] condition_csv 尚未与 TabDiff 前向编码对齐，回退无条件 sample()。")
        syn_df = sample_unconditional(
            diffusion, dataset, info,
            num_samples=num_samples, batch_size=batch_size,
        )
    else:
        syn_df = sample_unconditional(
            diffusion, dataset, info,
            num_samples=num_samples, batch_size=batch_size,
        )

    if not output_csv:
        output_dir = os.path.join(str(CDT_ROOT), "result", "nyc_crash", "sampled")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "samples.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    syn_df.to_csv(output_csv, index=False)
    print(f"\n[saved] Raw samples: {output_csv} ({len(syn_df)} rows)")

    if do_postprocess:
        from src.postprocess_samples import postprocess
        base, ext = os.path.splitext(output_csv)
        physical_csv = f"{base}_physical{ext}"
        print(f"\n[postprocess] Restoring physical values...")
        # 用 data_dir/train.csv 作为分类映射参照（含新 schema 列名）
        ref_train_csv = os.path.join(data_dir, "train.csv") if data_dir and os.path.exists(os.path.join(data_dir, "train.csv")) else None
        postprocess(
            samples_csv=output_csv,
            output_csv=physical_csv,
            processed_csv=ref_train_csv,
            road_graphml=road_graphml,
            road_signals=road_signals,
            snap_max_dist_m=snap_max_dist_m,
            recompute_osm_after_snap=recompute_osm_after_snap,
        )

    return syn_df


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical CausalDiffTab - Complete Sampler"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--condition_csv", type=str, default=None)
    parser.add_argument(
        "--condition_train_indices",
        type=str,
        default=None,
        help="训练集行索引：逗号分隔，或 txt 文件每行一个整数；与 TabDiffDataset 张量一致",
    )
    parser.add_argument("--impute_resample_rounds", type=int, default=1)
    parser.add_argument("--impute_condition", type=str, default="x_0", choices=["x_0", "x_t"])
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--no_postprocess", action="store_true")

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_sampling(
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        condition_csv=args.condition_csv,
        condition_train_indices=args.condition_train_indices,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_csv=args.output_csv,
        do_postprocess=not args.no_postprocess,
        impute_resample_rounds=args.impute_resample_rounds,
        impute_condition=args.impute_condition,
    )


if __name__ == "__main__":
    main()
