"""
仅基于现有 info.json 把 soft 邻接矩阵展开为训练用 num/cat mask,
不重新切分训练/测试数据, 不动现有 binary mask 目录.

用法（默认使用 nyc_crash / nyc_stage1）:
  python pipeline/_zs_expand_soft_masks.py

指定 2024 数据集:
  python pipeline/_zs_expand_soft_masks.py --dataname nyc_crash_2024_v2

输出:
  data/{dataname}/causal_masks_soft/{num,cat}_causal_mask.npy
  data/{stage1_dataname}/causal_masks_soft/{num,cat}_causal_mask.npy
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

from src.prepare_dataset import (  # noqa: E402
    build_causal_mask_for_model,
    _align_causal_matrix_to_features,
    _apply_domain_causal_rules,
)

CFG = CDT_ROOT / "configs"
# 2026-05-16: 修复——使用新版宏观因果骨架（37 列），而非旧版 v2_constrained_soft（47 列）
SOFT_FULL = CFG / "causal_matrix_macro_soft.npy"


def expand_for_stage1(W_full: np.ndarray, info_full: dict, info_s1: dict,
                      groups: dict) -> tuple[np.ndarray, np.ndarray]:
    """仿照 prepare_dataset.run_prepare 中 Stage1 mask 的展开方式."""
    all_num = groups["continuous_cols"]
    all_cat = groups["categorical_cols"]
    all_features = all_num + all_cat
    s1_num = groups["stage1_continuous"]
    s1_cat = groups["stage1_categorical"]

    # num
    s1_num_idx = [all_features.index(f) for f in s1_num if f in all_features]
    d_with_y = len(s1_num) + 1
    W_s1 = W_full[np.ix_(s1_num_idx, s1_num_idx)]
    s1_num_mask = np.zeros((d_with_y, d_with_y), dtype=np.float32)
    s1_num_mask[0, 1:] = 1.0
    s1_num_mask[1:, 1:] = W_s1

    # cat (regression -> no y prepended)
    s1_cat_sizes = info_s1["cat_sizes"]
    s1_expanded = [s + 1 for s in s1_cat_sizes]
    s1_total = sum(s1_expanded)
    s1_cat_idx = [all_features.index(f) for f in s1_cat if f in all_features]
    n_s1_cat = len(s1_cat_idx)
    W_s1c = W_full[np.ix_(s1_cat_idx, s1_cat_idx)]
    s1_cat_mask = np.zeros((s1_total, s1_total), dtype=np.float32)
    offsets, cur = [], 0
    for s in s1_expanded:
        offsets.append(cur)
        cur += s
    for i in range(n_s1_cat):
        for j in range(n_s1_cat):
            v = float(W_s1c[i, j])
            if v > 0:
                si, sj = s1_expanded[i], s1_expanded[j]
                oi, oj = offsets[i], offsets[j]
                s1_cat_mask[oi:oi + si, oj:oj + sj] = v
    return s1_num_mask, s1_cat_mask


def main():
    parser = argparse.ArgumentParser(description="将宏观 soft 因果矩阵展开为 num/cat mask")
    parser.add_argument(
        "--dataname", type=str, default="nyc_crash",
        help="Stage3 数据目录名（位于 data/ 下），如 nyc_crash 或 nyc_crash_2024_v2",
    )
    parser.add_argument(
        "--stage1_dataname", type=str, default=None,
        help="Stage1 数据目录名；默认自动从 dataname 推导（nyc_crash_X → nyc_stage1_X）",
    )
    parser.add_argument(
        "--soft_matrix", type=str, default=str(SOFT_FULL),
        help="soft 邻接矩阵 .npy 路径（默认 configs/causal_matrix_macro_soft.npy）",
    )
    args = parser.parse_args()

    soft_path = Path(args.soft_matrix)
    assert soft_path.exists(), f"soft matrix not found: {soft_path}"

    # 推导 Stage1 数据目录
    dataname = args.dataname
    if args.stage1_dataname:
        stage1_name = args.stage1_dataname
    elif dataname.startswith("nyc_crash_"):
        stage1_name = dataname.replace("nyc_crash", "nyc_stage1", 1)
    else:
        stage1_name = "nyc_stage1"

    DATA_FULL = CDT_ROOT / "data" / dataname
    DATA_S1   = CDT_ROOT / "data" / stage1_name

    assert DATA_FULL.exists(), f"Stage3 data dir not found: {DATA_FULL}"
    assert DATA_S1.exists(),   f"Stage1 data dir not found: {DATA_S1}"

    assert SOFT_FULL.exists(), SOFT_FULL

    # --- 加载 column_groups 用于对齐 ---
    groups_path = CDT_ROOT / "data" / "processed" / "column_groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))

    # --- Stage 3 (full) ---
    info_full = json.loads((DATA_FULL / "info.json").read_text(encoding="utf-8"))
    out_full = DATA_FULL / "causal_masks_soft"
    out_full.mkdir(parents=True, exist_ok=True)
    num_mask, cat_mask = build_causal_mask_for_model(
        str(soft_path), info_full, str(out_full), task_type="regression",
        groups=groups,
    )
    print(f"[stage3] num: shape={num_mask.shape}, "
          f"nz={int((num_mask>0).sum())}, mean(nz)={num_mask[num_mask>0].mean():.4f}, "
          f"max={num_mask.max():.4f}")
    print(f"[stage3] cat: shape={cat_mask.shape}, "
          f"nz={int((cat_mask>0).sum())}, "
          f"mean(nz)={(cat_mask[cat_mask>0].mean() if (cat_mask>0).any() else 0):.4f}, "
          f"max={cat_mask.max():.4f}")

    # --- Stage 1 ---
    info_s1 = json.loads((DATA_S1 / "info.json").read_text(encoding="utf-8"))
    # 先对齐原始 37×37 矩阵 → 45×45，再传给 expand_for_stage1
    W_full_raw = np.load(soft_path).astype(np.float32)
    feature_names = groups["continuous_cols"] + groups["categorical_cols"]
    W_full = _align_causal_matrix_to_features(W_full_raw, feature_names)
    W_full = _apply_domain_causal_rules(W_full, feature_names, groups)
    s1_num, s1_cat = expand_for_stage1(W_full, info_full, info_s1, groups)
    out_s1 = DATA_S1 / "causal_masks_soft"
    out_s1.mkdir(parents=True, exist_ok=True)
    np.save(out_s1 / "num_causal_mask.npy", s1_num)
    np.save(out_s1 / "cat_causal_mask.npy", s1_cat)
    print(f"[stage1] num: shape={s1_num.shape}, nz={int((s1_num>0).sum())}")
    print(f"[stage1] cat: shape={s1_cat.shape}, nz={int((s1_cat>0).sum())}")

    # --- 与现有 binary 对比（仅在 binary mask 存在时比较）---
    bin_mask_path = DATA_FULL / "causal_masks" / "num_causal_mask.npy"
    if bin_mask_path.exists():
        bin_num = np.load(bin_mask_path)
        same_support = ((bin_num > 0) == (num_mask > 0)).all()
        print(f"[compare-stage3] num support identical to binary: {same_support}")
        print(f"[compare-stage3] binary edges={int((bin_num>0).sum())}, "
              f"soft edges={int((num_mask>0).sum())}")
    else:
        print(f"[compare-stage3] binary mask not found, skipping comparison")
    
    print(f"\n✓ Soft masks 已写入:")
    print(f"  Stage3: {out_full}")
    print(f"  Stage1: {out_s1}")


if __name__ == "__main__":
    main()
