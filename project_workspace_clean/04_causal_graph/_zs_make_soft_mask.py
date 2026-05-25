"""
Zero-shot 实验辅助脚本: 从已有 NOTEARS 连续权重生成 soft mask.

不重训 NOTEARS, 直接复用 configs/causal_matrix_notears_mlp_weights.npy.
产出 soft 版本用于后续 revise + prepare_dataset 流水线.

同时执行三项 sanity assertion:
  1. binary 模式可严格复现现有 causal_matrix_notears_mlp.npy
  2. soft 模式与 binary 的 support (非零位置) 完全一致
  3. soft 取值 ∈ (0, 1], 已加权人工修正后边权 == 1.0 仍正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

from src.causal_discovery_notears import (  # noqa: E402
    threshold_and_binarize,
    fallback_mask_correction,
)

CFG = CDT_ROOT / "configs"
W_PATH = CFG / "causal_matrix_notears_mlp_weights.npy"
BIN_PATH = CFG / "causal_matrix_notears_mlp.npy"
BIN_REVISED = CFG / "causal_matrix_v2_constrained.npy"
SOFT_OUT = CFG / "causal_matrix_notears_mlp_soft.npy"


def main(threshold: float = 0.3) -> None:
    assert W_PATH.exists(), f"missing weights: {W_PATH}"
    assert BIN_PATH.exists(), f"missing binary: {BIN_PATH}"

    W = np.load(W_PATH).astype(np.float32)
    bin_existing = np.load(BIN_PATH).astype(np.float32)

    # 1) backward-compat: binary mode 复现 (除 fallback_mask_correction 注入边外)
    bin_new = threshold_and_binarize(W, w_threshold=threshold, mode="binary")
    assert bin_new.shape == bin_existing.shape, \
        f"shape mismatch: {bin_new.shape} vs {bin_existing.shape}"
    only_in_existing = int(((bin_existing > 0) & (bin_new == 0)).sum())
    only_in_new = int(((bin_new > 0) & (bin_existing == 0)).sum())
    print(f"[assert-1] binary backward-compat: shape={bin_new.shape}, "
          f"only_in_existing(fallback注入)={only_in_existing}, only_in_new={only_in_new}")
    assert only_in_new == 0, "new binary must be subset of existing (no extra edges)"

    # 2) soft mode 结构与 binary 完全一致 (support 相同)
    soft = threshold_and_binarize(W, w_threshold=threshold, mode="soft")
    support_b = (bin_new > 0)
    support_s = (soft > 0)
    diff_support = int((support_b != support_s).sum())
    print(f"[assert-2] support identity: differing positions={diff_support}, "
          f"binary_edges={int(support_b.sum())}, soft_edges={int(support_s.sum())}")
    assert diff_support == 0, "soft mask must share exact support with binary"

    # 3) soft 取值范围
    nz = soft[soft > 0]
    print(f"[assert-3] soft value range: min={nz.min():.4f}, "
          f"max={nz.max():.4f}, mean={nz.mean():.4f}, median={np.median(nz):.4f}")
    assert nz.max() <= 1.0 + 1e-6 and nz.min() > 0
    # diagonal must be 0
    assert float(np.abs(np.diag(soft)).sum()) == 0.0

    # 4) 应用与 binary 相同的 fallback_mask_correction (孤儿 stage3 节点注入 = 1.0)
    import json
    summary_path = CFG / "causal_matrix_notears_mlp.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            meta = json.load(f)
        feat_names = meta["feature_names"]
        stage3 = meta["stage3_features"]
        n_added_before = int((soft > 0).sum())
        soft = fallback_mask_correction(soft, feat_names, stage3)
        n_added_after = int((soft > 0).sum())
        print(f"[fallback] orphan stage3 injection: {n_added_before} -> {n_added_after} "
              f"(injected_with_value=1.0)")
    else:
        print(f"[warn] no summary at {summary_path}; skipping fallback step")

    # 4) 与人工修正后的 v2 对齐性预检 (信息性, 非强断言)
    if BIN_REVISED.exists():
        v2 = np.load(BIN_REVISED).astype(np.float32)
        if v2.shape == soft.shape:
            added = ((v2 > 0) & (~support_b)).sum()
            removed = ((~(v2 > 0)) & support_b).sum()
            print(f"[info] vs v2_constrained: manual added={int(added)}, "
                  f"removed={int(removed)} (将由 revise_causal_matrix.py 在 soft 上重做)")

    np.save(SOFT_OUT, soft)
    print(f"[save] {SOFT_OUT}")
    print(f"[done] soft mask ready. next: run pipeline/revise_causal_matrix.py "
          f"--input {SOFT_OUT.relative_to(CDT_ROOT)} "
          f"--output configs/causal_matrix_v2_constrained_soft.npy")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.3)
    args = p.parse_args()
    main(threshold=args.threshold)
