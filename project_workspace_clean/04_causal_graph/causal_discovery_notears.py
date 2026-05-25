"""
Hierarchical CausalDiffTab - NOTEARS-MLP 因果发现 (GPU 加速 + 先验掩码)
========================================================================
基于 NOTEARS-MLP (Zheng et al. 2020) 的非线性因果发现，全程 CUDA 加速。

核心特性:
  1. NOTEARS-MLP: 用 MLP 替代线性模型，捕捉非线性因果关系
  2. Prior Mask (背景知识约束):
     - 禁止互斥二元指标间的因果边 (如 is_suv ↛ is_sedan)
     - 强制时序方向: Stage 3 变量禁止反向指向 Stage 1/2 变量
  3. 增强 Lagrangian 优化 + DAG 约束 h(W) = tr(e^{W∘W}) - d

输出: configs/causal_matrix_notears_mlp.npy  (N×N 二元邻接矩阵)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. NOTEARS-MLP 网络定义
# ============================================================

class NotearsMLP(nn.Module):
    """
    NOTEARS-MLP: 对每个变量 j 学习一个 MLP f_j(x) 来拟合 x_j。
    邻接矩阵 A[i,j] = ||W1[j,i,:]||_F 表示变量 i 对变量 j 的因果影响强度。

    参数:
      W1: (d, d, hidden) - W1[j, i, k] 是变量 i 到 MLP_j 隐层单元 k 的权重
      W2: (d, hidden, 1) - MLP_j 隐层到输出的权重
    """

    def __init__(self, d: int, hidden_dim: int = 32, device: str = "cuda"):
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim

        self.W1 = nn.Parameter(torch.randn(d, d, hidden_dim, device=device) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(d, hidden_dim, device=device))
        self.W2 = nn.Parameter(torch.randn(d, hidden_dim, 1, device=device) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(d, 1, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n, d) -> x_hat: (n, d)"""
        # h[n,j,k] = sigmoid( sum_i x[n,i] * W1[j,i,k] + b1[j,k] )
        h = torch.sigmoid(torch.einsum("ni,jik->njk", x, self.W1) + self.b1)
        # out[n,j] = sum_k h[n,j,k] * W2[j,k,0] + b2[j,0]
        out = torch.einsum("njk,jkl->njl", h, self.W2).squeeze(-1) + self.b2.squeeze(-1)
        return out

    def get_w_adj(self) -> torch.Tensor:
        """提取邻接矩阵: A[i,j] = ||W1[j,i,:]||_2 (i->j 的影响强度)"""
        return torch.sqrt((self.W1 ** 2).sum(dim=2) + 1e-20).T  # (d, d)

    def apply_prior_mask(self, mask: torch.Tensor):
        """将先验掩码投影到 W1: mask[i,j]=0 => 禁止 i->j 边"""
        with torch.no_grad():
            # mask[i,j] -> 需要作用于 W1[j,i,:], 即 mask.T[j,i]
            self.W1.data *= mask.T.unsqueeze(-1)


# ============================================================
# 2. DAG 约束与 NOTEARS 优化
# ============================================================

def h_dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """DAG 约束: h(W) = tr(exp(W∘W)) - d = 0"""
    d = W.shape[0]
    M = W * W
    E = torch.matrix_exp(M)
    return torch.trace(E) - d


def notears_mlp_train(
    model: NotearsMLP,
    X: torch.Tensor,
    prior_mask: torch.Tensor,
    lambda1: float = 0.02,
    lr: float = 0.003,
    max_outer: int = 30,
    max_inner: int = 1500,
    h_tol: float = 1e-8,
    rho_init: float = 1.0,
    rho_max: float = 1e16,
    rho_factor: float = 10.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    NOTEARS-MLP 增广 Lagrangian 优化。

    min  (1/n)||X - f(X)||^2 + λ₁||W||₁
    s.t. h(W) = 0   (DAG 约束)
         W ∘ (1-mask) = 0  (先验掩码)
    """
    n, d = X.shape
    alpha = 0.0
    rho = rho_init
    prev_h = float("inf")
    W_final = None

    for outer in range(max_outer):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for inner in range(max_inner):
            X_hat = model(X)
            mse = ((X - X_hat) ** 2).mean()

            W = model.get_w_adj()
            l1_penalty = lambda1 * W.abs().sum() / (d * d)
            h_val = h_dag_constraint(W)

            loss = mse + l1_penalty + alpha * h_val + 0.5 * rho * h_val * h_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.apply_prior_mask(prior_mask)

        with torch.no_grad():
            W = model.get_w_adj()
            h_val = h_dag_constraint(W).item()

        if verbose:
            with torch.no_grad():
                X_hat = model(X)
                mse_val = ((X - X_hat) ** 2).mean().item()
            print(f"  [outer={outer:2d}] h={h_val:.3e}, rho={rho:.1e}, "
                  f"MSE={mse_val:.4f}, |W|_1={W.abs().sum().item():.1f}")

        if h_val < h_tol:
            if verbose:
                print(f"  [OK] DAG constraint satisfied (h={h_val:.2e})")
            break

        if h_val > 0.25 * prev_h:
            rho = min(rho * rho_factor, rho_max)

        alpha += rho * h_val
        prev_h = h_val

    with torch.no_grad():
        W_final = model.get_w_adj().cpu().numpy()

    return W_final


# ============================================================
# 3. 先验掩码构建 (Background Knowledge)
# ============================================================

def build_prior_mask(
    feature_names: List[str],
    stage1_features: List[str],
    stage2_features: List[str],
    stage3_features: List[str],
    mutual_exclusion_groups: List[List[str]],
) -> np.ndarray:
    """
    构建先验掩码矩阵 M (d×d), M[i,j]=0 表示禁止 i→j 边。

    约束规则:
      1. 禁止自环: M[i,i] = 0
      2. 禁止互斥指标间连边: 同组内所有 (i,j) 对 M[i,j]=M[j,i]=0
      3. 强制时序方向: Stage 3 变量禁止指向 Stage 1/2 变量
    """
    d = len(feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    mask = np.ones((d, d), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)

    # 约束 1: 互斥指标组内禁止连边
    n_mutual = 0
    for group in mutual_exclusion_groups:
        indices = [name_to_idx[f] for f in group if f in name_to_idx]
        for i in indices:
            for j in indices:
                if i != j:
                    mask[i, j] = 0.0
                    n_mutual += 1

    # 约束 2: Stage 3 → Stage 1/2 方向禁止
    s3_idx = [name_to_idx[f] for f in stage3_features if f in name_to_idx]
    s12_idx = [name_to_idx[f] for f in (stage1_features + stage2_features)
               if f in name_to_idx]
    n_temporal = 0
    for i in s3_idx:
        for j in s12_idx:
            mask[i, j] = 0.0
            n_temporal += 1

    total_forbidden = int((mask == 0).sum()) - d  # 减去对角线
    total_allowed = int(mask.sum())
    print(f"[prior_mask] d={d}, forbidden={total_forbidden} "
          f"(mutual={n_mutual}, temporal={n_temporal}), "
          f"allowed={total_allowed}/{d*(d-1)}")

    return mask


# ============================================================
# 4. 数据加载
# ============================================================

def load_preprocessed_data(
    processed_dir: str,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    从预处理输出目录加载数据。
    返回 (X, feature_names, column_groups)
    """
    npy_dir = os.path.join(processed_dir, "npy")
    groups_json = os.path.join(processed_dir, "column_groups.json")

    if os.path.exists(groups_json):
        with open(groups_json, "r", encoding="utf-8") as f:
            groups = json.load(f)
    else:
        groups = {}

    if os.path.exists(os.path.join(npy_dir, "X_num_train.npy")):
        X_num = np.load(os.path.join(npy_dir, "X_num_train.npy"))
        X_cat = np.load(os.path.join(npy_dir, "X_cat_train.npy"))
        X = np.hstack([X_num, X_cat.astype(np.float32)])

        with open(os.path.join(npy_dir, "info.json"), "r") as f:
            info = json.load(f)
        feature_names = info["num_col_names"] + info["cat_col_names"]

        print(f"[load] From npy: X_num={X_num.shape}, X_cat={X_cat.shape}")
        return X, feature_names, groups

    csv_path = os.path.join(processed_dir, "processed_hierarchical.csv")
    if os.path.exists(csv_path) and groups:
        df = pd.read_csv(csv_path, low_memory=False)
        all_cols = groups["continuous_cols"] + groups["categorical_cols"]
        valid_cols = [c for c in all_cols if c in df.columns]

        for col in valid_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[valid_cols] = df[valid_cols].fillna(0)
        X = df[valid_cols].values.astype(np.float32)

        print(f"[load] From CSV: {X.shape}")
        return X, valid_cols, groups

    raise FileNotFoundError(
        f"No preprocessed data found in {processed_dir}. "
        "Run data_processor.py first."
    )


# ============================================================
# 5. 动态剪枝与物理常识兜底
# ============================================================

def dynamic_pruning(
    W: np.ndarray,
    target_density: float = 0.08,
    lo: float = 0.01,
    hi: float = 0.50,
    tol: float = 0.005,
    max_iter: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    二分查找最优剪枝阈值，使剪枝后边密度最接近 target_density。

    返回 (W_pruned, best_threshold)
    """
    d = W.shape[0]
    max_edges = d * (d - 1)
    if max_edges == 0:
        return W.copy(), 0.0

    def density_at(thresh):
        W_t = W.copy()
        W_t[np.abs(W_t) < thresh] = 0.0
        np.fill_diagonal(W_t, 0.0)
        return (np.abs(W_t) > 1e-12).sum() / max_edges

    best_thresh = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        d_mid = density_at(mid)
        if abs(d_mid - target_density) < tol:
            best_thresh = mid
            break
        if d_mid > target_density:
            lo = mid
        else:
            hi = mid
        best_thresh = mid

    W_pruned = W.copy()
    W_pruned[np.abs(W_pruned) < best_thresh] = 0.0
    np.fill_diagonal(W_pruned, 0.0)
    actual_density = (np.abs(W_pruned) > 1e-12).sum() / max_edges

    print(f"[dynamic_prune] target_density={target_density:.4f}, "
          f"found threshold={best_thresh:.4f}, actual_density={actual_density:.4f}")

    return W_pruned, best_thresh


def fallback_mask_correction(
    W_bin: np.ndarray,
    feature_names: List[str],
    stage3_features: List[str],
    anchor_features: List[str] = None,
) -> np.ndarray:
    """
    物理常识兜底: 检查 Stage 3 节点是否存在入度为 0 的孤儿节点。
    若有，强制将时空锚点 (LATITUDE, LONGITUDE, TIME_PERIOD) 连向该节点。

    Args:
        W_bin: 二值化邻接矩阵 (d x d), W_bin[i,j]=1 表示 i->j
        feature_names: 完整特征名列表
        stage3_features: Stage 3 特征名列表
        anchor_features: 强制注入的锚点特征名 (默认 LAT, LON, TIME_PERIOD)
    """
    if anchor_features is None:
        anchor_features = ["LATITUDE", "LONGITUDE", "TIME_PERIOD"]

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    anchor_idx = [name_to_idx[f] for f in anchor_features if f in name_to_idx]

    n_fixed = 0
    orphans = []
    for feat in stage3_features:
        if feat not in name_to_idx:
            continue
        j = name_to_idx[feat]
        in_degree = W_bin[:, j].sum()
        if in_degree == 0:
            orphans.append(feat)
            for ai in anchor_idx:
                W_bin[ai, j] = 1.0
            n_fixed += 1

    if n_fixed > 0:
        print(f"[fallback] Fixed {n_fixed} orphan Stage 3 nodes: {orphans}")
        print(f"  -> Injected edges from {anchor_features} to each orphan")
    else:
        print(f"[fallback] All Stage 3 nodes have parents, no correction needed")

    return W_bin


# ============================================================
# 6. DAG 后处理与摘要
# ============================================================

def threshold_and_binarize(
    W: np.ndarray,
    w_threshold: float = 0.3,
    mode: str = "binary",
) -> np.ndarray:
    """将连续权重矩阵阈值化。

    mode='binary' (默认, 向后兼容): (|W| > thr) -> {0, 1}
    mode='soft' : 保留 |W|, 低于 thr 的边置 0, 再归一化到 [0, 1]
                  (结构与 binary 相同, 只是把 0/1 硬约束换成连续强度)
    """
    A = np.abs(W).astype(np.float32)
    if mode == "binary":
        out = (A > w_threshold).astype(np.float32)
    elif mode == "soft":
        out = np.where(A > w_threshold, A, 0.0).astype(np.float32)
        m = float(out.max())
        if m > 0:
            out = out / m
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    np.fill_diagonal(out, 0.0)
    return out


def summarize_dag(W_bin: np.ndarray, feature_names: List[str]):
    """打印 DAG 摘要信息"""
    d = W_bin.shape[0]
    n_edges = int(W_bin.sum())
    density = n_edges / (d * (d - 1)) if d > 1 else 0

    root_nodes = []
    for j in range(d):
        if W_bin[:, j].sum() == 0:
            root_nodes.append(feature_names[j])

    print(f"\n--- DAG Summary ---")
    print(f"  Nodes: {d}, Edges: {n_edges}, Density: {density:.4f}")
    print(f"  Root nodes (no parents): {root_nodes[:15]}")

    edges = []
    for i in range(d):
        for j in range(d):
            if W_bin[i, j] > 0:
                edges.append((feature_names[i], feature_names[j]))

    print(f"  Top edges (showing {min(30, len(edges))}/{len(edges)}):")
    for src, tgt in edges[:30]:
        print(f"    {src} -> {tgt}")

    return edges


# ============================================================
# 7. 主流程
# ============================================================

def run_causal_discovery(
    processed_dir: str,
    output_npy: str,
    hidden_dim: int = 32,
    lambda1: float = 0.02,
    lr: float = 0.003,
    w_threshold: float = 0.3,
    target_density: float = 0.08,
    max_samples: int = 0,
    max_outer: int = 30,
    max_inner: int = 1500,
    device: str = "cuda",
    seed: int = 42,
    mask_mode: str = "binary",
) -> np.ndarray:
    """完整的 NOTEARS-MLP 因果发现流程

    mask_mode: 'binary' (默认) 输出 0/1 硬掩码; 'soft' 输出 [0,1] 软掩码
    (结构与 binary 一致, 仅把硬约束改为连续强度, 用于 zero-shot 稳健性实验)
    """

    print("=" * 60)
    print("Hierarchical CausalDiffTab - NOTEARS-MLP Causal Discovery")
    print(f"  Device: {device}")
    print("=" * 60)

    # ---- 加载数据 ----
    X, feature_names, groups = load_preprocessed_data(processed_dir)
    d = X.shape[1]

    # ---- 子采样 ----
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        print(f"[subsample] -> {X.shape[0]} rows")

    # ---- 标准化 ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    n_nan = np.isnan(X_scaled).sum()
    if n_nan > 0:
        print(f"[warn] {n_nan} NaNs after scaling, filling with 0")
        X_scaled = np.nan_to_num(X_scaled, 0.0)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    # ---- 构建先验掩码 ----
    stage1 = groups.get("stage1_features", [])
    stage2 = groups.get("stage2_features", [])
    stage3 = groups.get("stage3_features", [])
    vehicle_binary = groups.get("vehicle_binary", [])

    mutual_exclusion_groups = []
    if vehicle_binary:
        mutual_exclusion_groups.append(vehicle_binary)

    prior_mask_np = build_prior_mask(
        feature_names, stage1, stage2, stage3, mutual_exclusion_groups
    )
    prior_mask = torch.tensor(prior_mask_np, dtype=torch.float32, device=device)

    # ---- 构建 NOTEARS-MLP ----
    torch.manual_seed(seed)
    model = NotearsMLP(d, hidden_dim=hidden_dim, device=device)

    print(f"\n[NOTEARS-MLP] d={d}, hidden={hidden_dim}, n={X_tensor.shape[0]}")
    print(f"  lambda1={lambda1}, lr={lr}, threshold={w_threshold}")

    t0 = time.time()
    W = notears_mlp_train(
        model, X_tensor, prior_mask,
        lambda1=lambda1, lr=lr,
        max_outer=max_outer, max_inner=max_inner,
    )
    elapsed = time.time() - t0
    print(f"\n[time] NOTEARS-MLP completed in {elapsed:.1f}s")

    # ---- 动态稀疏剪枝: 二分搜索最优阈值逼近目标密度 ----
    n_before = int((np.abs(W) > 0.01).sum()) - d
    W, dyn_thresh = dynamic_pruning(W, target_density=target_density)
    n_after = int((np.abs(W) > 1e-12).sum()) - d
    print(f"[prune] Dynamic pruning: {n_before} -> {n_after} non-trivial entries "
          f"(threshold={dyn_thresh:.4f})")

    # ---- 阈值化 + 保存 ----
    W_bin = threshold_and_binarize(W, w_threshold, mode=mask_mode)
    print(f"[mask] mode={mask_mode}, nonzero={int((W_bin > 0).sum())}, "
          f"max={float(W_bin.max()):.4f}, mean(nz)={float(W_bin[W_bin>0].mean()) if (W_bin>0).any() else 0:.4f}")

    # ---- 物理常识兜底: 孤儿 Stage 3 节点强制注入时空锚点 ----
    W_bin = fallback_mask_correction(W_bin, feature_names, stage3)

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, W_bin)
    print(f"[save] Binary adjacency -> {output_npy}")

    weight_path = output_npy.replace(".npy", "_weights.npy")
    np.save(weight_path, W)
    print(f"[save] Weight matrix -> {weight_path}")

    edges = summarize_dag(W_bin, feature_names)

    summary = {
        "algorithm": "NOTEARS-MLP",
        "device": device,
        "d": d,
        "n_samples": int(X_tensor.shape[0]),
        "hidden_dim": hidden_dim,
        "lambda1": lambda1,
        "w_threshold": w_threshold,
        "mask_mode": mask_mode,
        "target_density": target_density,
        "dynamic_prune_threshold": round(float(dyn_thresh), 4),
        "n_edges": int(W_bin.sum()),
        "density": float(W_bin.sum() / (d * (d - 1))) if d > 1 else 0,
        "elapsed_seconds": round(elapsed, 1),
        "feature_names": feature_names,
        "stage1_features": stage1,
        "stage2_features": stage2,
        "stage3_features": stage3,
        "mutual_exclusion_groups": mutual_exclusion_groups,
        "edges": [{"src": s, "tgt": t} for s, t in edges],
    }
    summary_path = output_npy.replace(".npy", ".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[save] Summary -> {summary_path}")

    return W_bin


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOTEARS-MLP Causal Discovery (GPU + Prior Mask)"
    )
    parser.add_argument(
        "--processed_dir", type=str, required=True,
        help="Directory containing processed data (npy/ + column_groups.json)",
    )
    parser.add_argument(
        "--output_npy", type=str, default=None,
        help="Output path for binary adjacency matrix",
    )
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--w_threshold", type=float, default=0.3)
    parser.add_argument("--target_density", type=float, default=0.08,
                        help="Target edge density for dynamic pruning (e.g. 0.08 = 8%%)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = use all data; >0 = subsample to this many rows")
    parser.add_argument("--max_outer", type=int, default=30)
    parser.add_argument("--max_inner", type=int, default=1500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_mode", type=str, default="binary",
                        choices=["binary", "soft"],
                        help="binary=0/1 hard mask (legacy); soft=[0,1] continuous strength")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.output_npy is None:
        cdt_root = Path(__file__).resolve().parent.parent
        args.output_npy = str(cdt_root / "configs" / "causal_matrix_notears_mlp.npy")

    run_causal_discovery(
        processed_dir=args.processed_dir,
        output_npy=args.output_npy,
        hidden_dim=args.hidden_dim,
        lambda1=args.lambda1,
        lr=args.lr,
        w_threshold=args.w_threshold,
        target_density=args.target_density,
        max_samples=args.max_samples,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        device=args.device,
        seed=args.seed,
        mask_mode=args.mask_mode,
    )


if __name__ == "__main__":
    main()
