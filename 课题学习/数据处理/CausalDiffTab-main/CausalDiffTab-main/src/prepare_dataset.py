"""
Hierarchical CausalDiffTab - 数据集格式转换
=============================================
将 data_processor.py 产出的预处理数据转换为 CausalDiffTab 原生的
X_num / X_cat / y npy 目录格式，同时为 Stage 1 (spatial) 和
Stage 3 (full) 分别创建独立的数据集目录。
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def build_dataset_dir(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    target_col: str,
    output_dir: str,
    task_type: str = "regression",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    将 DataFrame 按 CausalDiffTab 的目录格式写出。
    CausalDiffTab 的 make_dataset 会在加载时:
      - regression: 将 y prepend 到 X_num
      - classification: 将 y prepend 到 X_cat
    所以这里 **不** 手动拼接 y，只需分别存 X_num / X_cat / y。
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_num = [c for c in num_cols if c in df.columns]
    valid_cat = [c for c in cat_cols if c in df.columns]
    df_export = df.copy()
    missing_category = "__MISSING__"
    for col in valid_cat:
        df_export[col] = df_export[col].where(df_export[col].notna(), missing_category).astype(str)

    # ---- 数值特征 (不含 y) ----
    X_num_all = df_export[valid_num].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    # ---- 分类特征 → ordinal 编码 (整数索引) ----
    cat_encoded = df_export[valid_cat].copy()
    cat_categories = {}
    for col in valid_cat:
        cat_col = cat_encoded[col].astype("category")
        cat_encoded[col] = cat_col.cat.codes
        cat_categories[col] = int(cat_col.cat.categories.size)
    X_cat_all = cat_encoded.values.astype(np.int64)

    cat_label_mappings = {}
    for col in valid_cat:
        cat_col = df_export[col].astype("category")
        cat_label_mappings[col] = {int(i): str(v) for i, v in enumerate(cat_col.cat.categories)}

    # ---- 目标列 ----
    if target_col and target_col in df.columns:
        y_all = pd.to_numeric(df_export[target_col], errors="coerce").fillna(0).values.astype(np.float32)
    else:
        y_all = np.zeros(len(df), dtype=np.float32)

    # ---- train / test 划分 ----
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)

    for split_name, split_idx in [("train", idx_train), ("test", idx_test)]:
        np.save(os.path.join(output_dir, f"X_num_{split_name}.npy"), X_num_all[split_idx])
        np.save(os.path.join(output_dir, f"X_cat_{split_name}.npy"), X_cat_all[split_idx])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y_all[split_idx])

    # ---- 保存 train.csv (evaluation metrics 需要) ----
    all_cols = valid_num + valid_cat + ([target_col] if target_col else [])
    df_train = df_export.iloc[idx_train][all_cols].copy()
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test = df_export.iloc[idx_test][all_cols].copy()
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # ---- category sizes ----
    categories = [cat_categories[c] for c in valid_cat]

    # ---- int_col_idx_wrt_num: 整数列在 X_num 中的索引 ----
    int_col_idx = []
    for i, col in enumerate(valid_num):
        vals = df_export[col].dropna()
        if len(vals) > 0 and (vals == vals.astype(int)).all():
            int_col_idx.append(i)

    # ---- column index mapping (CausalDiffTab 需要) ----
    # train.csv 列顺序: [num_cols..., cat_cols..., target]
    n_num = len(valid_num)
    n_cat = len(valid_cat)
    num_col_idx = list(range(n_num))
    cat_col_idx = list(range(n_num, n_num + n_cat))
    target_col_idx = [n_num + n_cat] if target_col else []

    all_col_names = valid_num + valid_cat + ([target_col] if target_col else [])
    total_cols = len(all_col_names)

    idx_mapping = {i: i for i in range(total_cols)}
    inverse_idx_mapping = {i: i for i in range(total_cols)}
    idx_name_mapping = {i: all_col_names[i] for i in range(total_cols)}

    # ---- info.json ----
    info = {
        "task_type": task_type,
        "n_num_features": len(valid_num),
        "n_cat_features": len(valid_cat),
        "n_classes": None if task_type == "regression" else int(y_all.max() + 1),
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "train_num": len(idx_train),
        "test_num": len(idx_test),
        "val_num": 0,
        "num_col_names": valid_num,
        "cat_col_names": valid_cat,
        "cat_sizes": categories,
        "target_col": target_col,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "column_names": all_col_names,
        "int_col_idx_wrt_num": int_col_idx,
    }

    metadata = {"columns": {}}
    for i in num_col_idx:
        metadata["columns"][i] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
    for i in cat_col_idx:
        metadata["columns"][i] = {"sdtype": "categorical"}
    for i in target_col_idx:
        if task_type == "regression":
            metadata["columns"][i] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        else:
            metadata["columns"][i] = {"sdtype": "categorical"}

    info["metadata"] = metadata
    info["cat_label_mappings"] = cat_label_mappings

    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[prepare] {output_dir}")
    print(f"  X_num: {X_num_all.shape[1]} cols, X_cat: {X_cat_all.shape[1]} cols")
    print(f"  categories: {categories}")
    print(f"  train={len(idx_train)}, test={len(idx_test)}")

    return info


def build_dataset_dir_from_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    target_col: str,
    output_dir: str,
    task_type: str = "regression",
):
    """Write an existing train/test split in TabDiff npy format without reshuffling."""
    os.makedirs(output_dir, exist_ok=True)

    valid_num = [c for c in num_cols if c in train_df.columns or c in test_df.columns]
    valid_cat = [c for c in cat_cols if c in train_df.columns or c in test_df.columns]
    missing_category = "__MISSING__"

    train_export = train_df.copy()
    test_export = test_df.copy()
    for col in valid_num + valid_cat + ([target_col] if target_col else []):
        if col not in train_export.columns:
            train_export[col] = np.nan
        if col not in test_export.columns:
            test_export[col] = np.nan

    combined = pd.concat([train_export, test_export], axis=0, ignore_index=True)
    for col in valid_cat:
        combined[col] = combined[col].where(combined[col].notna(), missing_category).astype(str)

    train_n = len(train_export)
    combined_num = combined[valid_num].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    cat_encoded = combined[valid_cat].copy()
    cat_categories = {}
    cat_label_mappings = {}
    for col in valid_cat:
        cat_col = cat_encoded[col].astype("category")
        cat_encoded[col] = cat_col.cat.codes
        cat_categories[col] = int(cat_col.cat.categories.size)
        cat_label_mappings[col] = {int(i): str(v) for i, v in enumerate(cat_col.cat.categories)}
    combined_cat = cat_encoded.values.astype(np.int64)

    if target_col and target_col in combined.columns:
        y_all = pd.to_numeric(combined[target_col], errors="coerce").fillna(0).values.astype(np.float32)
    else:
        y_all = np.zeros(len(combined), dtype=np.float32)

    split_indices = {
        "train": np.arange(train_n),
        "test": np.arange(train_n, len(combined)),
    }
    for split_name, split_idx in split_indices.items():
        np.save(os.path.join(output_dir, f"X_num_{split_name}.npy"), combined_num[split_idx])
        np.save(os.path.join(output_dir, f"X_cat_{split_name}.npy"), combined_cat[split_idx])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y_all[split_idx])

    all_cols = valid_num + valid_cat + ([target_col] if target_col else [])
    train_export = combined.iloc[:train_n][all_cols].copy()
    test_export = combined.iloc[train_n:][all_cols].copy()
    train_export.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_export.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    categories = [cat_categories[c] for c in valid_cat]
    int_col_idx = []
    for i, col in enumerate(valid_num):
        vals = combined[col].dropna()
        if len(vals) > 0 and (vals == vals.astype(int)).all():
            int_col_idx.append(i)

    n_num = len(valid_num)
    n_cat = len(valid_cat)
    num_col_idx = list(range(n_num))
    cat_col_idx = list(range(n_num, n_num + n_cat))
    target_col_idx = [n_num + n_cat] if target_col else []
    all_col_names = all_cols
    total_cols = len(all_col_names)

    info = {
        "task_type": task_type,
        "n_num_features": n_num,
        "n_cat_features": n_cat,
        "n_classes": None if task_type == "regression" else int(y_all.max() + 1),
        "train_size": train_n,
        "test_size": len(test_export),
        "train_num": train_n,
        "test_num": len(test_export),
        "val_num": 0,
        "num_col_names": valid_num,
        "cat_col_names": valid_cat,
        "cat_sizes": categories,
        "target_col": target_col,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "idx_mapping": {i: i for i in range(total_cols)},
        "inverse_idx_mapping": {i: i for i in range(total_cols)},
        "idx_name_mapping": {i: all_col_names[i] for i in range(total_cols)},
        "column_names": all_col_names,
        "int_col_idx_wrt_num": int_col_idx,
        "cat_label_mappings": cat_label_mappings,
    }

    metadata = {"columns": {}}
    for i in num_col_idx:
        metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
    for i in cat_col_idx:
        metadata["columns"][i] = {"sdtype": "categorical"}
    for i in target_col_idx:
        metadata["columns"][i] = {"sdtype": "numerical", "computer_representation": "Float"}
    info["metadata"] = metadata

    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[prepare-split] {output_dir}")
    print(f"  X_num: {combined_num.shape[1]} cols, X_cat: {combined_cat.shape[1]} cols")
    print(f"  categories: {categories}")
    print(f"  train={train_n}, test={len(test_export)}")
    return info


def build_causal_mask_for_model(
    notears_npy: str,
    info: dict,
    output_dir: str,
    task_type: str = "regression",
):
    """
    将 NOTEARS 产出的 45×45 全局因果矩阵，
    映射为 CausalDiffTab 所需的 num_causal_mask 和 cat_causal_mask。

    由于 CausalDiffTab 的 make_dataset 会 prepend y 到 X_num (regression)
    或 X_cat (classification)，所以掩码的维度需要对应扩展。

    num_causal_mask: (d_num_with_y) × (d_num_with_y)
    cat_causal_mask: (sum_one_hot_with_mask) × (sum_one_hot_with_mask)
    """
    W = np.load(notears_npy).astype(np.float32)
    n_num = info["n_num_features"]
    n_cat = info["n_cat_features"]
    cat_sizes = info["cat_sizes"]

    W_num_num = W[:n_num, :n_num]
    W_num_cat = W[:n_num, n_num:]
    W_cat_num = W[n_num:, :n_num]
    W_cat_cat = W[n_num:, n_num:]

    # ---- num_causal_mask: prepend y 行/列 ----
    if task_type == "regression":
        d_with_y = n_num + 1
        num_mask = np.zeros((d_with_y, d_with_y), dtype=np.float32)
        # y 行/列: y 受所有 num 特征影响 (保守设为全 1 让正则发现)
        num_mask[0, 1:] = 1.0  # y 行 (y 的因由所有 num 特征产生)
        num_mask[1:, 0] = 0.0  # 原始 num 特征不由 y 产生
        # 原始 num×num 部分
        num_mask[1:, 1:] = W_num_num
    else:
        num_mask = W_num_num.copy()

    # ---- cat_causal_mask: 展开到 one-hot 维度 ----
    # CausalDiffTab 的 cat 掩码是在 one-hot(+mask) 展开后的大矩阵上操作的
    # 每个 cat 特征 i 有 cat_sizes[i]+1 维 (含 mask class)
    expanded_sizes = [s + 1 for s in cat_sizes]
    total_cat_dim = sum(expanded_sizes)

    if task_type != "regression":
        # classification: y is prepended to cat, need extra expansion
        y_n_classes = info.get("n_classes", 2)
        expanded_sizes = [y_n_classes + 1] + expanded_sizes
        total_cat_dim = sum(expanded_sizes)

    cat_mask = np.zeros((total_cat_dim, total_cat_dim), dtype=np.float32)

    # 将 col-level W_cat_cat 展开到 one-hot 块级
    offset_start = 0 if task_type == "regression" else expanded_sizes[0]
    offsets = []
    cur = offset_start
    for s in (expanded_sizes if task_type == "regression" else expanded_sizes[1:]):
        offsets.append(cur)
        cur += s

    for i in range(n_cat):
        for j in range(n_cat):
            v = float(W_cat_cat[i, j])
            if v > 0:
                si, sj = expanded_sizes[i] if task_type == "regression" else expanded_sizes[i + 1], \
                          expanded_sizes[j] if task_type == "regression" else expanded_sizes[j + 1]
                oi, oj = offsets[i], offsets[j]
                # 保留软强度；binary 输入时 v == 1.0 与原行为一致
                cat_mask[oi:oi + si, oj:oj + sj] = v

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "num_causal_mask.npy"), num_mask)
    np.save(os.path.join(output_dir, "cat_causal_mask.npy"), cat_mask)

    print(f"[causal_mask] num_causal_mask: {num_mask.shape}, edges={int(num_mask.sum())}")
    print(f"[causal_mask] cat_causal_mask: {cat_mask.shape}, edges={int(cat_mask.sum())}")

    return num_mask, cat_mask


def run_prepare(
    input_csv: str,
    column_groups_json: str,
    notears_npy: str,
    base_output: str,
    target_col: str = "NUMBER OF PERSONS INJURED",
    seed: int = 42,
    input_test_csv: str = None,
    full_dataname: str = "nyc_crash",
    stage1_dataname: str = "nyc_stage1",
):
    """
    主流程:
    1. 读取预处理后的 CSV + column_groups
    2. 创建 Stage 1 (spatial) 数据集
    3. 创建 Stage 3 (full) 数据集
    4. 为两个 Stage 分别生成对应维度的因果掩码
    """
    print("=" * 60)
    print("Hierarchical CausalDiffTab - Dataset Preparation")
    print("=" * 60)

    with open(column_groups_json, "r", encoding="utf-8") as f:
        groups = json.load(f)

    df = pd.read_csv(input_csv, low_memory=False)
    df_test = pd.read_csv(input_test_csv, low_memory=False) if input_test_csv else None
    print(f"[load] {len(df)} rows from {input_csv}")
    if df_test is not None:
        print(f"[load] {len(df_test)} test rows from {input_test_csv}")

    # ============================================
    # Stage 1: 时空锚点 (Spatiotemporal Anchor)
    # ============================================
    stage1_dir = os.path.join(base_output, "data", stage1_dataname)
    stage1_num = groups["stage1_continuous"]   # LAT, LON, time_sin, time_cos
    stage1_cat = groups["stage1_categorical"]  # SEASON, DAY_OF_WEEK, TIME_PERIOD
    if df_test is not None:
        stage1_info = build_dataset_dir_from_splits(
            df, df_test, num_cols=stage1_num, cat_cols=stage1_cat,
            target_col=target_col, output_dir=stage1_dir,
            task_type="regression",
        )
    else:
        stage1_info = build_dataset_dir(
            df, num_cols=stage1_num, cat_cols=stage1_cat,
            target_col=target_col, output_dir=stage1_dir,
            task_type="regression", seed=seed,
        )

    # Stage 1 causal mask: extract sub-block from NOTEARS full matrix
    W_full = np.load(notears_npy).astype(np.float32)
    all_num = groups["continuous_cols"]
    all_cat = groups["categorical_cols"]
    all_features = all_num + all_cat
    s1_indices = [all_features.index(f) for f in stage1_num + stage1_cat if f in all_features]

    # num part of stage1
    d_s1_num = len(stage1_num)
    d_s1_num_with_y = d_s1_num + 1  # y prepended
    s1_num_idx = [all_features.index(f) for f in stage1_num if f in all_features]
    W_s1_num = W_full[np.ix_(s1_num_idx, s1_num_idx)]
    s1_num_mask = np.zeros((d_s1_num_with_y, d_s1_num_with_y), dtype=np.float32)
    s1_num_mask[0, 1:] = 1.0  # y <- all stage1 num
    s1_num_mask[1:, 1:] = W_s1_num

    # cat part of stage1
    s1_cat_sizes = stage1_info["cat_sizes"]
    s1_expanded = [s + 1 for s in s1_cat_sizes]
    s1_total_cat = sum(s1_expanded)
    s1_cat_idx = [all_features.index(f) for f in stage1_cat if f in all_features]
    n_s1_cat = len(s1_cat_idx)
    W_s1_cat = W_full[np.ix_(s1_cat_idx, s1_cat_idx)]
    s1_cat_mask = np.zeros((s1_total_cat, s1_total_cat), dtype=np.float32)
    offsets_s1 = []
    cur = 0
    for s in s1_expanded:
        offsets_s1.append(cur)
        cur += s
    for i in range(n_s1_cat):
        for j in range(n_s1_cat):
            v = float(W_s1_cat[i, j])
            if v > 0:
                si, sj = s1_expanded[i], s1_expanded[j]
                oi, oj = offsets_s1[i], offsets_s1[j]
                # 保留软强度；binary 输入时 v == 1.0 与原行为一致
                s1_cat_mask[oi:oi + si, oj:oj + sj] = v

    os.makedirs(os.path.join(stage1_dir, "causal_masks"), exist_ok=True)
    np.save(os.path.join(stage1_dir, "causal_masks", "num_causal_mask.npy"), s1_num_mask)
    np.save(os.path.join(stage1_dir, "causal_masks", "cat_causal_mask.npy"), s1_cat_mask)
    print(f"[stage1] Stage 1 spatiotemporal anchor dataset ready")

    # ============================================
    # Stage 3: Full (all features)
    # ============================================
    full_dir = os.path.join(base_output, "data", full_dataname)
    full_num = groups["continuous_cols"]
    full_cat = groups["categorical_cols"]
    if df_test is not None:
        full_info = build_dataset_dir_from_splits(
            df, df_test, num_cols=full_num, cat_cols=full_cat,
            target_col=target_col, output_dir=full_dir,
            task_type="regression",
        )
    else:
        full_info = build_dataset_dir(
            df, num_cols=full_num, cat_cols=full_cat,
            target_col=target_col, output_dir=full_dir,
            task_type="regression", seed=seed,
        )

    # Stage 3 causal mask: from NOTEARS
    mask_dir = os.path.join(full_dir, "causal_masks")
    build_causal_mask_for_model(
        notears_npy, full_info, mask_dir, task_type="regression"
    )
    print(f"[full] Stage 3 full dataset ready")

    # ============================================
    # 创建 synthetic 目录 (evaluation 需要)
    # ============================================
    synthetic_targets = [
        (stage1_dataname, stage1_dataname),
        ("nyc_spatial" if stage1_dataname == "nyc_stage1" else f"nyc_spatial_{full_dataname}", stage1_dataname),
        (full_dataname, full_dataname),
    ]
    for syn_name, data_name in synthetic_targets:
        syn_dir = os.path.join(base_output, "synthetic", syn_name)
        os.makedirs(syn_dir, exist_ok=True)
        src_dir = os.path.join(base_output, "data", data_name)
        for fn in ["train.csv", "test.csv"]:
            src_path = os.path.join(src_dir, fn)
            dst_path = os.path.join(syn_dir, fn.replace("train", "real") if fn == "train.csv" else fn)
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"  Stage 1 (spatiotemporal): {stage1_dir}")
    print(f"  Stage 3 (full):    {full_dir}")
    print("=" * 60)


def main():
    CDT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Prepare datasets for CausalDiffTab")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(CDT_ROOT / "data" / "processed" / "processed_hierarchical.csv"),
    )
    parser.add_argument(
        "--column_groups_json",
        type=str,
        default=str(CDT_ROOT / "data" / "processed" / "column_groups.json"),
    )
    parser.add_argument(
        "--notears_npy",
        type=str,
        default=str(CDT_ROOT / "configs" / "causal_matrix_notears_mlp.npy"),
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=str(CDT_ROOT),
    )
    parser.add_argument("--input_test_csv", type=str, default=None)
    parser.add_argument("--full_dataname", type=str, default="nyc_crash")
    parser.add_argument("--stage1_dataname", type=str, default="nyc_stage1")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_prepare(
        input_csv=args.input_csv,
        column_groups_json=args.column_groups_json,
        notears_npy=args.notears_npy,
        base_output=args.output_base,
        input_test_csv=args.input_test_csv,
        full_dataname=args.full_dataname,
        stage1_dataname=args.stage1_dataname,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
