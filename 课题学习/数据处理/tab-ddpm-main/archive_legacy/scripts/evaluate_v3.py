"""
evaluate_v3.py  —  v6 综合评估管线
====================================
评估维度:
  D1. 统计保真度 (Fidelity): JS 散度, Wasserstein 距离, 相关矩阵差
  D2. 下游任务 (Utility): TSTR CatBoost
      - 任务1: 回归 NUMBER_OF_PERSONS_INJURED (RMSE, R², MAE)
      - 任务2: 分类 PRIMARY_CAUSE (Accuracy, Weighted F1, Macro F1) ← 核心对比
      - 任务3: 分类 IS_INJURY (AUC, Accuracy, F1)
  D3. 多车事故专项分析
  D4. 隐私性 (DCR)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def load_npy_data(data_dir, split="train"):
    """加载 TabDDPM 标准 npy 格式数据。"""
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"), allow_pickle=True)
    X_num, X_cat = None, None
    p_num = os.path.join(data_dir, f"X_num_{split}.npy")
    p_cat = os.path.join(data_dir, f"X_cat_{split}.npy")
    if os.path.exists(p_num):
        X_num = np.load(p_num, allow_pickle=True).astype(float)
    if os.path.exists(p_cat):
        X_cat = np.load(p_cat, allow_pickle=True)
    return X_num, X_cat, y


def load_info(data_dir):
    with open(os.path.join(data_dir, "info.json")) as f:
        return json.load(f)


# =============================================
# D1. 统计保真度
# =============================================
def compute_fidelity(syn_dir, real_dir, info):
    """计算合成 vs 真实的统计保真度。"""
    print("  📊 D1: 统计保真度...")
    X_num_real, X_cat_real, y_real = load_npy_data(real_dir, "train")
    X_num_syn, X_cat_syn, y_syn = load_npy_data(syn_dir, "train")

    results = {}

    # 连续特征
    num_cols = info.get("num_columns", [])
    wd = {}
    for i, col in enumerate(num_cols):
        if X_num_real is not None and X_num_syn is not None:
            if i < X_num_real.shape[1] and i < X_num_syn.shape[1]:
                w = wasserstein_distance(X_num_real[:, i], X_num_syn[:, i])
                wd[col] = round(float(w), 4)
    results["wasserstein_distances"] = wd

    # 分类特征 JS 散度
    cat_cols = info.get("cat_columns", [])
    cat_sizes = info.get("cat_sizes", [])
    js = {}
    for j, col in enumerate(cat_cols):
        if X_cat_real is not None and X_cat_syn is not None:
            if j < X_cat_real.shape[1] and j < X_cat_syn.shape[1]:
                real_v = X_cat_real[:, j].astype(int)
                syn_v = X_cat_syn[:, j].astype(int)
                n_cls = cat_sizes[j] if j < len(cat_sizes) else max(real_v.max(), syn_v.max()) + 1
                h_real = np.bincount(real_v, minlength=n_cls).astype(float)
                h_syn = np.bincount(syn_v[syn_v >= 0], minlength=n_cls).astype(float)
                h_real /= h_real.sum() + 1e-10
                h_syn /= h_syn.sum() + 1e-10
                js[col] = round(float(jensenshannon(h_real, h_syn)), 4)
    results["js_divergences"] = js

    # y 分布
    results["y_wasserstein"] = round(float(wasserstein_distance(y_real, y_syn)), 4)
    results["y_stats"] = {
        "real_mean": round(float(y_real.mean()), 4),
        "syn_mean": round(float(y_syn.mean()), 4),
        "real_std": round(float(y_real.std()), 4),
        "syn_std": round(float(y_syn.std()), 4),
    }

    # 相关矩阵 Frobenius 范数差
    if X_num_real is not None and X_num_syn is not None:
        try:
            corr_real = np.corrcoef(X_num_real.T)
            corr_syn = np.corrcoef(X_num_syn.T)
            fro = np.linalg.norm(corr_real - corr_syn, "fro")
            results["corr_frobenius_diff"] = round(float(fro), 4)
        except Exception:
            pass

    # CAUSE 列 JS 散度
    cause_js = {k: v for k, v in js.items() if "CAUSE" in k}
    if cause_js:
        results["cause_js_divergence_avg"] = round(np.mean(list(cause_js.values())), 4)

    avg_js = np.mean(list(js.values())) if js else 0
    results["avg_js_divergence"] = round(float(avg_js), 4)

    print(f"    平均 JS 散度: {results['avg_js_divergence']}")
    print(f"    y Wasserstein: {results['y_wasserstein']}")
    if cause_js:
        print(f"    CAUSE 平均 JS: {results.get('cause_js_divergence_avg', 'N/A')}")
    return results


# =============================================
# D2. 下游任务 (TSTR CatBoost)
# =============================================
def compute_utility(syn_dir, real_dir, info):
    """Train-on-Synthetic, Test-on-Real 评估。"""
    print("  🎯 D2: 下游任务评估 (TSTR CatBoost)...")
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool

    X_num_syn, X_cat_syn, y_syn = load_npy_data(syn_dir, "train")
    X_num_test, X_cat_test, y_test = load_npy_data(real_dir, "test")
    X_num_val, X_cat_val, y_val = load_npy_data(real_dir, "val")

    n_num = X_num_syn.shape[1] if X_num_syn is not None else 0
    n_cat = X_cat_syn.shape[1] if X_cat_syn is not None else 0
    cat_features = list(range(n_num, n_num + n_cat))

    def _concat(X_num, X_cat):
        parts = []
        if X_num is not None:
            parts.append(X_num.astype(float))
        if X_cat is not None:
            parts.append(X_cat.astype(str))
        return np.concatenate(parts, axis=1) if parts else None

    X_train = _concat(X_num_syn, X_cat_syn)
    X_test = _concat(X_num_test, X_cat_test)
    X_val = _concat(X_num_val, X_cat_val)

    # 转 DataFrame 以便正确设置列类型
    def _to_df(X, cat_features):
        df = pd.DataFrame(X)
        for c in cat_features:
            df[c] = df[c].astype(str)
        for c in range(n_num):
            df[c] = df[c].astype(float)
        return df

    X_train_df = _to_df(X_train, cat_features)
    X_test_df = _to_df(X_test, cat_features)
    X_val_df = _to_df(X_val, cat_features)

    results = {}

    # === 任务 1: 回归 (NUMBER OF PERSONS INJURED) ===
    print("    📈 任务1: 回归 (RMSE, R², MAE)")
    catboost_params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
        "cat_features": cat_features,
        "early_stopping_rounds": 50,
        "random_seed": 42,
    }

    reg = CatBoostRegressor(**catboost_params, eval_metric="RMSE")
    reg.fit(X_train_df, y_syn, eval_set=(X_val_df, y_val), verbose=0)

    pred_test = reg.predict(X_test_df)
    pred_val = reg.predict(X_val_df)

    results["task1_regression"] = {
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 4),
        "test_r2": round(float(r2_score(y_test, pred_test)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, pred_test)), 4),
        "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, pred_val))), 4),
        "val_r2": round(float(r2_score(y_val, pred_val)), 4),
    }
    print(f"      Test RMSE={results['task1_regression']['test_rmse']} R²={results['task1_regression']['test_r2']}")

    # === 任务 2: 分类 PRIMARY_CAUSE (核心对比) ===
    print("    📈 任务2: PRIMARY_CAUSE 分类 (核心)")
    cat_cols = info.get("cat_columns", [])
    cause_cols = [c for c in cat_cols if c.startswith("CAUSE_")]

    if cause_cols:
        # 构建 PRIMARY_CAUSE: CAUSE 列中值为 "1" 的第一个（按频率排序）
        def _get_primary_cause_from_cat(X_cat, cat_cols, cause_cols):
            """从分类编码数据构建 PRIMARY_CAUSE。"""
            cause_indices = [cat_cols.index(c) for c in cause_cols if c in cat_cols]
            # 如果有 column_mapping，用它来判断哪个类别值代表 "1"
            # 否则简单取 argmax
            primary = np.zeros(len(X_cat), dtype=int)
            for i in range(len(X_cat)):
                # 取第一个为 "1" 的 cause (对应的 cat 值需要检查)
                # 简化: 取值最大的 cause 列
                cause_vals = [int(X_cat[i, idx]) for idx in cause_indices]
                primary[i] = np.argmax(cause_vals)  # 哪个 CAUSE 列值最大
            return primary

        # 加载映射来找出 "1" 对应的编码值
        mapping_path = os.path.join(real_dir, "column_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                col_mapping = json.load(f)
        else:
            col_mapping = {}

        def _build_primary_cause(X_cat, cat_cols, cause_cols, col_mapping):
            """从分类数据构建 PRIMARY_CAUSE 标签 (0-based)。"""
            n = len(X_cat)
            primary = np.full(n, len(cause_cols) - 1, dtype=int)  # 默认最后一个
            cause_indices = [cat_cols.index(c) for c in cause_cols if c in cat_cols]
            for i in range(n):
                for k, cidx in enumerate(cause_indices):
                    col_name = cause_cols[k]
                    val = int(X_cat[i, cidx])
                    # 检查这个值是否对应 "1"
                    if col_name in col_mapping:
                        idx_for_one = col_mapping[col_name].get("1", -1)
                        if val == idx_for_one:
                            primary[i] = k
                            break
                    else:
                        if val == 1:  # 直接比较
                            primary[i] = k
                            break
            return primary

        y_cause_syn = _build_primary_cause(X_cat_syn, cat_cols, cause_cols, col_mapping)
        y_cause_test = _build_primary_cause(X_cat_test, cat_cols, cause_cols, col_mapping)
        y_cause_val = _build_primary_cause(X_cat_val, cat_cols, cause_cols, col_mapping)

        # 从特征集中移除 CAUSE 列来训练分类器
        non_cause_cat_indices = [i for i, c in enumerate(cat_cols) if not c.startswith("CAUSE_")]
        non_cause_num = n_num  # 数值列全保留
        X_train_no_cause = np.concatenate([
            X_num_syn.astype(float) if X_num_syn is not None else np.empty((len(y_syn), 0)),
            X_cat_syn[:, non_cause_cat_indices].astype(str) if (X_cat_syn is not None and non_cause_cat_indices) else np.empty((len(y_syn), 0)),
        ], axis=1) if non_cause_num + len(non_cause_cat_indices) > 0 else X_train
        X_test_no_cause = np.concatenate([
            X_num_test.astype(float) if X_num_test is not None else np.empty((len(y_test), 0)),
            X_cat_test[:, non_cause_cat_indices].astype(str) if (X_cat_test is not None and non_cause_cat_indices) else np.empty((len(y_test), 0)),
        ], axis=1)
        X_val_no_cause = np.concatenate([
            X_num_val.astype(float) if X_num_val is not None else np.empty((len(y_val), 0)),
            X_cat_val[:, non_cause_cat_indices].astype(str) if (X_cat_val is not None and non_cause_cat_indices) else np.empty((len(y_val), 0)),
        ], axis=1)

        cat_feat_no_cause = list(range(non_cause_num, non_cause_num + len(non_cause_cat_indices)))

        clf_params = {
            "iterations": 2000, "learning_rate": 0.05, "depth": 6,
            "verbose": 0, "cat_features": cat_feat_no_cause,
            "early_stopping_rounds": 50, "random_seed": 42,
        }

        n_classes = len(cause_cols)
        clf = CatBoostClassifier(
            **clf_params,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            class_names=[str(i) for i in range(n_classes)],
        )

        X_train_nc_df = pd.DataFrame(X_train_no_cause)
        X_test_nc_df = pd.DataFrame(X_test_no_cause)
        X_val_nc_df = pd.DataFrame(X_val_no_cause)
        for c in cat_feat_no_cause:
            X_train_nc_df[c] = X_train_nc_df[c].astype(str)
            X_test_nc_df[c] = X_test_nc_df[c].astype(str)
            X_val_nc_df[c] = X_val_nc_df[c].astype(str)
        for c in range(non_cause_num):
            X_train_nc_df[c] = X_train_nc_df[c].astype(float)
            X_test_nc_df[c] = X_test_nc_df[c].astype(float)
            X_val_nc_df[c] = X_val_nc_df[c].astype(float)

        clf.fit(X_train_nc_df, y_cause_syn, eval_set=(X_val_nc_df, y_cause_val), verbose=0)
        pred_cause = clf.predict(X_test_nc_df).flatten().astype(int)

        results["task2_primary_cause"] = {
            "test_accuracy": round(float(accuracy_score(y_cause_test, pred_cause)), 4),
            "test_weighted_f1": round(float(f1_score(y_cause_test, pred_cause, average="weighted")), 4),
            "test_macro_f1": round(float(f1_score(y_cause_test, pred_cause, average="macro")), 4),
            "n_classes": n_classes,
            "cause_columns": cause_cols,
        }
        print(f"      Accuracy={results['task2_primary_cause']['test_accuracy']} "
              f"W-F1={results['task2_primary_cause']['test_weighted_f1']} "
              f"M-F1={results['task2_primary_cause']['test_macro_f1']}")
    else:
        results["task2_primary_cause"] = {"status": "no_cause_columns"}

    # === 任务 3: IS_INJURY 二分类 ===
    print("    📈 任务3: IS_INJURY 二分类")
    y_injury_syn = (y_syn > 0).astype(int)
    y_injury_test = (y_test > 0).astype(int)
    y_injury_val = (y_val > 0).astype(int)

    clf_binary = CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=6,
        verbose=0, cat_features=cat_features,
        early_stopping_rounds=50, random_seed=42,
        eval_metric="AUC",
    )
    clf_binary.fit(X_train_df, y_injury_syn, eval_set=(X_val_df, y_injury_val), verbose=0)
    pred_injury = clf_binary.predict(X_test_df).flatten().astype(int)
    pred_proba_injury = clf_binary.predict_proba(X_test_df)[:, 1]

    results["task3_is_injury"] = {
        "test_accuracy": round(float(accuracy_score(y_injury_test, pred_injury)), 4),
        "test_f1": round(float(f1_score(y_injury_test, pred_injury)), 4),
        "test_auc": round(float(roc_auc_score(y_injury_test, pred_proba_injury)), 4),
    }
    print(f"      Accuracy={results['task3_is_injury']['test_accuracy']} "
          f"F1={results['task3_is_injury']['test_f1']} "
          f"AUC={results['task3_is_injury']['test_auc']}")

    return results


# =============================================
# D3. 多车事故分析
# =============================================
def compute_multi_vehicle_analysis(syn_dir, real_dir, info):
    """多车事故专项分析。"""
    print("  🚗 D3: 多车事故分析...")
    cat_cols = info.get("cat_columns", [])

    X_cat_syn = np.load(os.path.join(syn_dir, "X_cat_train.npy"), allow_pickle=True)
    X_cat_real = np.load(os.path.join(real_dir, "X_cat_train.npy"), allow_pickle=True)

    results = {}

    # VEHICLE TYPE CODE 3/4/5 UNSPECIFIED 比例
    mapping_path = os.path.join(real_dir, "column_mapping.json")
    col_mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            col_mapping = json.load(f)

    for vt_col in ["VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"]:
        if vt_col in cat_cols:
            idx = cat_cols.index(vt_col)
            if vt_col in col_mapping:
                unspec_code = col_mapping[vt_col].get("UNSPECIFIED", -1)
                if unspec_code >= 0:
                    real_ratio = (X_cat_real[:, idx].astype(int) == unspec_code).mean()
                    syn_ratio = (X_cat_syn[:, idx].astype(int) == unspec_code).mean()
                    results[f"{vt_col}_unspecified_real"] = round(float(real_ratio), 4)
                    results[f"{vt_col}_unspecified_syn"] = round(float(syn_ratio), 4)
                    results[f"{vt_col}_unspecified_diff"] = round(abs(real_ratio - syn_ratio), 4)

    # TOTAL_VEHICLES 分布
    if "TOTAL_VEHICLES" in cat_cols:
        idx = cat_cols.index("TOTAL_VEHICLES")
        real_tv = X_cat_real[:, idx].astype(int)
        syn_tv = X_cat_syn[:, idx].astype(int)
        # 多车比例 (>=3 类)
        if "TOTAL_VEHICLES" in col_mapping:
            tv_map = col_mapping["TOTAL_VEHICLES"]
            multi_codes = [int(v) for k, v in tv_map.items() if k not in ("1", "2")]
            real_multi = np.isin(real_tv, multi_codes).mean()
            syn_multi = np.isin(syn_tv, multi_codes).mean()
            results["multi_vehicle_ratio_real"] = round(float(real_multi), 4)
            results["multi_vehicle_ratio_syn"] = round(float(syn_multi), 4)

    print(f"    多车事故分析: {json.dumps(results, indent=2)}")
    return results


# =============================================
# D4. 隐私性 (DCR)
# =============================================
def compute_privacy(syn_dir, real_dir, info, sample_size=5000):
    """计算 DCR (Distance to Closest Record)。"""
    print("  🔒 D4: 隐私性分析 (DCR)...")
    X_num_real = np.load(os.path.join(real_dir, "X_num_train.npy"), allow_pickle=True).astype(float)
    X_num_syn = np.load(os.path.join(syn_dir, "X_num_train.npy"), allow_pickle=True).astype(float)

    # 只用连续特征计算 DCR (更快)
    n_syn = min(sample_size, len(X_num_syn))
    n_real = len(X_num_real)

    idx_syn = np.random.choice(len(X_num_syn), n_syn, replace=False)
    X_syn_sample = X_num_syn[idx_syn]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_num_real)
    dists, _ = nn.kneighbors(X_syn_sample)
    dists = dists.flatten()

    results = {
        "dcr_mean": round(float(dists.mean()), 4),
        "dcr_median": round(float(np.median(dists)), 4),
        "dcr_min": round(float(dists.min()), 6),
        "dcr_5th_percentile": round(float(np.percentile(dists, 5)), 4),
        "dcr_exact_match_ratio": round(float((dists < 1e-6).mean()), 6),
    }
    print(f"    DCR mean={results['dcr_mean']} median={results['dcr_median']} min={results['dcr_min']}")
    print(f"    精确复制比例: {results['dcr_exact_match_ratio']}")
    return results


# =============================================
# D5. 可视化
# =============================================
def generate_plots(all_results, output_dir):
    """生成对比可视化图表。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. TSTR 回归指标对比
        models = list(all_results.keys())
        r2_vals = [all_results[m].get("utility", {}).get("task1_regression", {}).get("test_r2", 0) for m in models]
        rmse_vals = [all_results[m].get("utility", {}).get("task1_regression", {}).get("test_rmse", 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(models, r2_vals, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
        ax1.set_title("TSTR Test R² (Regression)", fontsize=12)
        ax1.set_ylabel("R²")
        ax1.tick_params(axis="x", rotation=30)

        ax2.bar(models, rmse_vals, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
        ax2.set_title("TSTR Test RMSE (Regression)", fontsize=12)
        ax2.set_ylabel("RMSE")
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "tstr_regression.png"), dpi=150)
        plt.close()

        # 2. 事故原因分类对比
        acc_vals = [all_results[m].get("utility", {}).get("task2_primary_cause", {}).get("test_accuracy", 0) for m in models]
        wf1_vals = [all_results[m].get("utility", {}).get("task2_primary_cause", {}).get("test_weighted_f1", 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(models, acc_vals, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
        ax1.set_title("PRIMARY_CAUSE Accuracy (Core Metric)", fontsize=12)
        ax1.set_ylabel("Accuracy")
        ax1.tick_params(axis="x", rotation=30)

        ax2.bar(models, wf1_vals, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
        ax2.set_title("PRIMARY_CAUSE Weighted F1", fontsize=12)
        ax2.set_ylabel("Weighted F1")
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "primary_cause_classification.png"), dpi=150)
        plt.close()

        # 3. JS 散度对比
        avg_js = [all_results[m].get("fidelity", {}).get("avg_js_divergence", 0) for m in models]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(models, avg_js, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
        ax.set_title("Average JS Divergence (lower = better)", fontsize=12)
        ax.set_ylabel("JS Divergence")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "js_divergence.png"), dpi=150)
        plt.close()

        print(f"    📊 图表已保存到 {plots_dir}")

    except ImportError:
        print("    ⚠️ matplotlib 未安装，跳过图表生成")


# =============================================
# 主评估函数
# =============================================
def evaluate_model(syn_dir, real_dir, info, model_name="model"):
    """对单个模型运行全部评估。"""
    print(f"\n{'='*60}")
    print(f"📊 评估: {model_name}")
    print(f"{'='*60}")

    results = {"model": model_name, "syn_dir": syn_dir}

    results["fidelity"] = compute_fidelity(syn_dir, real_dir, info)
    results["utility"] = compute_utility(syn_dir, real_dir, info)
    results["multi_vehicle"] = compute_multi_vehicle_analysis(syn_dir, real_dir, info)
    results["privacy"] = compute_privacy(syn_dir, real_dir, info)

    return results


def evaluate_all(real_dir, model_dirs, output_dir=None):
    """评估所有模型并生成对比报告。"""
    info = load_info(real_dir)
    all_results = {}

    for name, syn_dir in model_dirs.items():
        if not os.path.exists(os.path.join(syn_dir, "y_train.npy")):
            print(f"⚠️ 跳过 {name}: {syn_dir} 无数据")
            continue
        try:
            result = evaluate_model(syn_dir, real_dir, info, model_name=name)
            all_results[name] = result
        except Exception as e:
            print(f"❌ {name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"model": name, "error": str(e)}

    # 保存
    if output_dir is None:
        output_dir = os.path.dirname(list(model_dirs.values())[0]) if model_dirs else "."
    os.makedirs(output_dir, exist_ok=True)

    # JSON 报告
    report_path = os.path.join(output_dir, "model_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 对比报告: {report_path}")

    # CSV 汇总表
    rows = []
    for name, res in all_results.items():
        if "error" in res:
            continue
        row = {"model": name}
        # 回归
        t1 = res.get("utility", {}).get("task1_regression", {})
        row["test_rmse"] = t1.get("test_rmse", "")
        row["test_r2"] = t1.get("test_r2", "")
        row["test_mae"] = t1.get("test_mae", "")
        # 事故原因分类
        t2 = res.get("utility", {}).get("task2_primary_cause", {})
        row["cause_accuracy"] = t2.get("test_accuracy", "")
        row["cause_weighted_f1"] = t2.get("test_weighted_f1", "")
        row["cause_macro_f1"] = t2.get("test_macro_f1", "")
        # IS_INJURY
        t3 = res.get("utility", {}).get("task3_is_injury", {})
        row["injury_accuracy"] = t3.get("test_accuracy", "")
        row["injury_f1"] = t3.get("test_f1", "")
        row["injury_auc"] = t3.get("test_auc", "")
        # 保真度
        row["avg_js"] = res.get("fidelity", {}).get("avg_js_divergence", "")
        # 隐私
        row["dcr_mean"] = res.get("privacy", {}).get("dcr_mean", "")
        rows.append(row)

    if rows:
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"📁 CSV 汇总: {csv_path}")

    # 图表
    generate_plots(all_results, output_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="v3 综合评估管线")
    parser.add_argument("--real_dir", type=str, default="data/nyc_crash_v3",
                        help="真实数据目录")
    parser.add_argument("--syn_dir", type=str, default=None,
                        help="单个合成数据目录 (与 --model_name 搭配)")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--output_dir", type=str, default="exp/nyc_crash_v3")
    parser.add_argument("--all", action="store_true",
                        help="评估所有模型 (自动发现 exp/nyc_crash_v3/ 下的子目录)")
    args = parser.parse_args()

    if args.all:
        base = args.output_dir
        model_dirs = {}
        for d in sorted(os.listdir(base)):
            full_path = os.path.join(base, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "y_train.npy")):
                model_dirs[d] = full_path
        if not model_dirs:
            print("❌ 没有找到可评估的模型目录")
            return
        print(f"📋 发现 {len(model_dirs)} 个模型: {list(model_dirs.keys())}")
        evaluate_all(args.real_dir, model_dirs, args.output_dir)
    elif args.syn_dir:
        info = load_info(args.real_dir)
        result = evaluate_model(args.syn_dir, args.real_dir, info, model_name=args.model_name)
        output_path = os.path.join(args.syn_dir, "evaluation_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📁 评估报告: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
