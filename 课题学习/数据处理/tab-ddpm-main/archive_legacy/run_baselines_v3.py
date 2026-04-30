"""
run_baselines_v3.py  —  v3 基线模型统一运行器
================================================
支持模型: TVAE, CTGAN, SMOTE, DDPM_MLP
流程: 训练 → 采样 → 保存 npy → CatBoost TSTR → 综合评估

用法:
  python run_baselines_v3.py --model tvae
  python run_baselines_v3.py --model ctgan
  python run_baselines_v3.py --model smote
  python run_baselines_v3.py --model ddpm_mlp
  python run_baselines_v3.py --model all        # 运行全部
  python run_baselines_v3.py --eval_only         # 只运行综合评估
"""

import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Any

import pandas as pd

import lib
from scripts.eval_catboost import train_catboost


REAL_DATA = "data/nyc_crash_v3"
EXP_BASE = "exp/nyc_crash_v3"


# ==================================
# 通用工具
# ==================================
def load_real_data(real_data_path, split="train"):
    rp = Path(real_data_path)
    X_num, X_cat, y = lib.read_pure_data(rp, split)
    info = lib.load_json(rp / "info.json")
    return X_num, X_cat, y, info


def build_dataframe(X_num, X_cat, y, info: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    n_num = X_num.shape[1] if X_num is not None else 0
    n_cat = X_cat.shape[1] if X_cat is not None else 0
    cols_num = [f"num_{i}" for i in range(n_num)]
    cols_cat = [f"cat_{i}" for i in range(n_cat)]
    parts: list[pd.DataFrame] = []
    if X_num is not None:
        parts.append(pd.DataFrame(X_num.astype(float), columns=cols_num))
    if X_cat is not None:
        parts.append(pd.DataFrame(X_cat.astype(str), columns=cols_cat))
    parts.append(pd.DataFrame(y, columns=["y"]))
    df = pd.concat(parts, axis=1)
    discrete_columns = list(cols_cat)
    if info["task_type"] != "regression":
        discrete_columns.append("y")
    return df, discrete_columns, cols_num, cols_cat


def save_synthetic(parent_dir, df_or_arrays, cols_num=None, cols_cat=None, info=None,
                   real_data_path=None):
    """保存合成数据为 npy 格式。"""
    parent_dir = Path(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)
    import pandas as pd

    if isinstance(df_or_arrays, pd.DataFrame):
        df = df_or_arrays
        if cols_num:
            np.save(parent_dir / "X_num_train.npy", np.asarray(df[cols_num].values, dtype=float))
        if cols_cat:
            np.save(parent_dir / "X_cat_train.npy", np.asarray(df[cols_cat].values, dtype=str))
        y = np.asarray(df["y"].values, dtype=float)
        np.save(parent_dir / "y_train.npy", y)
    else:
        X_num, X_cat, y = df_or_arrays
        if X_num is not None:
            np.save(parent_dir / "X_num_train.npy", X_num.astype(float))
        if X_cat is not None:
            np.save(parent_dir / "X_cat_train.npy", X_cat.astype(str))
        np.save(parent_dir / "y_train.npy", y.astype(float))

    # 复制 info.json
    rdp = real_data_path or REAL_DATA
    src = Path(rdp) / "info.json"
    if src.exists():
        shutil.copy2(src, parent_dir / "info.json")

    # 复制 column_mapping.json
    src_cm = Path(rdp) / "column_mapping.json"
    if src_cm.exists():
        shutil.copy2(src_cm, parent_dir / "column_mapping.json")


def run_catboost_eval(parent_dir, real_data_path=REAL_DATA, seed=0):
    """运行 CatBoost TSTR 评估。"""
    T_dict = {
        "seed": 0,
        "normalization": None,
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }
    report = train_catboost(
        parent_dir=parent_dir,
        real_data_path=real_data_path,
        eval_type="synthetic",
        T_dict=T_dict,
        seed=seed,
        change_val=False,
    )
    return report


# ==================================
# TVAE
# ==================================
def run_tvae(real_data_path=REAL_DATA, exp_base=EXP_BASE, epochs=300, batch_size=512, lr=1e-3, num_samples=None):
    from ctgan import TVAE

    parent_dir = f"{exp_base}/tvae"
    X_num, X_cat, y, info = load_real_data(real_data_path)
    if num_samples is None:
        num_samples = info["train_size"]
    df, discrete_columns, cols_num, cols_cat = build_dataframe(X_num, X_cat, y, info)

    print("=" * 70)
    print(f"[TVAE] 训练 | epochs={epochs}  batch={batch_size}  lr={lr}")
    print(f"  数据: {df.shape}  (num={len(cols_num)}, cat={len(cols_cat)})")
    t0 = time.time()

    model = TVAE(
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=min(batch_size, len(df)),
        epochs=epochs,
        loss_factor=2,
        enable_gpu=True,
        verbose=True,
    )
    model.fit(df, discrete_columns)
    t_train = time.time() - t0
    print(f"[TVAE] 训练完成, 用时 {t_train:.1f}s")

    syn = model.sample(num_samples)
    save_synthetic(parent_dir, syn, cols_num, cols_cat, info, real_data_path=real_data_path)
    t_total = time.time() - t0
    print(f"[TVAE] 采样 {num_samples} 条完成, 总用时 {t_total:.1f}s")

    # CatBoost 评估
    eval_report = run_catboost_eval(parent_dir, real_data_path=real_data_path)
    return {"train_time": t_train, "total_time": t_total, "catboost": eval_report}


# ==================================
# CTGAN
# ==================================
def run_ctgan(real_data_path=REAL_DATA, exp_base=EXP_BASE, epochs=300, batch_size=500, num_samples=None):
    from ctgan import CTGAN

    parent_dir = f"{exp_base}/ctgan"
    X_num, X_cat, y, info = load_real_data(real_data_path)
    if num_samples is None:
        num_samples = info["train_size"]
    df, discrete_columns, cols_num, cols_cat = build_dataframe(X_num, X_cat, y, info)

    print("=" * 70)
    print(f"[CTGAN] 训练 | epochs={epochs}  batch={batch_size}")
    print(f"  数据: {df.shape}  (num={len(cols_num)}, cat={len(cols_cat)})")
    t0 = time.time()

    model = CTGAN(
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=min(batch_size, len(df)),
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=epochs,
        pac=10,
        enable_gpu=True,
    )
    model.fit(df, discrete_columns)
    t_train = time.time() - t0
    print(f"[CTGAN] 训练完成, 用时 {t_train:.1f}s")

    syn = model.sample(num_samples)
    save_synthetic(parent_dir, syn, cols_num, cols_cat, info, real_data_path=real_data_path)
    t_total = time.time() - t0
    print(f"[CTGAN] 采样 {num_samples} 条完成, 总用时 {t_total:.1f}s")

    eval_report = run_catboost_eval(parent_dir, real_data_path=real_data_path)
    return {"train_time": t_train, "total_time": t_total, "catboost": eval_report}


# ==================================
# SMOTE
# ==================================
def run_smote(real_data_path=REAL_DATA, exp_base=EXP_BASE, k_neighbours=5, num_samples=None):
    """运行 SMOTE 基线。"""
    parent_dir = f"{exp_base}/smote"
    os.makedirs(parent_dir, exist_ok=True)

    X_num, X_cat, y, info = load_real_data(real_data_path)
    if num_samples is None:
        num_samples = info["train_size"]

    print("=" * 70)
    n_num = X_num.shape[1] if X_num is not None else 0
    n_cat = X_cat.shape[1] if X_cat is not None else 0
    print(f"[SMOTE] 运行 | k={k_neighbours}  n_samples={num_samples}")
    print(f"  数据: train={info['train_size']}  (num={n_num}, cat={n_cat})")
    t0 = time.time()

    # 将连续特征直接做 SMOTE；分类特征随机从真实数据中采样
    from imblearn.over_sampling import SMOTE as SMOTEModel

    if X_num is not None and X_num.shape[1] > 0:
        # SMOTE 需要至少 2 个类。对回归任务，简单按 y 分箱
        y_binned = np.digitize(y, np.percentile(y, [25, 50, 75]))
        try:
            smote = SMOTEModel(k_neighbors=min(k_neighbours, 3), random_state=0)
            _resample_result = smote.fit_resample(X_num, y_binned)
            X_num_res = _resample_result[0]
            # 只取 num_samples 条
            idx = np.random.RandomState(0).choice(len(X_num_res), size=num_samples, replace=True)
            X_num_out = X_num_res[idx]
        except Exception:
            # fallback: 随机重采样
            idx = np.random.RandomState(0).choice(len(X_num), size=num_samples, replace=True)
            X_num_out = X_num[idx]
    else:
        idx = np.random.RandomState(0).choice(len(y), size=num_samples, replace=True)
        X_num_out = None

    # 分类特征：随机重采样
    if X_cat is not None:
        idx_cat = np.random.RandomState(0).choice(len(X_cat), size=num_samples, replace=True)
        X_cat_out = X_cat[idx_cat]
    else:
        X_cat_out = None

    # y: 随机重采样
    idx_y = np.random.RandomState(0).choice(len(y), size=num_samples, replace=True)
    y_out = y[idx_y]

    save_synthetic(parent_dir, (X_num_out, X_cat_out, y_out), real_data_path=real_data_path)
    t_total = time.time() - t0
    print(f"[SMOTE] 完成, 用时 {t_total:.1f}s")

    eval_report = run_catboost_eval(parent_dir, real_data_path=real_data_path)
    return {"total_time": t_total, "catboost": eval_report}


# ==================================
# DDPM MLP (TabDDPM 原始管线)
# ==================================
def run_ddpm_mlp(exp_base=EXP_BASE, steps=5000, lr=1e-3, batch_size=1024, num_samples=None):
    """运行原始 TabDDPM MLP 基线。"""
    parent_dir = f"{exp_base}/ddpm_mlp"
    config_path = f"{parent_dir}/config.toml"

    print("=" * 70)
    print(f"[DDPM_MLP] 训练 | steps={steps}  lr={lr}  batch={batch_size}")

    # 使用标准 TabDDPM pipeline
    from scripts.pipeline import main as pipeline_main
    sys.argv = [
        "pipeline",
        "--config", config_path,
        "--train",
        "--sample",
        "--eval",
    ]
    try:
        pipeline_main()
    except SystemExit:
        pass

    # 读取结果
    results_path = os.path.join(parent_dir, "results_catboost.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {"status": "completed"}


# ==================================
# 综合评估
# ==================================
def run_comprehensive_eval(real_data_path=REAL_DATA, exp_base=EXP_BASE, models=None):
    """对所有已生成的基线模型运行 evaluate_v3 综合评估。"""
    from scripts.evaluate_v3 import evaluate_all

    model_dirs = {}
    for name in (models or ["tvae", "ctgan", "smote", "ddpm_mlp", "causal_m4_v6"]):
        d = os.path.join(exp_base, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "y_train.npy")):
            model_dirs[name] = d

    if not model_dirs:
        print("❌ 没有找到已生成数据的模型")
        return {}

    print(f"\n📋 综合评估 {len(model_dirs)} 个模型: {list(model_dirs.keys())}")
    results = evaluate_all(real_data_path, model_dirs, exp_base)
    return results


# ==================================
# main
# ==================================
def main():
    parser = argparse.ArgumentParser(description="v3 基线模型统一运行器")
    parser.add_argument("--model", choices=["tvae", "ctgan", "smote", "ddpm_mlp", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=300, help="TVAE/CTGAN epochs")
    parser.add_argument("--steps", type=int, default=5000, help="DDPM_MLP steps")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--real_data", type=str, default=REAL_DATA,
                        help="真实数据目录 (TabDDPM npy 格式)")
    parser.add_argument("--exp_base", type=str, default=EXP_BASE,
                        help="实验输出根目录")
    parser.add_argument("--eval_only", action="store_true",
                        help="跳过训练，只运行综合评估")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过综合评估")
    args = parser.parse_args()

    results = {}
    real_data_path = args.real_data
    exp_base = args.exp_base

    if args.eval_only:
        run_comprehensive_eval(real_data_path=real_data_path, exp_base=exp_base)
        return

    if args.model in ("tvae", "all"):
        try:
            results["tvae"] = run_tvae(
                real_data_path=real_data_path,
                exp_base=exp_base,
                epochs=args.epochs,
                num_samples=args.num_samples,
            )
        except Exception as e:
            print(f"❌ TVAE 失败: {e}")
            import traceback; traceback.print_exc()

    if args.model in ("ctgan", "all"):
        try:
            results["ctgan"] = run_ctgan(
                real_data_path=real_data_path,
                exp_base=exp_base,
                epochs=args.epochs,
                num_samples=args.num_samples,
            )
        except Exception as e:
            print(f"❌ CTGAN 失败: {e}")
            import traceback; traceback.print_exc()

    if args.model in ("smote", "all"):
        try:
            results["smote"] = run_smote(
                real_data_path=real_data_path,
                exp_base=exp_base,
                num_samples=args.num_samples,
            )
        except Exception as e:
            print(f"❌ SMOTE 失败: {e}")
            import traceback; traceback.print_exc()

    if args.model in ("ddpm_mlp", "all"):
        try:
            results["ddpm_mlp"] = run_ddpm_mlp(
                exp_base=exp_base,
                steps=args.steps,
                num_samples=args.num_samples,
            )
        except Exception as e:
            print(f"❌ DDPM_MLP 失败: {e}")
            import traceback; traceback.print_exc()

    # 保存基线结果汇总
    os.makedirs(exp_base, exist_ok=True)
    summary_path = f"{exp_base}/baseline_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 基线结果: {summary_path}")

    # 综合评估
    if not args.skip_eval:
        run_comprehensive_eval(real_data_path=real_data_path, exp_base=exp_base)

    print("\n" + "=" * 70)
    print("✅ v3 基线实验完成！")
    for name, res in results.items():
        t = res.get("total_time", res.get("train_time", "?"))
        print(f"  {name.upper()}: {t}s")


if __name__ == "__main__":
    main()
