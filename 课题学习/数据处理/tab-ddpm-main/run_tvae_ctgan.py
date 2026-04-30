"""
运行 TVAE / CTGAN 基线模型: 训练 → 采样 → CatBoost 评估
使用已安装的 ctgan 0.11.x (CTGAN, TVAE 类)
"""

import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

import lib
from scripts.eval_catboost import train_catboost


# ---------- 通用辅助 ----------
def load_real_data(real_data_path):
    """加载真实训练数据，返回 (X_num, X_cat, y, info)"""
    rp = Path(real_data_path)
    X_num, X_cat, y = lib.read_pure_data(rp, 'train')
    info = lib.load_json(rp / 'info.json')
    return X_num, X_cat, y, info


def build_dataframe(X_num, X_cat, y, info):
    """拼合为 pandas DataFrame，并返回离散列名列表"""
    import pandas as pd
    n_num = X_num.shape[1] if X_num is not None else 0
    n_cat = X_cat.shape[1] if X_cat is not None else 0

    cols_num = [f"num_{i}" for i in range(n_num)]
    cols_cat = [f"cat_{i}" for i in range(n_cat)]

    parts = []
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


def save_synthetic(parent_dir, df, cols_num, cols_cat, info):
    """将生成的 DataFrame 拆分保存为 npy"""
    parent_dir = Path(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)

    if cols_num:
        X_num = df[cols_num].values.astype(float)
        np.save(parent_dir / "X_num_train.npy", X_num)
    if cols_cat:
        X_cat = df[cols_cat].values.astype(str)
        np.save(parent_dir / "X_cat_train.npy", X_cat)

    y = df["y"].values.astype(float)
    if info["task_type"] != "regression":
        y = y.astype(int)
    # 防止 y 只有一个类别
    if len(np.unique(y)) == 1:
        y[0] = 0
        y[1] = 1
    np.save(parent_dir / "y_train.npy", y)

    # 复制 info.json
    import shutil
    src = Path(info["_real_data_path"]) / "info.json"
    dst = parent_dir / "info.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


# ---------- TVAE ----------
def run_tvae(real_data_path, parent_dir, epochs=300, batch_size=512, lr=1e-3,
             num_samples=159990, seed=0):
    from ctgan import TVAE
    print("=" * 80)
    print(f"[TVAE] 开始训练 | epochs={epochs}  batch_size={batch_size}  lr={lr}")
    t0 = time.time()

    X_num, X_cat, y, info = load_real_data(real_data_path)
    info["_real_data_path"] = real_data_path
    df, discrete_columns, cols_num, cols_cat = build_dataframe(X_num, X_cat, y, info)

    print(f"  数据形状: {df.shape}  (连续 {len(cols_num)}, 分类 {len(cols_cat)})")
    print(f"  离散列数: {len(discrete_columns)}")

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

    # 采样
    print(f"[TVAE] 采样 {num_samples} 条...")
    syn = model.sample(num_samples)
    save_synthetic(parent_dir, syn, cols_num, cols_cat, info)
    t_sample = time.time() - t0
    print(f"[TVAE] 采样完成, 总用时 {t_sample:.1f}s")

    return t_train, t_sample


# ---------- CTGAN ----------
def run_ctgan(real_data_path, parent_dir, epochs=300, batch_size=500,
              num_samples=159990, seed=0):
    from ctgan import CTGAN
    print("=" * 80)
    print(f"[CTGAN] 开始训练 | epochs={epochs}  batch_size={batch_size}")
    t0 = time.time()

    X_num, X_cat, y, info = load_real_data(real_data_path)
    info["_real_data_path"] = real_data_path
    df, discrete_columns, cols_num, cols_cat = build_dataframe(X_num, X_cat, y, info)

    print(f"  数据形状: {df.shape}  (连续 {len(cols_num)}, 分类 {len(cols_cat)})")
    print(f"  离散列数: {len(discrete_columns)}")

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

    # 采样
    print(f"[CTGAN] 采样 {num_samples} 条...")
    syn = model.sample(num_samples)
    save_synthetic(parent_dir, syn, cols_num, cols_cat, info)
    t_sample = time.time() - t0
    print(f"[CTGAN] 采样完成, 总用时 {t_sample:.1f}s")

    return t_train, t_sample


# ---------- CatBoost 评估 ----------
def run_eval(parent_dir, real_data_path, seed=0):
    print(f"\n[EVAL] CatBoost Train-on-Synthetic for {parent_dir}")
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


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tvae", "ctgan", "both"], default="both")
    parser.add_argument("--data", default="data/nyc_crash_c4")
    parser.add_argument("--tvae_dir", default="exp/nyc_crash_c4/tvae")
    parser.add_argument("--ctgan_dir", default="exp/nyc_crash_c4/ctgan")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=159990)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {}

    if args.model in ("tvae", "both"):
        t_train, t_total = run_tvae(
            real_data_path=args.data,
            parent_dir=args.tvae_dir,
            epochs=args.epochs,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        eval_report = run_eval(args.tvae_dir, args.data, args.seed)
        results["tvae"] = {"train_time": t_train, "total_time": t_total}

    if args.model in ("ctgan", "both"):
        t_train, t_total = run_ctgan(
            real_data_path=args.data,
            parent_dir=args.ctgan_dir,
            epochs=args.epochs,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        eval_report = run_eval(args.ctgan_dir, args.data, args.seed)
        results["ctgan"] = {"train_time": t_train, "total_time": t_total}

    print("\n" + "=" * 80)
    print("全部完成！")
    for name, t in results.items():
        print(f"  {name.upper()}: 训练 {t['train_time']:.1f}s, 总计 {t['total_time']:.1f}s")


if __name__ == "__main__":
    main()
