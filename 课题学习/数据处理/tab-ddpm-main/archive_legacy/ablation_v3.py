"""
ablation_v3.py  —  v6 消融实验框架
====================================
消融维度:
  F1. 学习率调度: linear / cosine / warmup+cosine
  F2. 模型深度:   [512,512] / [768,768,768,768] / [1024,1024,1024,1024,1024]
  F3. 因果结构权重: 0.0 / 0.5 / 1.0 / 2.0
  F4. 时间步数:   500 / 1000 / 2000

每组消融训练后 → 采样 → CatBoost 评估 → 汇总到 ablation_results.json
"""

import argparse
import copy
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

# 确保可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================================
# 消融配置空间
# ==================================
ABLATION_CONFIGS = {
    "lr_schedule": {
        "param": "lr_scheduler",
        "values": ["linear", "cosine", "warmup_cosine"],
        "fixed": {"d_layers": [768, 768, 768, 768], "causal_weight": 1.0,
                  "num_timesteps": 1000, "steps": 3000},
    },
    "model_depth": {
        "param": "d_layers",
        "values": [
            [512, 512],
            [768, 768, 768, 768],
            [1024, 1024, 1024, 1024, 1024],
        ],
        "fixed": {"lr_scheduler": "cosine", "causal_weight": 1.0,
                  "num_timesteps": 1000, "steps": 3000},
    },
    "causal_weight": {
        "param": "causal_weight",
        "values": [0.0, 0.5, 1.0, 2.0],
        "fixed": {"d_layers": [768, 768, 768, 768], "lr_scheduler": "cosine",
                  "num_timesteps": 1000, "steps": 3000},
    },
    "num_timesteps": {
        "param": "num_timesteps",
        "values": [500, 1000, 2000],
        "fixed": {"d_layers": [768, 768, 768, 768], "lr_scheduler": "cosine",
                  "causal_weight": 1.0, "steps": 3000},
    },
}

REAL_DATA = "data/nyc_crash_v3"
EXP_BASE = "exp/nyc_crash_v3"


def build_train_config(ablation_dim, value, fixed_params):
    """构建单次消融实验的训练参数。"""
    cfg = {
        "data_dir": REAL_DATA,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "ema_rate": 0.999,
        "dropout": 0.0,
        "gaussian_loss_type": "mse",
        "scheduler": "cosine",
    }
    cfg.update(fixed_params)
    cfg[ablation_dim] = value
    return cfg


def run_single_ablation(cfg, exp_name, output_dir):
    """运行单次消融实验: 训练 + 采样 + 评估。"""
    from train_causal_v6 import train_causal_v6

    print(f"\n{'='*60}")
    print(f"[ABLATION] {exp_name}")
    print(f"  参数: {json.dumps({k: str(v) for k, v in cfg.items() if k != 'data_dir'}, indent=2)}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    try:
        # 训练
        train_causal_v6(
            data_dir=cfg["data_dir"],
            output_base=output_dir,
            steps=cfg.get("steps", 3000),
            lr_override=cfg.get("lr", 1e-3),
            batch_size_override=cfg.get("batch_size", 1024),
            d_layers_override=cfg.get("d_layers", [768, 768, 768, 768]),
            num_timesteps_override=cfg.get("num_timesteps", 1000),
            scheduler_override=cfg.get("scheduler", "cosine"),
            lr_scheduler_override=cfg.get("lr_scheduler", "cosine"),
            causal_weight_override=cfg.get("causal_weight", 1.0),
            dropout_override=cfg.get("dropout", 0.0),
            ema_rate_override=cfg.get("ema_rate", 0.999),
            weight_decay_override=cfg.get("weight_decay", 1e-5),
        )
        t_train = time.time() - t0

        # CatBoost 评估
        from scripts.eval_catboost import train_catboost
        T_dict = {
            "seed": 0, "normalization": None, "num_nan_policy": None,
            "cat_nan_policy": None, "cat_min_frequency": None,
            "cat_encoding": None, "y_policy": "default",
        }
        eval_report = train_catboost(
            parent_dir=output_dir,
            real_data_path=REAL_DATA,
            eval_type="synthetic",
            T_dict=T_dict,
            seed=0,
            change_val=False,
        )

        # 综合评估
        try:
            from scripts.evaluate_v3 import evaluate_model, load_info
            info = load_info(REAL_DATA)
            full_eval = evaluate_model(output_dir, REAL_DATA, info, model_name=exp_name)
        except Exception as e:
            full_eval = {"error": str(e)}

        t_total = time.time() - t0
        result = {
            "exp_name": exp_name,
            "config": {k: str(v) for k, v in cfg.items()},
            "train_time": round(t_train, 1),
            "total_time": round(t_total, 1),
            "catboost_eval": eval_report,
            "full_eval": full_eval,
            "status": "success",
        }

    except Exception as e:
        import traceback
        result = {
            "exp_name": exp_name,
            "config": {k: str(v) for k, v in cfg.items()},
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed",
        }
        print(f"❌ {exp_name} 失败: {e}")

    # 保存单次结果
    with open(os.path.join(output_dir, "ablation_result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


def run_ablation_dimension(dim_name, dim_config):
    """运行某一消融维度的全部实验。"""
    param = dim_config["param"]
    values = dim_config["values"]
    fixed = dim_config["fixed"]

    print(f"\n{'#'*70}")
    print(f"[ABLATION DIM] {dim_name} → {param}")
    print(f"  Values: {values}")
    print(f"{'#'*70}")

    dim_results = []
    for val in values:
        cfg = build_train_config(param, val, fixed)
        val_str = str(val).replace(", ", "_").replace("[", "").replace("]", "")
        exp_name = f"abl_{dim_name}_{val_str}"
        output_dir = os.path.join(EXP_BASE, "ablation", dim_name, exp_name)

        result = run_single_ablation(cfg, exp_name, output_dir)
        dim_results.append(result)

    return dim_results


def summarize_ablation(all_results, output_path):
    """汇总所有消融结果。"""
    summary = {}
    for dim_name, dim_results in all_results.items():
        summary[dim_name] = []
        for r in dim_results:
            entry = {
                "exp_name": r["exp_name"],
                "status": r["status"],
            }
            if r["status"] == "success":
                # 提取关键指标
                fe = r.get("full_eval", {})
                utility = fe.get("utility", {})

                t1 = utility.get("task1_regression", {})
                entry["test_rmse"] = t1.get("test_rmse", None)
                entry["test_r2"] = t1.get("test_r2", None)

                t2 = utility.get("task2_primary_cause", {})
                entry["cause_accuracy"] = t2.get("test_accuracy", None)
                entry["cause_weighted_f1"] = t2.get("test_weighted_f1", None)

                t3 = utility.get("task3_is_injury", {})
                entry["injury_auc"] = t3.get("test_auc", None)

                fidelity = fe.get("fidelity", {})
                entry["avg_js"] = fidelity.get("avg_js_divergence", None)

                entry["train_time"] = r.get("train_time", None)
            else:
                entry["error"] = r.get("error", "unknown")
            summary[dim_name].append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📁 消融结果汇总: {output_path}")

    # 打印汇总表
    print("\n" + "=" * 90)
    print("消融实验汇总")
    print("=" * 90)
    for dim_name, entries in summary.items():
        print(f"\n--- {dim_name} ---")
        header = f"{'实验':<35} {'RMSE':>8} {'R²':>8} {'Cause-Acc':>10} {'Cause-F1':>10} {'JS':>8}"
        print(header)
        print("-" * len(header))
        for e in entries:
            if e["status"] == "success":
                print(f"{e['exp_name']:<35} "
                      f"{e.get('test_rmse', 'N/A'):>8} "
                      f"{e.get('test_r2', 'N/A'):>8} "
                      f"{e.get('cause_accuracy', 'N/A'):>10} "
                      f"{e.get('cause_weighted_f1', 'N/A'):>10} "
                      f"{e.get('avg_js', 'N/A'):>8}")
            else:
                print(f"{e['exp_name']:<35} FAILED: {e.get('error', '')[:40]}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="v6 消融实验框架")
    parser.add_argument("--dim", choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                        default="all", help="消融维度")
    parser.add_argument("--steps", type=int, default=3000,
                        help="每组训练步数 (默认 3000, 快速测试用 500)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 500 步")
    args = parser.parse_args()

    if args.quick:
        for cfg in ABLATION_CONFIGS.values():
            cfg["fixed"]["steps"] = 500
    elif args.steps != 3000:
        for cfg in ABLATION_CONFIGS.values():
            cfg["fixed"]["steps"] = args.steps

    all_results = {}
    dims = list(ABLATION_CONFIGS.keys()) if args.dim == "all" else [args.dim]

    for dim in dims:
        dim_config = ABLATION_CONFIGS[dim]
        results = run_ablation_dimension(dim, dim_config)
        all_results[dim] = results

    # 汇总
    output_path = os.path.join(EXP_BASE, "ablation_results.json")
    summarize_ablation(all_results, output_path)

    print("\n✅ 消融实验全部完成！")


if __name__ == "__main__":
    main()
