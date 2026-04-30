"""
V2 因果矩阵重建管线（自动化编排）

重建步骤：
  Step 1  仅重训 Stage3（Stage1 已用 v2 masks 完成，直接复用）
  Step 2  ours_full_model 采样 → results/synthetic/ours_full_model_full.csv
  Step 3  更新 compare_n10000 文件（需要更新的模型）
  Step 4  下游评测 ─ 回归（no_rule profile）
  Step 5  下游评测 ─ 分类（no_rule profile）
  Step 6  2025 迁移评测

用法：
  python pipeline/run_v2_rebuild.py
  python pipeline/run_v2_rebuild.py --skip_train       # 仅采样+评测（Stage3已训完）
  python pipeline/run_v2_rebuild.py --skip_train --skip_sample   # 仅更新compare+评测
  python pipeline/run_v2_rebuild.py --eval_only        # 仅重跑评测
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SYN_DIR = ROOT / "results" / "synthetic"

# 需要更新 compare 文件的模型 → 对应的 full.csv 来源
COMPARE_UPDATE = {
    "ours_full_model": 10000,
    "baseline_ctgan": 10000,
    "baseline_tvae": 10000,
    "baseline_smote": 10000,
}


def _run(cmd: list[str], desc: str = "") -> None:
    label = desc or " ".join(cmd[-3:])
    print(f"\n{'='*60}")
    print(f"[run] {label}")
    print(f"{'='*60}")
    start = time.time()
    subprocess.check_call(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    print(f"[done] {label}  ({elapsed/60:.1f} min)")


def step1_train_stage3(device: str = "cuda:0") -> None:
    """仅重训 Stage3 ours_full_model（Stage1 已完成）。"""
    print("\n>>> Step 1: 训练 Stage3 ours_full_model [full tier]")
    _run(
        [
            PYTHON, str(ROOT / "src" / "train_hierarchical.py"),
            "--stage", "3",
            "--tier", "full",
            "--experiment_id", "ours_full_model",
            "--lambda_causal", "1.0",
            "--device", device,
        ],
        "Stage3 full ours_full_model",
    )


def step2_sample_ours_full_model(device: str = "cuda:0", road_graphml: str = None, road_signals: str = None) -> None:
    """采样 ours_full_model → results/synthetic/ours_full_model_full.csv。"""
    print("\n>>> Step 2: 采样 ours_full_model [skip_train]")
    cmd = [
        PYTHON, str(ROOT / "pipeline" / "run_all_experiments.py"),
        "--model", "ours_full_model",
        "--tier", "full",
        "--device", device,
        "--skip_train",
    ]
    if road_graphml:
        cmd += ["--road_graphml", road_graphml]
    if road_signals:
        cmd += ["--road_signals", road_signals]
    _run(cmd, "sample ours_full_model")


def step3_update_compare_files() -> None:
    """把需要更新的 *_full.csv 拷贝为 *_compare_n10000.csv。"""
    print("\n>>> Step 3: 更新 compare_n10000 文件")
    for name, n in COMPARE_UPDATE.items():
        src = SYN_DIR / f"{name}_full.csv"
        dst = SYN_DIR / f"{name}_compare_n{n}.csv"
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  [copy] {src.name} → {dst.name}")
        else:
            print(f"  [warn] 源文件不存在，跳过: {src}")


def step4_eval_regression() -> None:
    """下游评测：回归，no_rule，file_glob = *_compare_n*.csv。"""
    print("\n>>> Step 4: 下游评测 ─ 回归（no_rule）")
    _run(
        [
            PYTHON, str(ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", "synthetic/nyc_crash/test.csv",
            "--file_glob", "*_compare_n*.csv",
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
        ],
        "evaluate regression no_rule",
    )
    # 保存带时间戳的副本
    _backup_eval_report("regression")


def step5_eval_classification() -> None:
    """下游评测：分类，no_rule，file_glob = *_compare_n*.csv。"""
    print("\n>>> Step 5: 下游评测 ─ 分类（no_rule）")
    _run(
        [
            PYTHON, str(ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", "synthetic/nyc_crash/test.csv",
            "--file_glob", "*_compare_n*.csv",
            "--task_type", "classification",
            "--primary_metrics_profile", "no_rule",
        ],
        "evaluate classification no_rule",
    )
    _backup_eval_report("classification")


def _backup_eval_report(tag: str) -> None:
    """将 eval_report_latest.* 备份为含任务标签的版本。"""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "results"
    for suffix in ["json", "md"]:
        src = results_dir / f"eval_report_latest.{suffix}"
        if src.is_file():
            dst = results_dir / f"eval_report_{tag}_{ts}.{suffix}"
            shutil.copy2(src, dst)
            print(f"  [backup] {dst.name}")


def step6_transfer_eval(postcovid_path: str = None) -> None:
    """迂移评测。"""
    print("\n>>> Step 6: 2025 迂移评测")
    if postcovid_path is None:
        postcovid_path = str(ROOT / "results" / "postcovid_2025_fully_enriched_like_2017.csv")
    postcovid = Path(postcovid_path)
    if not postcovid.is_file():
        # fallback 链：旧 like-2017 输出 → 旧 n82698 → n5000
        for _fallback in (
            ROOT / "results" / "postcovid_test_2025_n82698.csv",
            ROOT / "results" / "postcovid_test_2025_n5000.csv",
        ):
            if _fallback.is_file():
                postcovid = _fallback
                break
    if not postcovid.is_file():
        print(f"  [warn] 2025 测试集不存在: {postcovid}，跳过迁移评测")
        return
    _run(
        [
            PYTHON, str(ROOT / "pipeline" / "evaluate_postcovid_transfer.py"),
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
            "--postcovid_2025", str(postcovid),
        ],
        "evaluate postcovid 2025 transfer",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 因果矩阵重建管线")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过 Stage3 训练（Stage3 checkpoint 已存在时使用）")
    parser.add_argument("--skip_sample", action="store_true",
                        help="跳过采样（full.csv 已是最新时使用）")
    parser.add_argument("--skip_compare", action="store_true",
                        help="跳过更新 compare_n10000 文件")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅执行评测步骤（Step 4-6），跳过训练和采样")
    parser.add_argument("--device", default="cuda:0",
                        help="训练/采样设备，如 cuda:0 或 cpu")
    parser.add_argument("--postcovid_2025", default=None,
                        help="2025 迁移测试集路径，默认用 postcovid_2025_fully_enriched_like_2017.csv")
    parser.add_argument("--road_graphml", default=str(ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml"),
                        help="OSM graphml 路径，为采样斶路网 snap 提供圖数据")
    parser.add_argument("--road_signals", default=str(ROOT / "raw_data" / "osm" / "nyc_traffic_signals.geojson"),
                        help="信号灯 geojson 路径")
    args = parser.parse_args()

    if args.eval_only:
        args.skip_train = args.skip_sample = args.skip_compare = True

    if not args.skip_train:
        step1_train_stage3(args.device)

    if not args.skip_sample:
        step2_sample_ours_full_model(
            args.device,
            road_graphml=args.road_graphml,
            road_signals=args.road_signals,
        )

    if not args.skip_compare:
        step3_update_compare_files()

    step4_eval_regression()
    step5_eval_classification()
    step6_transfer_eval(args.postcovid_2025)

    print("\n" + "="*60)
    print("[全流程完成] V2 重建管线执行结束")
    print("="*60)


if __name__ == "__main__":
    main()
