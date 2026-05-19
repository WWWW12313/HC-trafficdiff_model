"""
v2 完整 Benchmark 运行器
=========================
统一运行 2024 45-col schema (nyc_crash_2024_v2) 上的所有基线和消融实验：

基线模型 (Baselines)：
  - CTGAN          (run_external_baselines.py)
  - TVAE           (run_external_baselines.py)
  - SMOTE          (run_external_baselines.py)
  - TabDDPM (纯扩散)  (run_all_experiments.py --model baseline_tabddpm_v2)

消融实验 (Ablation)：
  - 消融因果       (run_all_experiments.py --model ablation_no_causal_v2)
  - 消融分层       (run_all_experiments.py --model ablation_no_hierarchy_v2)

完全体 (Ours)：
  - macro_soft_2024_v2  (已有 ckpt，默认跳过训练)

评估：
  - evaluate_all.py 对每个模型输出统一评估，输出对比表

使用示例：
  # 完整运行（含训练）
  python pipeline/run_v2_benchmark.py --device cuda:0

  # 跳过训练，仅采样+评估
  python pipeline/run_v2_benchmark.py --skip_train --device cuda:0

  # 仅评估（需已有所有 .csv 文件）
  python pipeline/run_v2_benchmark.py --eval_only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

CDT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DATANAME = "nyc_crash_2024_v2"
TIER = "full"
DEVICE = "cuda:0"

SYN_DIR = CDT_ROOT / "results" / "synthetic"
REAL_TRAIN = CDT_ROOT / "data" / DATANAME / "train.csv"
REAL_TEST  = CDT_ROOT / "data" / DATANAME / "test.csv"
INFO_JSON  = CDT_ROOT / "data" / DATANAME / "info.json"

# 已有的完全体采样结果（上次已运行完毕，直接复制到标准命名位置）
EXISTING_OURS_CSV = SYN_DIR / "macro_soft_2024_v2_full_samples_physical.csv"
OURS_FINAL_CSV    = SYN_DIR / "macro_soft_2024_v2_full.csv"

# (model_tag, output_csv, note)
ALL_MODELS: list[tuple[str, Path, str]] = [
    ("baseline_ctgan",          SYN_DIR / "baseline_ctgan_full.csv",         "外部基线 CTGAN"),
    ("baseline_tvae",           SYN_DIR / "baseline_tvae_full.csv",          "外部基线 TVAE"),
    ("baseline_smote",          SYN_DIR / "baseline_smote_full.csv",         "外部基线 SMOTE"),
    ("baseline_tabddpm_v2",     SYN_DIR / "baseline_tabddpm_v2_full.csv",    "纯扩散 TabDDPM v2"),
    ("ablation_no_causal_v2",   SYN_DIR / "ablation_no_causal_v2_full.csv",  "消融：无因果"),
    ("ablation_no_hierarchy_v2",SYN_DIR / "ablation_no_hierarchy_v2_full.csv","消融：无分层"),
    ("macro_soft_2024_v2",      OURS_FINAL_CSV,                              "完全体 (Ours)"),
]


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print("="*60)
    subprocess.check_call(cmd, cwd=str(CDT_ROOT))


def run_external_baselines(skip: bool, seed: int) -> None:
    if skip:
        print("[skip] 外部基线 CTGAN/TVAE/SMOTE")
        return
    _run([
        PYTHON,
        str(CDT_ROOT / "pipeline" / "run_external_baselines.py"),
        "--tier", TIER,
        "--dataname", DATANAME,
        "--seed", str(seed),
    ], "外部基线：CTGAN / TVAE / SMOTE")


def run_diffusion_model(model: str, skip_train: bool, skip_sample: bool, device: str) -> None:
    cmd = [
        PYTHON,
        str(CDT_ROOT / "pipeline" / "run_all_experiments.py"),
        "--model", model,
        "--tier", TIER,
        "--device", device,
        "--dataname", DATANAME,
        "--synthetic_dir", str(SYN_DIR),
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_sample:
        cmd.append("--skip_sample")
    _run(cmd, f"扩散模型：{model}")


def prepare_ours_result() -> None:
    """将已有的完全体采样结果复制到标准命名位置。"""
    if OURS_FINAL_CSV.is_file():
        print(f"[skip] 完全体结果已存在：{OURS_FINAL_CSV}")
        return
    if EXISTING_OURS_CSV.is_file():
        shutil.copy2(EXISTING_OURS_CSV, OURS_FINAL_CSV)
        print(f"[copy] {EXISTING_OURS_CSV.name} -> {OURS_FINAL_CSV.name}")
    else:
        print(f"[warn] 完全体采样结果不存在，将重新运行采样: {EXISTING_OURS_CSV}")
        run_diffusion_model("macro_soft_2024_v2", skip_train=True, skip_sample=False, device=DEVICE)


def run_evaluation(output_tag: str) -> dict:
    """运行 evaluate_all.py，返回解析后的结果 dict。"""
    import glob
    # 收集所有模型的 csv glob pattern
    all_csvs = [m[1] for m in ALL_MODELS if m[1].is_file()]
    if not all_csvs:
        print("[warn] 没有找到任何合成 CSV，跳过评估")
        return {}

    # evaluate_all 对每个文件单独运行，收集结果
    results = {}
    for model_tag, csv_path, note in ALL_MODELS:
        if not csv_path.is_file():
            print(f"[skip eval] 文件不存在: {csv_path.name}")
            continue
        tag = f"{output_tag}_{model_tag}"
        cmd = [
            PYTHON,
            str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test",    str(REAL_TEST),
            "--synthetic_dir", str(SYN_DIR),
            "--info_json",    str(INFO_JSON),
            "--file_glob",    csv_path.name,
            "--output_tag",   tag,
        ]
        print(f"\n[eval] {model_tag} ({note})")
        try:
            subprocess.check_call(cmd, cwd=str(CDT_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"[eval error] {model_tag}: {e}")
            continue

        # 尝试读取 JSON 结果
        result_json = CDT_ROOT / "results" / f"eval_report_{tag}.json"
        if result_json.is_file():
            with open(result_json, encoding="utf-8") as f:
                data = json.load(f)
            # 结果在 rows[0] 中（evaluate_all.py 格式）
            row = (data.get("rows") or [{}])[0]
            def _f(v):
                try:
                    return round(float(v), 4)
                except (TypeError, ValueError):
                    return float("nan")
            results[model_tag] = {
                "note":        note,
                "avg_score":   _f(row.get("tstr_avg_score")),
                "best_model":  row.get("tstr_best_model", ""),
                "best_score":  _f(row.get("tstr_best_model_score")),
                "r2":          _f(row.get("tstr_r2")),
                "mse":         _f(row.get("tstr_mse")),
                "mae":         _f(row.get("tstr_mae")),
                "wasserstein": _f(row.get("mean_wasserstein_numeric")),
                "js_div":      _f(row.get("mean_js_categorical")),
            }
    return results


def print_comparison_table(results: dict) -> None:
    if not results:
        return
    print("\n" + "="*100)
    print("  v2 Benchmark 对比结果 (nyc_crash_2024_v2, full tier, 10000 samples)")
    print("="*100)
    hdr = f"{'模型':<30} {'说明':<22} {'TSTR↑':>8} {'Best↑':>8} {'R²↑':>7} {'MSE↓':>7} {'WD↓':>8} {'JS↓':>8}"
    print(hdr)
    print("-"*100)
    for model_tag, r in results.items():
        print(
            f"{model_tag:<30} {r['note']:<22} "
            f"{r['avg_score']:>8.4f} {r['best_score']:>8.4f} "
            f"{r['r2']:>7.3f} {r['mse']:>7.3f} "
            f"{r['wasserstein']:>8.4f} {r['js_div']:>8.4f}"
        )
    print("="*100)


def save_comparison_table(results: dict, output_tag: str) -> Path:
    out = CDT_ROOT / "results" / f"benchmark_comparison_{output_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] 对比结果 -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 完整 Benchmark 运行器")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="跳过所有模型训练（需已有 checkpoint）",
    )
    parser.add_argument(
        "--skip_external",
        action="store_true",
        help="跳过外部基线 CTGAN/TVAE/SMOTE（需已有对应 CSV）",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅运行评估（跳过所有训练和采样，需已有所有合成 CSV）",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="跳过评估步骤（仅训练+采样）",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "指定只运行哪些模型（空格分隔）。可选值："
            " baseline_tabddpm_v2 ablation_no_causal_v2 ablation_no_hierarchy_v2 macro_soft_2024_v2"
        ),
    )
    parser.add_argument("--output_tag", type=str, default="v2_benchmark")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    skip_train  = args.skip_train or args.eval_only
    skip_sample = args.eval_only

    # 目标扩散模型列表（可通过 --models 过滤）
    diffusion_models = ["baseline_tabddpm_v2", "ablation_no_causal_v2", "ablation_no_hierarchy_v2"]
    if args.models:
        diffusion_models = [m for m in diffusion_models if m in args.models]
        run_ours = "macro_soft_2024_v2" in args.models
    else:
        run_ours = True

    # ── Step 1: 外部基线 ──────────────────────────────────────────────
    skip_ext = args.skip_external or args.eval_only
    run_external_baselines(skip=skip_ext, seed=args.seed)

    # ── Step 2: 扩散消融模型 ─────────────────────────────────────────
    for model in diffusion_models:
        run_diffusion_model(
            model,
            skip_train=skip_train,
            skip_sample=skip_sample,
            device=args.device,
        )

    # ── Step 3: 完全体结果（已有，复制到标准位置）────────────────────
    if run_ours:
        prepare_ours_result()

    # ── Step 4: 统一评估 ─────────────────────────────────────────────
    if not args.skip_eval:
        results = run_evaluation(output_tag=args.output_tag)
        print_comparison_table(results)
        if results:
            save_comparison_table(results, output_tag=args.output_tag)
    else:
        print("[skip] 评估步骤已跳过")

    print("\n[DONE] v2 Benchmark 全部完成！")


if __name__ == "__main__":
    main()
