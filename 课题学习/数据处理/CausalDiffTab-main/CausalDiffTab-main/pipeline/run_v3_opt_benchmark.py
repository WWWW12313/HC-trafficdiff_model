"""
v3 优化 Benchmark 运行器
========================
在 v2 Benchmark 结果基础上，新增因果正则强度灵敏度实验：
  - macro_soft_lam005_v2  (λ=0.05，弱因果约束)
  - macro_soft_lam01_v2   (λ=0.10，中等因果约束)

已有的 v2 模型结果（外部基线 + 消融 + 原始完全体）直接复用 JSON 缓存，不重复评估。

最终对比表包含 v2 全部 7 个模型 + 2 个新优化变体，共 9 个模型。

使用示例：
  # 完整运行（训练 + 采样 + 评估）
  python pipeline/run_v3_opt_benchmark.py --device cuda:0

  # 已训练好，只评估
  python pipeline/run_v3_opt_benchmark.py --eval_only

  # 跳过训练，重新采样 + 评估
  python pipeline/run_v3_opt_benchmark.py --skip_train --device cuda:0

目的：
  域内优化验证 — 确认弱化因果约束能否让完全体指标超过消融实验最优值
  (ablation_no_causal_v2: TSTR=0.7394, WD=0.2101)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CDT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DATANAME = "nyc_crash_2024_v2"
TIER = "full"

SYN_DIR = CDT_ROOT / "results" / "synthetic"
REAL_TEST = CDT_ROOT / "data" / DATANAME / "test.csv"
INFO_JSON  = CDT_ROOT / "data" / DATANAME / "info.json"
RESULTS_DIR = CDT_ROOT / "results"

# ─────────────────────────────────────────────
# v2 已有结果：直接复用 eval_report JSON，不重新评估
# ─────────────────────────────────────────────
V2_EXISTING_MODELS: list[tuple[str, str]] = [
    ("baseline_ctgan",           "外部基线 CTGAN"),
    ("baseline_tvae",            "外部基线 TVAE"),
    ("baseline_smote",           "外部基线 SMOTE"),
    ("baseline_tabddpm_v2",      "纯扩散 TabDDPM v2"),
    ("ablation_no_causal_v2",    "消融：无因果（分层+无λ）★最优"),
    ("ablation_no_hierarchy_v2", "消融：无分层（因果+无层级）"),
    ("macro_soft_2024_v2",       "完全体 v2 (λ=0.3)"),
]

# ─────────────────────────────────────────────
# 新实验：需要训练 + 采样 + 评估
# ─────────────────────────────────────────────
NEW_OPT_MODELS: list[tuple[str, Path, str]] = [
    ("macro_soft_lam005_v2",
     SYN_DIR / "macro_soft_lam005_v2_full.csv",
     "优化 Opt-A (λ=0.05)"),
    ("macro_soft_lam01_v2",
     SYN_DIR / "macro_soft_lam01_v2_full.csv",
     "优化 Opt-B (λ=0.10)"),
]

# v2 评估 JSON 前缀
V2_TAG_PREFIX = "v2_benchmark"
# v3 评估 JSON 前缀
V3_TAG_PREFIX = "v3_opt"


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print("="*60)
    subprocess.check_call(cmd, cwd=str(CDT_ROOT))


def _f(v) -> float:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return float("nan")


def load_v2_results() -> dict:
    """从 v2 benchmark 的 eval_report JSON 文件中读取已有结果。"""
    results = {}
    for model_tag, note in V2_EXISTING_MODELS:
        tag = f"{V2_TAG_PREFIX}_{model_tag}"
        result_json = RESULTS_DIR / f"eval_report_{tag}.json"
        if not result_json.is_file():
            print(f"[warn] v2 结果缺失: {result_json.name}，将跳过")
            continue
        with open(result_json, encoding="utf-8") as f:
            data = json.load(f)
        row = (data.get("rows") or [{}])[0]
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
        print(f"[load v2] {model_tag}: TSTR={results[model_tag]['avg_score']}, WD={results[model_tag]['wasserstein']}")
    return results


def run_new_opt_model(model_tag: str, skip_train: bool, skip_sample: bool, device: str) -> None:
    cmd = [
        PYTHON,
        str(CDT_ROOT / "pipeline" / "run_all_experiments.py"),
        "--model", model_tag,
        "--tier", TIER,
        "--device", device,
        "--dataname", DATANAME,
        "--synthetic_dir", str(SYN_DIR),
    ]
    if skip_train:
        cmd.append("--skip_train")
    if skip_sample:
        cmd.append("--skip_sample")
    _run(cmd, f"训练/采样：{model_tag}")


def evaluate_new_model(model_tag: str, csv_path: Path, note: str) -> dict | None:
    if not csv_path.is_file():
        print(f"[skip eval] 合成 CSV 不存在: {csv_path.name}")
        return None
    tag = f"{V3_TAG_PREFIX}_{model_tag}"
    cmd = [
        PYTHON,
        str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
        "--real_test",     str(REAL_TEST),
        "--synthetic_dir", str(SYN_DIR),
        "--info_json",     str(INFO_JSON),
        "--file_glob",     csv_path.name,
        "--output_tag",    tag,
    ]
    print(f"\n[eval] {model_tag} ({note})")
    try:
        subprocess.check_call(cmd, cwd=str(CDT_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"[eval error] {model_tag}: {e}")
        return None

    result_json = RESULTS_DIR / f"eval_report_{tag}.json"
    if not result_json.is_file():
        return None
    with open(result_json, encoding="utf-8") as f:
        data = json.load(f)
    row = (data.get("rows") or [{}])[0]
    return {
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


def print_comparison_table(results: dict, title: str = "v3 Opt Benchmark") -> None:
    if not results:
        return
    sep = "=" * 108
    print(f"\n{sep}")
    print(f"  {title}  (nyc_crash_2024_v2, full, 10000 samples)")
    print(f"  参考基准：ablation_no_causal_v2  TSTR=0.7394  WD=0.2101  JS=0.0094")
    print(sep)
    hdr = (f"{'模型':<32} {'说明':<26} {'TSTR↑':>8} {'Best↑':>8}"
           f" {'R²↑':>7} {'MSE↓':>7} {'MAE↓':>7} {'WD↓':>8} {'JS↓':>8}")
    print(hdr)
    print("-" * 108)
    # 按 TSTR 降序排列
    sorted_items = sorted(results.items(), key=lambda x: x[1].get("avg_score", 0), reverse=True)
    for model_tag, r in sorted_items:
        marker = "★" if r.get("avg_score", 0) == max(v.get("avg_score", 0) for v in results.values()) else " "
        print(
            f"{marker}{model_tag:<31} {r['note']:<26} "
            f"{r['avg_score']:>8.4f} {r['best_score']:>8.4f} "
            f"{r['r2']:>7.3f} {r['mse']:>7.3f} {r['mae']:>7.3f} "
            f"{r['wasserstein']:>8.4f} {r['js_div']:>8.4f}"
        )
    print(sep)


def save_results(results: dict, tag: str) -> Path:
    out = RESULTS_DIR / f"benchmark_comparison_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] 对比结果 -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 优化 Benchmark 运行器")
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--skip_train",  action="store_true", help="跳过训练（需已有 ckpt）")
    parser.add_argument("--eval_only",   action="store_true", help="仅评估（需已有所有 CSV）")
    parser.add_argument("--output_tag",  type=str, default="v3_opt_benchmark")
    args = parser.parse_args()

    # ── 1. 加载 v2 已有结果 ──────────────────────────────────────────────
    print("\n[Phase 1] 加载 v2 已有结果 ...")
    all_results = load_v2_results()

    # ── 2. 训练/采样新实验 ──────────────────────────────────────────────
    if not args.eval_only:
        print("\n[Phase 2] 训练/采样优化实验 ...")
        for model_tag, csv_path, note in NEW_OPT_MODELS:
            if csv_path.is_file():
                print(f"[skip] 合成 CSV 已存在: {csv_path.name}")
                continue
            run_new_opt_model(
                model_tag,
                skip_train=args.skip_train,
                skip_sample=False,
                device=args.device,
            )

    # ── 3. 评估新实验 ──────────────────────────────────────────────────
    print("\n[Phase 3] 评估新优化模型 ...")
    for model_tag, csv_path, note in NEW_OPT_MODELS:
        # 优先读缓存 JSON
        cached_json = RESULTS_DIR / f"eval_report_{V3_TAG_PREFIX}_{model_tag}.json"
        if cached_json.is_file():
            with open(cached_json, encoding="utf-8") as f:
                data = json.load(f)
            row = (data.get("rows") or [{}])[0]
            all_results[model_tag] = {
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
            print(f"[cache] {model_tag}: TSTR={all_results[model_tag]['avg_score']}, WD={all_results[model_tag]['wasserstein']}")
            continue

        r = evaluate_new_model(model_tag, csv_path, note)
        if r is not None:
            all_results[model_tag] = r

    # ── 4. 打印 + 保存对比表 ────────────────────────────────────────────
    print_comparison_table(all_results, title="v3 Opt Benchmark (域内 lambda 灵敏度分析)")
    save_results(all_results, args.output_tag)
    print("\n[DONE] v3 Opt Benchmark 完成！")


if __name__ == "__main__":
    main()
