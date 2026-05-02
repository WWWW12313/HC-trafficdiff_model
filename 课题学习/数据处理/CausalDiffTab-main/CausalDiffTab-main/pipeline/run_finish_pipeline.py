"""
训练完成后的 2024→2025 迁移收尾管线：
    1. 运行外部 baseline (CTGAN / TVAE / SMOTE, full tier, source=2024)
    2. 统一评估（2024 域内 + 2025 迁移）
        3. 输出对比报告到 results/eval_full_source2024_compare.md
"""
from __future__ import annotations

import subprocess
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

CDT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"[Step] {step}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(CDT_ROOT))
    if result.returncode != 0:
        print(f"[WARN] {step} returned exit code {result.returncode}, continuing...")


def main() -> None:
    parser = argparse.ArgumentParser(description="2024 源域训练后的 full-tier 收尾评估")
    parser.add_argument("--dataname", default="nyc_crash_2024", help="训练源域数据目录名")
    parser.add_argument("--target_year", default="2025", help="迁移测试年份标签")
    parser.add_argument("--tier", default="full", choices=["quick", "balanced", "full"])
    parser.add_argument("--synthetic_dir", default=str(CDT_ROOT / "results" / "synthetic_2024_source"))
    args = parser.parse_args()

    source_test = CDT_ROOT / "data" / args.dataname / "test.csv"
    target_test = CDT_ROOT / "data" / f"nyc_crash_{args.target_year}" / "test.csv"
    info_json = CDT_ROOT / "data" / args.dataname / "info.json"
    synthetic_dir = Path(args.synthetic_dir)
    if not synthetic_dir.is_absolute():
        synthetic_dir = CDT_ROOT / synthetic_dir

    # ── Step 1: External baselines ──────────────────────────────────
    run(
        [
            PYTHON, str(CDT_ROOT / "pipeline" / "run_external_baselines.py"),
            "--tier", args.tier,
            "--dataname", args.dataname,
            "--synthetic_dir", str(synthetic_dir),
        ],
        f"外部 baseline: CTGAN + TVAE + SMOTE ({args.tier} tier, source={args.dataname})",
    )

    # ── Step 2a: Evaluate 域内 2024 ─────────────────────────────────
    glob_pattern = f"*_{args.tier}.csv"
    run(
        [
            PYTHON, str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", str(source_test),
            "--synthetic_dir", str(synthetic_dir),
            "--file_glob", glob_pattern,
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
            "--info_json", str(info_json),
            "--output_tag", f"{args.tier}_source2024_in_domain",
        ],
        f"评估 2024 域内 (real_test={args.dataname}/test.csv)",
    )

    # ── Step 2b: Evaluate 迁移 2025 ─────────────────────────────────
    run(
        [
            PYTHON, str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", str(target_test),
            "--synthetic_dir", str(synthetic_dir),
            "--file_glob", glob_pattern,
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
            "--info_json", str(info_json),
            "--output_tag", f"{args.tier}_source2024_transfer{args.target_year}",
        ],
        f"评估 {args.target_year} 迁移 (real_test=nyc_crash_{args.target_year}/test.csv)",
    )

    # ── Step 3: 汇总报告 ─────────────────────────────────────────────
    report_2024 = CDT_ROOT / "results" / f"eval_report_{args.tier}_source2024_in_domain.md"
    report_2025 = CDT_ROOT / "results" / f"eval_report_{args.tier}_source2024_transfer{args.target_year}.md"

    lines = [
        f"# 全量实验评估报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        "## 训练配置",
        f"- source_train: `data/{args.dataname}/train.csv`",
        f"- source_test: `data/{args.dataname}/test.csv`",
        f"- transfer_test: `data/nyc_crash_{args.target_year}/test.csv`",
        f"- synthetic_dir: `{synthetic_dir}`",
        "",
        "| 模型 | lambda | 因果掩码 | 分层 |",
        "| ---- | ------ | -------- | ---- |",
        "| ours_full_model | 1.0 | ✅ | ✅ |",
        "| ablation_no_causal | 0.0 | ❌ | ✅ |",
        "| ablation_no_hierarchy | 1.0 | ✅ | ❌ |",
        "| baseline_tabddpm | 0.0 | ❌ | ❌ |",
        "| baseline_ctgan | — | — | — |",
        "| baseline_tvae | — | — | — |",
        "| baseline_smote | — | — | — |",
        "",
    ]

    for tag, rpath in [("2024 域内", report_2024), ("2025 迁移", report_2025)]:
        lines.append(f"## {tag} 评估结果")
        if rpath.is_file():
            lines.append(rpath.read_text(encoding="utf-8"))
        else:
            lines.append(f"> 报告文件未找到: {rpath}")
        lines.append("")

    summary_path = CDT_ROOT / "results" / f"eval_{args.tier}_source2024_compare.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] 综合对比报告: {summary_path}")

    # 也更新 eval_report_latest.md
    latest = CDT_ROOT / "results" / "eval_report_latest.md"
    latest.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] eval_report_latest.md 已更新")


if __name__ == "__main__":
    main()
