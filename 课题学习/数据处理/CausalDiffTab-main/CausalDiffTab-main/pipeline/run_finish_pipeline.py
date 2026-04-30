"""
训练完成后的收尾管线：
  1. 运行外部 baseline (CTGAN / TVAE / SMOTE, full tier)
  2. 统一评估（域内 2024 + 迁移 2025）
  3. 输出对比报告到 results/eval_full_compare.md
"""
from __future__ import annotations

import subprocess
import sys
import json
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
    # ── Step 1: External baselines ──────────────────────────────────
    run(
        [PYTHON, str(CDT_ROOT / "pipeline" / "run_external_baselines.py"), "--tier", "full"],
        "外部 baseline: CTGAN + TVAE + SMOTE (full tier, 10000 rows)",
    )

    # ── Step 2a: Evaluate 域内 2024 ─────────────────────────────────
    glob_pattern = "*_full.csv"
    run(
        [
            PYTHON, str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", str(CDT_ROOT / "data" / "nyc_crash" / "test.csv"),
            "--synthetic_dir", str(CDT_ROOT / "results" / "synthetic"),
            "--file_glob", glob_pattern,
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
            "--output_tag", "full_2024",
        ],
        "评估 2024 域内 (real_test=nyc_crash/test.csv)",
    )

    # ── Step 2b: Evaluate 迁移 2025 ─────────────────────────────────
    run(
        [
            PYTHON, str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test", str(CDT_ROOT / "data" / "nyc_crash_2025" / "test.csv"),
            "--synthetic_dir", str(CDT_ROOT / "results" / "synthetic"),
            "--file_glob", glob_pattern,
            "--task_type", "regression",
            "--primary_metrics_profile", "no_rule",
            "--output_tag", "full_2025",
        ],
        "评估 2025 迁移 (real_test=nyc_crash_2025/test.csv)",
    )

    # ── Step 3: 汇总报告 ─────────────────────────────────────────────
    report_2024 = CDT_ROOT / "results" / "eval_report_full_2024.md"
    report_2025 = CDT_ROOT / "results" / "eval_report_full_2025.md"

    lines = [
        f"# 全量实验评估报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        "## 训练配置",
        "| 模型 | lambda | 因果掩码 | 分层 |",
        "| ---- | ------ | -------- | ---- |",
        "| macro_soft_2024 | 0.3 | ✅ | ✅ |",
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

    summary_path = CDT_ROOT / "results" / "eval_full_compare.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] 综合对比报告: {summary_path}")

    # 也更新 eval_report_latest.md
    latest = CDT_ROOT / "results" / "eval_report_latest.md"
    latest.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] eval_report_latest.md 已更新")


if __name__ == "__main__":
    main()
