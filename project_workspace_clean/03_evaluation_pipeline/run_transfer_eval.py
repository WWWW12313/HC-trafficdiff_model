"""
一键批量迁移评估脚本（v2 schema 专用）。

自动收集 synthetic_dir 中所有 v2 full-tier 合成文件（含内部模型与外部基线），
统一在指定真实测试集（默认 2025 迁移集）上跑 evaluate_all.py，
输出一份合并的迁移评估报告。

用法示例:
  # 默认：评估所有 v2 full 文件在 2025 迁移集上的表现
  python pipeline/run_transfer_eval.py

  # 指定自定义测试集与输出标签
  python pipeline/run_transfer_eval.py \
    --real_test data/nyc_crash_2025_v2/test.csv \
    --output_tag v2_transfer_2025 \
    --primary_metrics_profile no_rule

  # 仅评估指定模型子集
  python pipeline/run_transfer_eval.py \
    --include_patterns "*_v2_full.csv" "baseline_*_full.csv"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

CDT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INTERNAL_PATTERNS = ["*_v2_full.csv"]
DEFAULT_BASELINE_PATTERNS = ["baseline_ctgan_full.csv", "baseline_tvae_full.csv", "baseline_smote_full.csv"]


def _collect_files(synthetic_dir: Path, patterns: List[str]) -> List[Path]:
    """按 glob 模式收集文件，去重并按文件名排序。"""
    seen = set()
    files: List[Path] = []
    for pat in patterns:
        for fp in sorted(synthetic_dir.glob(pat)):
            if fp.is_file() and not fp.name.startswith("_") and fp.resolve() not in seen:
                files.append(fp)
                seen.add(fp.resolve())
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transfer evaluation for v2 schema")
    parser.add_argument(
        "--real_test",
        type=str,
        default="data/nyc_crash_2025_v2/test.csv",
        help="真实测试集 CSV 路径（迁移目标域）",
    )
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default="results/synthetic",
        help="合成数据所在目录",
    )
    parser.add_argument(
        "--info_json",
        type=str,
        default=None,
        help="评估用的 info.json 路径；v2 实验建议传 data/nyc_crash_2024_v2/info.json",
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="+",
        default=None,
        help="覆盖默认的 include glob 模式列表",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="+",
        default=["*_debug_*", "*_smoke_*", "*_compare_*"],
        help="排除的 glob 模式列表",
    )
    parser.add_argument(
        "--primary_metrics_profile",
        type=str,
        default="no_rule",
        choices=["auto", "structural", "downstream", "no_rule", "full"],
        help="主指标 profile，迁移评估建议用 no_rule",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default="v2_transfer_2025",
        help="输出报告文件名标签",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=["regression", "classification"],
        help="覆盖 info.json 中的任务类型",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印会评估哪些文件，不实际运行",
    )
    args = parser.parse_args()

    syn_dir = Path(args.synthetic_dir)
    if not syn_dir.is_absolute():
        syn_dir = CDT_ROOT / syn_dir

    # 确定 include 模式
    if args.include_patterns:
        patterns = args.include_patterns
    else:
        patterns = DEFAULT_INTERNAL_PATTERNS + DEFAULT_BASELINE_PATTERNS

    # 收集文件
    files = _collect_files(syn_dir, patterns)

    # 应用排除模式
    if args.exclude_patterns:
        import fnmatch
        filtered = []
        for fp in files:
            exclude = False
            for pat in args.exclude_patterns:
                if fnmatch.fnmatch(fp.name, pat):
                    exclude = True
                    break
            if not exclude:
                filtered.append(fp)
        files = filtered

    if not files:
        print(f"[warn] 在 {syn_dir} 中未找到匹配 {patterns} 的合成文件")
        sys.exit(0)

    print(f"[transfer_eval] 发现 {len(files)} 个合成文件:")
    for fp in files:
        print(f"  - {fp.name}")

    if args.dry_run:
        print("[dry_run] 已结束，未实际运行评估")
        sys.exit(0)

    # 创建临时目录，复制所有文件（避免 glob 冲突，Windows 下 symlink 需要权限）
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for fp in files:
            dst = tmp_path / fp.name
            shutil.copy2(fp, dst)
            print(f"[copy] {fp.name}")

        real_test_path = Path(args.real_test)
        if not real_test_path.is_absolute():
            real_test_path = CDT_ROOT / real_test_path

        cmd = [
            sys.executable,
            str(CDT_ROOT / "pipeline" / "evaluate_all.py"),
            "--real_test",
            str(real_test_path),
            "--synthetic_dir",
            str(tmp_path),
            "--file_glob",
            "*.csv",
            "--primary_metrics_profile",
            args.primary_metrics_profile,
            "--output_tag",
            args.output_tag,
        ]
        if args.info_json:
            info_json_path = Path(args.info_json)
            if not info_json_path.is_absolute():
                info_json_path = CDT_ROOT / info_json_path
            cmd.extend(["--info_json", str(info_json_path)])
        if args.task_type:
            cmd.extend(["--task_type", args.task_type])

        print(f"[run] {' '.join(cmd)}")
        subprocess.check_call(cmd)

    # 报告输出位置
    out_dir = CDT_ROOT / "results"
    md_path = out_dir / f"eval_report_{args.output_tag}.md"
    json_path = out_dir / f"eval_report_{args.output_tag}.json"
    print(f"\n[done] 迁移评估报告:")
    print(f"  {md_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
