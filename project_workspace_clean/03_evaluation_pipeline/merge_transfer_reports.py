"""
合并域内评估报告与迁移评估报告，生成统一的对比表格。

用法:
  python pipeline/merge_transfer_reports.py \
    --in_domain_json results/eval_report_v2_in_domain_2024.json \
    --transfer_json results/eval_report_v2_transfer_2025.json \
    --output results/transfer_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_row(rows: List[dict], filename: str) -> Optional[dict]:
    for r in rows:
        if str(r.get("file", "")).strip() == filename.strip():
            return r
    return None


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, decimals: int = 4) -> str:
    if v != v:  # NaN
        return "N/A"
    return f"{v:.{decimals}f}"


def _delta_tag(in_val: float, tr_val: float, higher_better: bool = True) -> str:
    """返回迁移退化/提升的标记和百分比。"""
    if in_val != in_val or tr_val != tr_val or in_val == 0:
        return ""
    delta = (tr_val - in_val) / abs(in_val)
    if higher_better:
        sym = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    else:
        sym = "↓" if delta > 0 else "↑" if delta < 0 else "→"
    return f" {sym} {_fmt(delta * 100, 2)}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge in-domain and transfer eval reports")
    parser.add_argument("--in_domain_json", type=str, required=True, help="域内评估报告 JSON")
    parser.add_argument("--transfer_json", type=str, required=True, help="迁移评估报告 JSON")
    parser.add_argument("--output", type=str, default="results/transfer_comparison.md")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="指定要对比的模型文件名列表；默认取两份报告中共同出现的文件",
    )
    args = parser.parse_args()

    in_data = _load_json(Path(args.in_domain_json))
    tr_data = _load_json(Path(args.transfer_json))

    in_rows = {str(r.get("file", "")): r for r in in_data.get("rows", [])}
    tr_rows = {str(r.get("file", "")): r for r in tr_data.get("rows", [])}

    if args.models:
        common_files = args.models
    else:
        common_files = sorted(set(in_rows.keys()) & set(tr_rows.keys()))

    if not common_files:
        print("[warn] 两份报告中没有共同的模型文件")
        return

    lines = [
        "# 域内 vs 迁移评估对比表",
        "",
        f"- 域内报告: `{args.in_domain_json}`",
        f"- 迁移报告: `{args.transfer_json}`",
        f"- 对比模型数: {len(common_files)}",
        "",
        "## 关键指标对比",
        "",
        "| 模型 | W-num ↓ | JS-cat ↓ | TSTR avg ↑ | R2 ↑ |",
        "|------|:-------:|:--------:|:----------:|:----:|",
    ]

    for fname in common_files:
        ir = in_rows[fname]
        tr = tr_rows[fname]

        w_in = _safe_float(ir.get("mean_wasserstein_numeric"))
        w_tr = _safe_float(tr.get("mean_wasserstein_numeric"))
        js_in = _safe_float(ir.get("mean_js_categorical"))
        js_tr = _safe_float(tr.get("mean_js_categorical"))
        tstr_in = _safe_float(ir.get("tstr_avg_score"))
        tstr_tr = _safe_float(tr.get("tstr_avg_score"))
        r2_in = _safe_float(ir.get("tstr_r2"))
        r2_tr = _safe_float(tr.get("tstr_r2"))

        w_delta = _delta_tag(w_in, w_tr, higher_better=False)
        js_delta = _delta_tag(js_in, js_tr, higher_better=False)
        tstr_delta = _delta_tag(tstr_in, tstr_tr, higher_better=True)
        r2_delta = _delta_tag(r2_in, r2_tr, higher_better=True)

        lines.append(
            f"| {fname} | {_fmt(w_in)} → {_fmt(w_tr)}{w_delta} | "
            f"{_fmt(js_in)} → {_fmt(js_tr)}{js_delta} | "
            f"{_fmt(tstr_in)} → {_fmt(tstr_tr)}{tstr_delta} | "
            f"{_fmt(r2_in)} → {_fmt(r2_tr)}{r2_delta} |"
        )

    lines.extend([
        "",
        "## 指标说明",
        "",
        "- **W-num**: 连续变量 Wasserstein 距离均值（越小越好）",
        "- **JS-cat**: 类别变量 Jensen-Shannon 散度均值（越小越好）",
        "- **TSTR avg**: 下游任务 XGBoost/RandomForest/MLP 平均得分（越大越好）",
        "- **R2**: XGBoost 回归 R²（跨年份不稳定，仅供参考）",
        "- 箭头方向表示迁移相对于域内的变化：↑ 提升 / ↓ 退化 / → 持平",
    ])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
