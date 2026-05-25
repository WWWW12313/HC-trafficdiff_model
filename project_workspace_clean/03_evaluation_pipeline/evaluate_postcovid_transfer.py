from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent


def _run_eval(real_test: str, profile: str, task_type: str) -> Path:
    cmd = [
        sys.executable,
        "pipeline/evaluate_all.py",
        "--real_test",
        real_test,
        "--file_glob",
        "*_compare_n*.csv",
        "--task_type",
        task_type,
        "--primary_metrics_profile",
        profile,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / "results" / "eval_report_latest.json"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index_rows(rows: List[dict]) -> Dict[str, dict]:
    return {str(r.get("file")): r for r in rows}


def _degradation(base: float, new: float, higher_better: bool) -> float:
    if base == 0:
        return 0.0
    if higher_better:
        return (new - base) / abs(base)
    return (base - new) / abs(base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate post-COVID transfer degradation/improvement")
    parser.add_argument("--task_type", default="regression", choices=["regression", "classification"])
    parser.add_argument("--primary_metrics_profile", default="no_rule")
    parser.add_argument("--in_domain_test", default="synthetic/nyc_crash/test.csv")
    parser.add_argument("--postcovid_2025", default="results/postcovid_2025_fully_enriched_like_2017.csv")
    parser.add_argument("--old_file", default="ablation_no_causal_compare_n10000.csv")
    parser.add_argument("--new_file", default="ours_full_model_compare_n10000.csv")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results"

    in_domain_json = _run_eval(args.in_domain_test, args.primary_metrics_profile, args.task_type)
    in_domain = _load_json(in_domain_json)
    (out_dir / f"eval_report_in_domain_{ts}.json").write_text(
        json.dumps(in_domain, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    y2025_json = _run_eval(args.postcovid_2025, args.primary_metrics_profile, args.task_type)
    y2025 = _load_json(y2025_json)
    (out_dir / f"eval_report_postcovid_2025_{ts}.json").write_text(
        json.dumps(y2025, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows_base = _index_rows(in_domain.get("rows", []))
    rows_2025 = _index_rows(y2025.get("rows", []))

    task_primary = "tstr_r2" if args.task_type == "regression" else "tstr_accuracy"
    stat_primary = "mean_js_categorical"

    def _line(model_file: str) -> dict:
        b = rows_base[model_file]
        d = rows_2025[model_file]
        return {
            "file": model_file,
            "task_metric": task_primary,
            "in_domain": b.get(task_primary),
            "postcovid_2025": d.get(task_primary),
            "degradation_2025": _degradation(
                float(b.get(task_primary) or 0), float(d.get(task_primary) or 0), True
            ),
            "stat_metric": stat_primary,
            "in_domain_js": b.get(stat_primary),
            "postcovid_2025_js": d.get(stat_primary),
        }

    compare_rows = [_line(args.old_file), _line(args.new_file)]

    report = {
        "generated_at": ts,
        "task_type": args.task_type,
        "primary_metrics_profile": args.primary_metrics_profile,
        "proxy_old_vs_new": {
            "old_file": args.old_file,
            "new_file": args.new_file,
            "note": "If dedicated old/new constrained model files are unavailable, this uses ablation_no_causal vs ours_full_model as a proxy.",
        },
        "transfer_summary": compare_rows,
    }

    latest_json = out_dir / "postcovid_transfer_report_latest.json"
    stamp_json = out_dir / f"postcovid_transfer_report_{ts}.json"
    latest_md = out_dir / "postcovid_transfer_report_latest.md"
    stamp_md = out_dir / f"postcovid_transfer_report_{ts}.md"

    txt = json.dumps(report, indent=2, ensure_ascii=False)
    latest_json.write_text(txt, encoding="utf-8")
    stamp_json.write_text(txt, encoding="utf-8")

    md = [
        "# Post-COVID Transfer Report",
        "",
        f"- generated_at: `{ts}`",
        f"- task_type: `{args.task_type}`",
        f"- primary_metrics_profile: `{args.primary_metrics_profile}`",
        "",
        "## Proxy Pair",
        "",
        f"- old_file: `{args.old_file}`",
        f"- new_file: `{args.new_file}`",
        "- note: dedicated old/new constrained synthetic files were not found; proxy pair is used.",
        "",
        "## Transfer Degradation Table",
        "",
        "| file | metric | in_domain | postcovid_2025 | deg_2025 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in compare_rows:
        md.append(
            f"| {r['file']} | {r['task_metric']} | {r['in_domain']:.6f} | {r['postcovid_2025']:.6f} | {r['degradation_2025']:.2%} |"
        )
    md_text = "\n".join(md)
    latest_md.write_text(md_text, encoding="utf-8")
    stamp_md.write_text(md_text, encoding="utf-8")

    print(txt)


if __name__ == "__main__":
    main()
