from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_avg_fidelity(v7_payload: Dict[str, Any]) -> float:
    rows = v7_payload.get("sparse_tstr", [])
    vals: List[float] = []
    for row in rows:
        if isinstance(row, dict) and "fidelity_ratio" in row:
            try:
                vals.append(float(row["fidelity_ratio"]))
            except Exception:
                pass
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def build_report(eval_state: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Non-Baseline Unified Report (v7/v8/v9)")
    lines.append("")
    lines.append(f"Reference CSV: `{eval_state.get('reference_csv', '')}`")
    lines.append("")

    lines.append("## v7 Profile Comparison")
    lines.append("| profile | logic_violation_rate | avg_sparse_fidelity_ratio |")
    lines.append("|---|---:|---:|")

    lines.append("")
    lines.append("## v8 Profile Comparison")
    lines.append("| profile | mode | logic_violation | commonsense_violation | correction_rate | tstr_f1_macro |")
    lines.append("|---|---|---:|---:|---:|---:|")

    profiles = eval_state.get("profiles", {})
    v7_rows: List[str] = []
    v8_rows: List[str] = []

    for profile, p in profiles.items():
        v7_eval = json.loads(Path(p["v7_eval_json"]).read_text(encoding="utf-8"))
        logic = float(v7_eval.get("logic_violation_rate", 1.0))
        avg_fid = _safe_avg_fidelity(v7_eval)
        v7_rows.append(f"| {profile} | {logic:.6f} | {avg_fid:.6f} |")

        v8_eval = json.loads(Path(p["v8_eval_json"]).read_text(encoding="utf-8"))
        for row in v8_eval:
            v8_rows.append(
                "| {profile} | {mode} | {logic:.6f} | {cs:.6f} | {corr:.6f} | {tstr:.6f} |".format(
                    profile=profile,
                    mode=row.get("mode", "?"),
                    logic=float(row.get("Logic_Violation_Rate", 1.0)),
                    cs=float(row.get("Commonsense_Violation_Rate", 1.0)),
                    corr=float(row.get("Correction_Rate", 0.0)),
                    tstr=float(row.get("TSTR_F1_Macro", 0.0)),
                )
            )

    lines.extend(v7_rows)
    lines.append("")
    lines.extend(v8_rows)
    lines.append("")
    lines.append("## Notes")
    lines.append("- quick/balanced/full for v7 uses native train mode.")
    lines.append("- quick/balanced/full for v8 is mapped to different sample size and soft-mode LLM budget.")
    lines.append("- v9 LLM DAG output path is tracked in train_state.json when enabled.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final markdown report for non-baseline suite")
    parser.add_argument("--eval_state", type=str, default="exp/nonbaseline_suite_v9/eval_state.json")
    parser.add_argument("--out_md", type=str, default="exp/nonbaseline_suite_v9/nonbaseline_unified_report.md")
    args = parser.parse_args()

    eval_state = json.loads(Path(args.eval_state).read_text(encoding="utf-8"))
    report = build_report(eval_state)

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved report: {out.as_posix()}")


if __name__ == "__main__":
    main()
