from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from benchmark_evaluator import BenchmarkEvaluator

ROOT = Path(__file__).resolve().parent.parent


def _align_xy(df: pd.DataFrame, target: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    x = df.reindex(columns=feature_cols).copy()
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(x[c].median())
        else:
            x[c] = x[c].astype(str).fillna("_nan_")
    x = pd.get_dummies(x, drop_first=False)
    return x, y


def _evaluate_xgb_r2(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> float:
    feats = [c for c in train_df.columns if c != target and c in test_df.columns]
    x_train, y_train = _align_xy(train_df, target, feats)
    x_test, y_test = _align_xy(test_df, target, feats)
    all_cols = sorted(set(x_train.columns) | set(x_test.columns))
    x_train = x_train.reindex(columns=all_cols, fill_value=0)
    x_test = x_test.reindex(columns=all_cols, fill_value=0)

    ev = BenchmarkEvaluator(task_type="regression")
    detail_df, _ = ev.evaluate(x_train, y_train, x_test, y_test)
    row = detail_df[detail_df["model"] == "xgboost"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0].get("r2", np.nan))


def _duplicate_ratio(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float(df.duplicated().mean())


def _const_col_ratio(df: pd.DataFrame) -> float:
    if df.shape[1] == 0:
        return 0.0
    const_cnt = sum(int(df[c].nunique(dropna=False) <= 1) for c in df.columns)
    return float(const_cnt / df.shape[1])


def _target_shift(real: pd.Series, syn: pd.Series) -> Dict[str, float]:
    r = pd.to_numeric(real, errors="coerce").dropna()
    s = pd.to_numeric(syn, errors="coerce").dropna()
    return {
        "real_mean": float(r.mean()),
        "syn_mean": float(s.mean()),
        "real_std": float(r.std(ddof=0)),
        "syn_std": float(s.std(ddof=0)),
        "real_q95": float(r.quantile(0.95)),
        "syn_q95": float(s.quantile(0.95)),
    }


def _score_causes(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    causes = []
    causes.append({
        "cause": "Sample quality / mode collapse",
        "score": round(100.0 * min(1.0, stats["duplicate_ratio"] * 10.0), 2),
        "evidence": f"duplicate_ratio={stats['duplicate_ratio']:.4f}",
    })
    shift = abs(stats["target_shift"]["syn_std"] - stats["target_shift"]["real_std"]) / max(
        abs(stats["target_shift"]["real_std"]), 1e-6
    )
    causes.append({
        "cause": "Label distribution mismatch",
        "score": round(100.0 * min(1.0, shift), 2),
        "evidence": f"std_gap_ratio={shift:.4f}",
    })
    causes.append({
        "cause": "Numerical stability / degenerate columns",
        "score": round(100.0 * min(1.0, stats["const_ratio"] + stats["nan_ratio"]), 2),
        "evidence": f"const_ratio={stats['const_ratio']:.4f}, nan_ratio={stats['nan_ratio']:.4f}",
    })
    causes.sort(key=lambda x: x["score"], reverse=True)
    return causes


def _diagnose_one(ctgan_csv: Path, real_test: pd.DataFrame, target: str) -> Dict[str, Any]:
    syn = pd.read_csv(ctgan_csv)

    missing = [c for c in real_test.columns if c not in syn.columns]
    extra = [c for c in syn.columns if c not in real_test.columns]
    common = [c for c in real_test.columns if c in syn.columns]

    num_syn = syn.select_dtypes(include=[np.number])
    nan_ratio = float(num_syn.isna().mean().mean()) if not num_syn.empty else 0.0
    inf_ratio = float(np.isinf(num_syn.to_numpy(dtype=np.float64)).mean()) if not num_syn.empty else 0.0

    stats = {
        "file": ctgan_csv.name,
        "n_rows": int(len(syn)),
        "feature_alignment": {
            "missing_cols_count": int(len(missing)),
            "extra_cols_count": int(len(extra)),
            "missing_cols": missing[:12],
            "extra_cols": extra[:12],
        },
        "target_shift": _target_shift(real_test[target], syn[target]),
        "duplicate_ratio": _duplicate_ratio(syn[common]),
        "const_ratio": _const_col_ratio(syn[common]),
        "nan_ratio": nan_ratio,
        "inf_ratio": inf_ratio,
    }

    before_r2 = _evaluate_xgb_r2(syn[common], real_test[common], target)

    # Low-risk fixes: deduplicate + clip target tail to real quantiles.
    fixed = syn[common].drop_duplicates().reset_index(drop=True)
    q_low = float(pd.to_numeric(real_test[target], errors="coerce").quantile(0.01))
    q_high = float(pd.to_numeric(real_test[target], errors="coerce").quantile(0.99))
    fixed[target] = pd.to_numeric(fixed[target], errors="coerce").fillna(0.0).clip(lower=q_low, upper=q_high)

    after_r2 = _evaluate_xgb_r2(fixed, real_test[common], target)

    stats["quick_fix"] = {
        "fixes": [
            "drop duplicated synthetic rows",
            "clip target to real 1%-99% quantiles",
        ],
        "r2_before": before_r2,
        "r2_after": after_r2,
        "delta_r2": float(after_r2 - before_r2),
    }

    stats["root_cause_top3"] = _score_causes(stats)[:3]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CTGAN negative R2 in regression")
    parser.add_argument("--real_test", default="synthetic/nyc_crash/test.csv")
    parser.add_argument("--target_col", default="NUMBER OF PERSONS INJURED")
    parser.add_argument("--ctgan_files", nargs="+", default=[
        "results/synthetic/baseline_ctgan_compare_n2000.csv",
        "results/synthetic/baseline_ctgan_compare_n10000.csv",
    ])
    args = parser.parse_args()

    real_path = ROOT / args.real_test
    if not real_path.is_file():
        raise SystemExit(f"Real test file not found: {real_path}")
    real_test = pd.read_csv(real_path)

    results = []
    for item in args.ctgan_files:
        p = ROOT / item
        if p.is_file():
            results.append(_diagnose_one(p, real_test, args.target_col))

    overall_top3 = []
    if results:
        pool = []
        for r in results:
            for c in r["root_cause_top3"]:
                pool.append(c)
        agg: Dict[str, Dict[str, Any]] = {}
        for c in pool:
            name = c["cause"]
            agg.setdefault(name, {"cause": name, "score": 0.0, "count": 0})
            agg[name]["score"] += float(c["score"])
            agg[name]["count"] += 1
        overall_top3 = sorted(
            [{"cause": v["cause"], "avg_score": v["score"] / max(v["count"], 1), "count": v["count"]} for v in agg.values()],
            key=lambda x: x["avg_score"],
            reverse=True,
        )[:3]

    recommendation = "Keep CTGAN as secondary baseline only; not suitable as primary baseline under current setup."
    report = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "real_test": args.real_test,
        "target_col": args.target_col,
        "diagnosis": results,
        "overall_root_cause_top3": overall_top3,
        "recommendation": recommendation,
    }

    ts = report["generated_at"]
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_json = out_dir / "ctgan_regression_diagnosis_latest.json"
    stamp_json = out_dir / f"ctgan_regression_diagnosis_{ts}.json"
    latest_md = out_dir / "ctgan_regression_diagnosis_latest.md"
    stamp_md = out_dir / f"ctgan_regression_diagnosis_{ts}.md"

    raw = json.dumps(report, indent=2, ensure_ascii=False)
    latest_json.write_text(raw, encoding="utf-8")
    stamp_json.write_text(raw, encoding="utf-8")

    md = [
        "# CTGAN Regression Diagnosis",
        "",
        f"- generated_at: `{ts}`",
        f"- target_col: `{args.target_col}`",
        "",
        "## Overall Root Cause Top3",
        "",
        "| rank | cause | avg_score | count |",
        "| --- | --- | --- | --- |",
    ]
    for i, c in enumerate(overall_top3, start=1):
        md.append(f"| {i} | {c['cause']} | {c['avg_score']:.2f} | {c['count']} |")
    md.append("")
    md.append("## Quick Fix Recheck")
    md.append("")
    md.append("| file | r2_before | r2_after | delta_r2 |")
    md.append("| --- | --- | --- | --- |")
    for d in results:
        q = d["quick_fix"]
        md.append(f"| {d['file']} | {q['r2_before']:.6f} | {q['r2_after']:.6f} | {q['delta_r2']:.6f} |")
    md.append("")
    md.append("Recommendation: keep CTGAN as a secondary baseline only.")

    md_text = "\n".join(md)
    latest_md.write_text(md_text, encoding="utf-8")
    stamp_md.write_text(md_text, encoding="utf-8")

    print(raw)


if __name__ == "__main__":
    main()
