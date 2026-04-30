"""
v8_ablation_eval.py

Evaluate Free/Hard/Soft ablation outputs and produce markdown report.
Metrics:
1) Logic_Violation_Rate
2) Commonsense_Violation_Rate
3) Correction_Rate
4) Marginal_Drift (Categorical TV)
5) TSTR F1-Macro on PRIMARY_CAUSE-like target

Outputs:
- exp/nyc_crash_v8_ablation/v8_ablation_report.md
- exp/nyc_crash_v8_ablation/v8_ablation_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("v8.ablation.eval")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def vehicle_present(s: pd.Series) -> pd.Series:
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return ~txt.isin(["", "nan", "none", "null", "unknown", "unspecified"])


def logic_violation_rate(df: pd.DataFrame) -> float:
    n = max(len(df), 1)

    # multi-vehicle math
    v1 = "VEHICLE TYPE CODE 1" if "VEHICLE TYPE CODE 1" in df.columns else None
    v2 = "VEHICLE TYPE CODE 2" if "VEHICLE TYPE CODE 2" in df.columns else None
    bad_mv = np.zeros(len(df), dtype=bool)
    if v1 and v2 and "TOTAL_VEHICLES" in df.columns:
        both = vehicle_present(df[v1]).to_numpy() & vehicle_present(df[v2]).to_numpy()
        total = safe_num(df["TOTAL_VEHICLES"], default=1.0).to_numpy(dtype=np.float64)
        bad_mv = both & (total < 2)

    # casualty sums
    req = [
        "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED",
        "NUMBER OF CYCLIST INJURED",
        "NUMBER OF MOTORIST INJURED",
        "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST KILLED",
    ]
    bad_cs = np.zeros(len(df), dtype=bool)
    if all(c in df.columns for c in req):
        p_inj = safe_num(df["NUMBER OF PERSONS INJURED"], default=0).to_numpy(dtype=np.float64)
        inj_sum = (
            safe_num(df["NUMBER OF PEDESTRIANS INJURED"], default=0).to_numpy(dtype=np.float64)
            + safe_num(df["NUMBER OF CYCLIST INJURED"], default=0).to_numpy(dtype=np.float64)
            + safe_num(df["NUMBER OF MOTORIST INJURED"], default=0).to_numpy(dtype=np.float64)
        )
        p_kill = safe_num(df["NUMBER OF PERSONS KILLED"], default=0).to_numpy(dtype=np.float64)
        kill_sum = (
            safe_num(df["NUMBER OF PEDESTRIANS KILLED"], default=0).to_numpy(dtype=np.float64)
            + safe_num(df["NUMBER OF CYCLIST KILLED"], default=0).to_numpy(dtype=np.float64)
            + safe_num(df["NUMBER OF MOTORIST KILLED"], default=0).to_numpy(dtype=np.float64)
        )
        bad_cs = (p_inj != inj_sum) | (p_kill != kill_sum)

    bad = bad_mv | bad_cs
    return float(np.mean(bad)) if n > 0 else 0.0


def commonsense_violation_rate(df: pd.DataFrame) -> float:
    n = max(len(df), 1)
    veh_cols = [c for c in [
        "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"
    ] if c in df.columns]
    if not veh_cols:
        return 0.0

    snow_any = np.zeros(len(df), dtype=bool)
    for c in veh_cols:
        snow_any |= df[c].fillna("").astype(str).str.lower().str.contains("snow plow|snowplow|plow", regex=True).to_numpy()

    month = pd.to_datetime(df.get("CRASH DATE", pd.Series(["2017-01-01"] * len(df))), errors="coerce").dt.month.fillna(1).astype(int)
    temp = safe_num(df.get("CTX_TEMP", pd.Series(20.0, index=df.index)), default=20.0)
    prcp = safe_num(df.get("CTX_PRCP", pd.Series(0.0, index=df.index)), default=0.0)
    coco = safe_num(df.get("CTX_COCO", pd.Series(1, index=df.index)), default=1).astype(int)

    winter = month.isin([12, 1, 2]).to_numpy()
    snowy = coco.isin([15, 16]).to_numpy() | ((temp <= 2.0) & (prcp > 0.0)).to_numpy()
    allowed = winter | snowy

    bad = snow_any & (~allowed)
    return float(np.mean(bad)) if n > 0 else 0.0


def categorical_tv(real_df: pd.DataFrame, syn_df: pd.DataFrame, cols: List[str]) -> float:
    tvs: List[float] = []
    for c in cols:
        if c not in real_df.columns or c not in syn_df.columns:
            continue
        rp = real_df[c].fillna("<NA>").astype(str).value_counts(normalize=True)
        sp = syn_df[c].fillna("<NA>").astype(str).value_counts(normalize=True)
        keys = set(rp.index) | set(sp.index)
        tv = 0.5 * sum(abs(float(rp.get(k, 0.0)) - float(sp.get(k, 0.0))) for k in keys)
        tvs.append(float(tv))
    return float(np.mean(tvs)) if tvs else 0.0


def resolve_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "PRIMARY_CAUSE",
        "CONTRIBUTING FACTOR VEHICLE 1",
        "CAUSE_1",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def tstr_f1_macro(real_df: pd.DataFrame, syn_df: pd.DataFrame, seed: int = 42) -> Tuple[float, str]:
    target = resolve_target_column(real_df)
    if target is None or target not in syn_df.columns:
        return 0.0, "target_not_found"

    # feature columns
    drop_cols = {
        target,
        "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF CYCLIST INJURED", "NUMBER OF MOTORIST INJURED",
        "NUMBER OF PEDESTRIANS KILLED", "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST KILLED",
    }
    fcols = [c for c in real_df.columns if c in syn_df.columns and c not in drop_cols]
    if not fcols:
        return 0.0, "no_features"

    # bounded-cardinality filter for categorical
    filtered: List[str] = []
    for c in fcols:
        s = real_df[c]
        if pd.api.types.is_numeric_dtype(s):
            filtered.append(c)
        else:
            if s.fillna("<NA>").astype(str).nunique(dropna=False) <= 500:
                filtered.append(c)
    if not filtered:
        return 0.0, "no_filtered_features"

    real_sub = real_df.sample(n=min(len(real_df), 50000), random_state=seed).copy()
    syn_sub = syn_df.sample(n=min(len(syn_df), 50000), random_state=seed).copy()

    y_real = real_sub[target].fillna("<NA>").astype(str)
    y_syn = syn_sub[target].fillna("<NA>").astype(str)

    vc = y_real.value_counts()
    strat = y_real if (len(vc) > 1 and vc.min() >= 2) else None

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        real_sub[filtered], y_real, test_size=0.3, random_state=seed, stratify=strat
    )

    def encode_ordinal(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x_tr = np.zeros((len(train_df), len(filtered)), dtype=np.float32)
        x_te = np.zeros((len(test_df), len(filtered)), dtype=np.float32)
        for j, c in enumerate(filtered):
            tr = train_df[c]
            te = test_df[c]
            if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
                x_tr[:, j] = safe_num(tr, default=0.0).to_numpy(dtype=np.float32)
                x_te[:, j] = safe_num(te, default=0.0).to_numpy(dtype=np.float32)
            else:
                trs = tr.fillna("<NA>").astype(str)
                tes = te.fillna("<NA>").astype(str)
                cats = pd.Index(pd.concat([trs, tes], axis=0).unique())
                c2i = {k: i for i, k in enumerate(cats)}
                x_tr[:, j] = trs.map(c2i).astype(float).to_numpy(dtype=np.float32)
                x_te[:, j] = tes.map(c2i).astype(float).to_numpy(dtype=np.float32)
        return x_tr, x_te

    X_syn, X_test_tstr = encode_ordinal(syn_sub[filtered], X_test_real)

    clf = RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
    clf.fit(X_syn, y_syn)
    pred = clf.predict(X_test_tstr)

    return float(f1_score(y_test_real, pred, average="macro", zero_division=0)), target


def read_correction_rate(base_dir: Path, mode: str) -> float:
    if mode == "free":
        return 0.0
    meta = base_dir / f"synthetic_{mode}_meta.json"
    if not meta.exists():
        return 0.0
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        return float(payload.get("correction_rate", 0.0))
    except Exception:
        return 0.0


def evaluate_mode(real_df: pd.DataFrame, syn_df: pd.DataFrame, base_dir: Path, mode: str) -> Dict[str, Any]:
    key_cols = [
        "BOROUGH", "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3",
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2", "OSM_TYPE", "OSM_SPEED_TAG",
    ]
    tstr, tcol = tstr_f1_macro(real_df, syn_df)

    return {
        "mode": mode,
        "Logic_Violation_Rate": logic_violation_rate(syn_df),
        "Commonsense_Violation_Rate": commonsense_violation_rate(syn_df),
        "Correction_Rate": read_correction_rate(base_dir, mode),
        "Marginal_Drift_Categorical_TV": categorical_tv(real_df, syn_df, key_cols),
        "TSTR_F1_Macro": tstr,
        "TSTR_Target": tcol,
        "rows": int(len(syn_df)),
    }


def write_markdown_report(results: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# v8 Ablation Report")
    lines.append("")
    lines.append("Three-way ablation on LLM-driven priors:")
    lines.append("- Free: no constraints")
    lines.append("- Hard: logic + commonsense hard overwrite")
    lines.append("- Soft: commonsense rejection sampling + logic hard overwrite")
    lines.append("")

    lines.append("## Metrics Table")
    lines.append("| mode | Logic_Violation_Rate | Commonsense_Violation_Rate | Correction_Rate | Marginal_Drift_Categorical_TV | TSTR_F1_Macro | target | rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for r in results:
        lines.append(
            f"| {r['mode']} | {r['Logic_Violation_Rate']:.6f} | {r['Commonsense_Violation_Rate']:.6f} | {r['Correction_Rate']:.6f} | {r['Marginal_Drift_Categorical_TV']:.6f} | {r['TSTR_F1_Macro']:.6f} | {r['TSTR_Target']} | {r['rows']} |"
        )

    lines.append("")
    lines.append("## Interpretation Hints")
    lines.append("- Lower Logic/Commonsense Violation is better.")
    lines.append("- Lower Categorical TV drift is better (distribution preservation).")
    lines.append("- Higher TSTR F1-Macro is better (downstream utility).")
    lines.append("- Soft mode is expected to be a Pareto compromise between Hard and Free.")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v8 ablation evaluator")
    parser.add_argument("--real_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--base_dir", type=str, default="exp/nyc_crash_v8_ablation")
    parser.add_argument("--out_md", type=str, default="exp/nyc_crash_v8_ablation/v8_ablation_report.md")
    parser.add_argument("--out_json", type=str, default="exp/nyc_crash_v8_ablation/v8_ablation_metrics.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    base_dir = Path(args.base_dir)
    real_df = pd.read_csv(args.real_csv)

    modes = ["free", "hard", "soft"]
    results: List[Dict[str, Any]] = []

    for mode in modes:
        syn_path = base_dir / f"synthetic_{mode}.csv"
        if not syn_path.exists():
            LOGGER.warning("Missing dataset for mode=%s at %s", mode, syn_path.as_posix())
            continue
        LOGGER.info("Evaluating mode=%s", mode)
        syn_df = pd.read_csv(syn_path)
        results.append(evaluate_mode(real_df, syn_df, base_dir, mode))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(results, out_md)

    LOGGER.info("Saved metrics json: %s", out_json.as_posix())
    LOGGER.info("Saved markdown report: %s", out_md.as_posix())


if __name__ == "__main__":
    main()
