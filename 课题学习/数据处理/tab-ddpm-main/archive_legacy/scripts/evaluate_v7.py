"""
evaluate_v7.py

Evaluate v7 synthetic data with focus on:
1) Logic violation rate for casualty sums.
2) Sparse-target TSTR for death/injury atomic components.
3) Basic marginal similarity summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ATOMIC_TARGETS = [
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
]


def logic_violation_rate(df: pd.DataFrame) -> float:
    req = set(ATOMIC_TARGETS + ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"])
    if not req.issubset(set(df.columns)):
        return 1.0

    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    p_inj = df["NUMBER OF PERSONS INJURED"].to_numpy(dtype=np.float64)
    ped_inj = df["NUMBER OF PEDESTRIANS INJURED"].to_numpy(dtype=np.float64)
    cyc_inj = df["NUMBER OF CYCLIST INJURED"].to_numpy(dtype=np.float64)
    mot_inj = df["NUMBER OF MOTORIST INJURED"].to_numpy(dtype=np.float64)

    p_kil = df["NUMBER OF PERSONS KILLED"].to_numpy(dtype=np.float64)
    ped_kil = df["NUMBER OF PEDESTRIANS KILLED"].to_numpy(dtype=np.float64)
    cyc_kil = df["NUMBER OF CYCLIST KILLED"].to_numpy(dtype=np.float64)
    mot_kil = df["NUMBER OF MOTORIST KILLED"].to_numpy(dtype=np.float64)

    lhs_inj = p_inj
    rhs_inj = ped_inj + cyc_inj + mot_inj

    lhs_kil = p_kil
    rhs_kil = ped_kil + cyc_kil + mot_kil

    violated = (lhs_inj != rhs_inj) | (lhs_kil != rhs_kil)
    return float(np.mean(violated))


def _feature_columns(df: pd.DataFrame, target: str, max_cat_card: int = 500) -> List[str]:
    drop_cols = set(ATOMIC_TARGETS + ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", target])
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
            continue
        nunique = s.fillna("<NA>").astype(str).nunique(dropna=False)
        if nunique <= max_cat_card:
            cols.append(c)
    return cols


def _encode_ordinal(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_train = np.zeros((len(train_df), len(cols)), dtype=np.float32)
    x_test = np.zeros((len(test_df), len(cols)), dtype=np.float32)

    for j, c in enumerate(cols):
        tr = train_df[c]
        te = test_df[c]
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            x_train[:, j] = pd.to_numeric(tr, errors="coerce").fillna(0.0).astype(float).values
            x_test[:, j] = pd.to_numeric(te, errors="coerce").fillna(0.0).astype(float).values
        else:
            trs = tr.fillna("<NA>").astype(str)
            tes = te.fillna("<NA>").astype(str)
            cats = pd.Index(pd.concat([trs, tes], axis=0).unique())
            c2i = {k: i for i, k in enumerate(cats)}
            x_train[:, j] = trs.map(c2i).astype(float).values
            x_test[:, j] = tes.map(c2i).astype(float).values

    return x_train, x_test


def tstr_for_target(real_df: pd.DataFrame, syn_df: pd.DataFrame, target: str, seed: int = 42) -> Dict:
    real_sub = real_df.sample(n=min(len(real_df), 50000), random_state=seed).copy()
    syn_sub = syn_df.sample(n=min(len(syn_df), 50000), random_state=seed).copy()

    cols_real = _feature_columns(real_sub, target)
    cols_syn = _feature_columns(syn_sub, target)
    cols = [c for c in cols_real if c in cols_syn]
    if not cols:
        return {"target": target, "error": "No common usable feature columns after filtering."}

    y_real = pd.to_numeric(real_sub[target], errors="coerce").fillna(0).astype(int)
    y_syn = pd.to_numeric(syn_sub[target], errors="coerce").fillna(0).astype(int)

    vc = y_real.value_counts()
    use_stratify = (len(vc) > 1) and (vc.min() >= 2)
    X_train_real_df, X_test_real_df, y_train_real, y_test_real = train_test_split(
        real_sub[cols],
        y_real,
        test_size=0.3,
        random_state=seed,
        stratify=y_real if use_stratify else None,
    )

    X_syn_arr, X_test_tstr = _encode_ordinal(syn_sub[cols], X_test_real_df, cols)
    X_train_real_arr, X_test_real_arr = _encode_ordinal(X_train_real_df, X_test_real_df, cols)

    clf_tstr = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf_tstr.fit(X_syn_arr, y_syn)
    pred_tstr = clf_tstr.predict(X_test_tstr)

    clf_real = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf_real.fit(X_train_real_arr, y_train_real)
    pred_real = clf_real.predict(X_test_real_arr)

    macro_tstr = f1_score(y_test_real, pred_tstr, average="macro", zero_division=0)
    macro_real = f1_score(y_test_real, pred_real, average="macro", zero_division=0)

    return {
        "target": target,
        "tstr_accuracy": float(accuracy_score(y_test_real, pred_tstr)),
        "real_accuracy": float(accuracy_score(y_test_real, pred_real)),
        "tstr_macro_f1": float(macro_tstr),
        "real_macro_f1": float(macro_real),
        "fidelity_ratio": float(macro_tstr / (macro_real + 1e-9)),
        "n_classes": int(y_real.nunique()),
        "positive_rate_real": float((y_real > 0).mean()),
        "positive_rate_syn": float((y_syn > 0).mean()),
    }


def marginal_similarity(real_df: pd.DataFrame, syn_df: pd.DataFrame, max_cols: int = 40) -> Dict:
    shared = [c for c in real_df.columns if c in syn_df.columns]
    shared = shared[:max_cols]

    numeric_drifts = {}
    categorical_tv = {}

    for c in shared:
        r = real_df[c]
        s = syn_df[c]

        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(s):
            r = pd.to_numeric(r, errors="coerce").fillna(0)
            s = pd.to_numeric(s, errors="coerce").fillna(0)
            numeric_drifts[c] = float(abs(r.mean() - s.mean()) / (r.std() + 1e-6))
        else:
            rp = r.fillna("<NA>").astype(str).value_counts(normalize=True)
            sp = s.fillna("<NA>").astype(str).value_counts(normalize=True)
            keys = set(rp.index) | set(sp.index)
            tv = 0.5 * sum(abs(rp.get(k, 0.0) - sp.get(k, 0.0)) for k in keys)
            categorical_tv[c] = float(tv)

    return {
        "numeric_mean_std_drift": numeric_drifts,
        "categorical_total_variation": categorical_tv,
        "numeric_drift_avg": float(np.mean(list(numeric_drifts.values())) if numeric_drifts else 0.0),
        "categorical_tv_avg": float(np.mean(list(categorical_tv.values())) if categorical_tv else 0.0),
    }


def evaluate(real_csv: str, syn_csv: str, out_json: str) -> None:
    real_df = pd.read_csv(real_csv)
    syn_df = pd.read_csv(syn_csv)

    result = {
        "real_csv": real_csv,
        "syn_csv": syn_csv,
        "rows_real": int(len(real_df)),
        "rows_syn": int(len(syn_df)),
    }

    result["logic_violation_rate"] = logic_violation_rate(syn_df.copy())

    tstr_results = []
    for t in ATOMIC_TARGETS:
        if t in real_df.columns and t in syn_df.columns:
            try:
                tstr_results.append(tstr_for_target(real_df, syn_df, t))
            except Exception as e:
                tstr_results.append({"target": t, "error": str(e)})
    result["sparse_tstr"] = tstr_results

    result["marginal_similarity"] = marginal_similarity(real_df, syn_df)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== v7 evaluation done ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v7 synthetic dataset")
    parser.add_argument("--real_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--syn_csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="exp/nyc_crash_v7/eval_v7.json")
    args = parser.parse_args()

    evaluate(args.real_csv, args.syn_csv, args.out_json)


if __name__ == "__main__":
    main()
