"""
Run classic external tabular baselines (CTGAN / TVAE / SMOTE) for internal benchmarking.

Before running, make sure dependencies are installed in crashgen:
  python -m pip install sdv imbalanced-learn

Example:
  python pipeline/run_external_baselines.py --tier full --dataname nyc_crash_2024_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CDT_ROOT = Path(__file__).resolve().parent.parent

TIER_ROWS = {
    "quick": 500,
    "balanced": 2000,
    "full": 10000,
}


def _load_info(dataname: str) -> dict:
    info_path = CDT_ROOT / "data" / dataname / "info.json"
    if not info_path.is_file():
        return {}
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_real_train_path(dataname: str) -> Path:
    candidates = [
        CDT_ROOT / "synthetic" / dataname / "real.csv",
        CDT_ROOT / "synthetic" / dataname / "train.csv",
        CDT_ROOT / "data" / dataname / "train.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find real training CSV. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def _sample_by_tier(df: pd.DataFrame, tier: str, seed: int) -> pd.DataFrame:
    n = int(TIER_ROWS[tier])
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _detect_metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import Metadata

        return Metadata.detect_from_dataframe(data=df)
    except Exception:
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        return metadata


def _fit_sample_ctgan(df: pd.DataFrame, n_rows: int, seed: int, tier: str) -> pd.DataFrame:
    from sdv.single_table import CTGANSynthesizer

    metadata = _detect_metadata(df)
    epochs = 30 if tier == "quick" else 100 if tier == "balanced" else 300
    try:
        synth = CTGANSynthesizer(metadata=metadata, epochs=epochs, verbose=False)
    except TypeError:
        synth = CTGANSynthesizer(metadata=metadata, epochs=epochs)
    synth.fit(df)
    sampled = synth.sample(num_rows=n_rows)
    return sampled.reset_index(drop=True)


def _fit_sample_tvae(df: pd.DataFrame, n_rows: int, seed: int, tier: str) -> pd.DataFrame:
    from sdv.single_table import TVAESynthesizer

    metadata = _detect_metadata(df)
    epochs = 30 if tier == "quick" else 100 if tier == "balanced" else 300
    try:
        synth = TVAESynthesizer(metadata=metadata, epochs=epochs, verbose=False)
    except TypeError:
        synth = TVAESynthesizer(metadata=metadata, epochs=epochs)
    synth.fit(df)
    sampled = synth.sample(num_rows=n_rows)
    return sampled.reset_index(drop=True)


def _pick_smote_target(df: pd.DataFrame, info: dict) -> str:
    preferred = [
        "INJURY_COUNT",
        "injury_count",
        info.get("target_col"),
        "NUMBER OF PERSONS INJURED",
        "target",
        "y",
    ]
    for col in preferred:
        if col and col in df.columns and df[col].nunique(dropna=True) > 1:
            return col

    candidates: List[Tuple[str, int]] = []
    for c in df.columns:
        ser = df[c]
        nunique = int(ser.nunique(dropna=True))
        if nunique <= 1:
            continue
        is_cat_like = (
            pd.api.types.is_object_dtype(ser)
            or pd.api.types.is_categorical_dtype(ser)
            or (pd.api.types.is_integer_dtype(ser) and nunique <= 100)
        )
        if is_cat_like:
            candidates.append((c, nunique))

    if not candidates:
        return max(
            (c for c in df.columns if df[c].nunique(dropna=True) > 1),
            key=lambda c: int(df[c].nunique(dropna=True)),
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _encode_for_smote(
    df: pd.DataFrame,
    target_col: str,
    info: dict,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[int, object]], List[int]]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    cat_names = list(info.get("cat_col_names", [])) if info else []
    if not cat_names:
        cat_names = [
            c
            for c in X.columns
            if pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_categorical_dtype(X[c])
        ]
    cat_names = [c for c in cat_names if c in X.columns]

    decode_maps: Dict[str, Dict[int, object]] = {}
    for c in X.columns:
        if c in cat_names:
            cats = pd.Categorical(X[c].astype(str).fillna("_nan_"))
            X[c] = cats.codes.astype(np.int64)
            decode_maps[c] = {int(i): v for i, v in enumerate(cats.categories.tolist())}
        else:
            xn = pd.to_numeric(X[c], errors="coerce")
            fill_v = float(xn.median()) if xn.notna().any() else 0.0
            X[c] = xn.fillna(fill_v)

    y_cat = pd.Categorical(y.astype(str).fillna("_nan_"))
    y_enc = pd.Series(y_cat.codes.astype(np.int64), name=target_col)
    y_decode = {int(i): v for i, v in enumerate(y_cat.categories.tolist())}
    decode_maps[target_col] = y_decode

    cat_indices = [X.columns.get_loc(c) for c in cat_names]
    return X, y_enc, decode_maps, cat_indices


def _build_sampling_strategy(y: pd.Series, n_new_needed: int) -> Dict[int, int]:
    counts = y.value_counts().sort_values(ascending=False)
    if len(counts) < 2:
        raise ValueError("SMOTE needs at least 2 classes in target.")

    target_count = int(counts.max() + np.ceil(n_new_needed / len(counts)))
    strategy = {
        int(cls): int(target_count)
        for cls, cnt in counts.items()
        if int(cnt) < target_count and int(cnt) >= 2
    }
    return strategy


def _fit_sample_smote(
    df: pd.DataFrame,
    n_rows: int,
    seed: int,
    info: dict,
) -> pd.DataFrame:
    from imblearn.over_sampling import SMOTENC

    target_col = _pick_smote_target(df, info)
    X_enc, y_enc, decode_maps, cat_indices = _encode_for_smote(df, target_col, info)

    n_new_needed = n_rows
    strategy = _build_sampling_strategy(y_enc, n_new_needed)
    min_count = int(y_enc.value_counts().min())
    k_neighbors = max(1, min(5, min_count - 1)) if min_count > 1 else 1

    if strategy:
        try:
            try:
                smote = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy=strategy,
                    random_state=seed,
                    k_neighbors=k_neighbors,
                    n_jobs=-1,
                )
            except TypeError:
                smote = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy=strategy,
                    random_state=seed,
                    k_neighbors=k_neighbors,
                )
            X_res, y_res = smote.fit_resample(X_enc, y_enc)
        except ValueError:
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=seed, sampling_strategy="not majority")
            X_res, y_res = ros.fit_resample(X_enc, y_enc)
    else:
        from imblearn.over_sampling import RandomOverSampler

        ros = RandomOverSampler(random_state=seed, sampling_strategy="not majority")
        X_res, y_res = ros.fit_resample(X_enc, y_enc)

    n_generated = max(0, len(X_res) - len(X_enc))
    if n_generated <= 0:
        raise ValueError("SMOTE did not generate new rows.")

    X_new = X_res.iloc[-n_generated:].copy().reset_index(drop=True)
    y_new = pd.Series(y_res[-n_generated:]).reset_index(drop=True)

    if n_generated < n_rows:
        extra_idx = np.random.default_rng(seed).choice(
            np.arange(n_generated), size=n_rows - n_generated, replace=True
        )
        X_extra = X_new.iloc[extra_idx].reset_index(drop=True)
        y_extra = y_new.iloc[extra_idx].reset_index(drop=True)
        X_new = pd.concat([X_new, X_extra], ignore_index=True)
        y_new = pd.concat([y_new, y_extra], ignore_index=True)

    X_new = X_new.iloc[:n_rows].copy().reset_index(drop=True)
    y_new = y_new.iloc[:n_rows].copy().reset_index(drop=True)

    for c in X_new.columns:
        if c in decode_maps:
            inv = decode_maps[c]
            X_new[c] = X_new[c].round().astype(int).map(inv).fillna("_nan_")

    y_inv = decode_maps[target_col]
    y_decoded = y_new.round().astype(int).map(y_inv).fillna("_nan_")

    syn = X_new.copy()
    insert_at = df.columns.get_loc(target_col)
    syn.insert(insert_at, target_col, y_decoded)
    return syn.reset_index(drop=True)


def _coerce_to_reference_schema(syn: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out = syn.copy()

    for c in ref.columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[ref.columns]

    for c in ref.columns:
        ref_col = ref[c]
        ref_dtype = ref_col.dtype

        if pd.api.types.is_numeric_dtype(ref_dtype):
            ser = pd.to_numeric(out[c], errors="coerce")
            if pd.api.types.is_integer_dtype(ref_dtype):
                fill_v = int(pd.to_numeric(ref_col, errors="coerce").dropna().mode().iloc[0]) if pd.to_numeric(ref_col, errors="coerce").dropna().shape[0] else 0
                ser = ser.fillna(fill_v).round()
                out[c] = ser.astype(ref_dtype)
            else:
                fill_v = float(pd.to_numeric(ref_col, errors="coerce").dropna().median()) if pd.to_numeric(ref_col, errors="coerce").dropna().shape[0] else 0.0
                ser = ser.fillna(fill_v)
                out[c] = ser.astype(ref_dtype)
        elif pd.api.types.is_bool_dtype(ref_dtype):
            sval = out[c].astype(str).str.lower().str.strip()
            out[c] = sval.isin({"1", "true", "t", "yes", "y"}).astype(bool)
        else:
            if pd.api.types.is_categorical_dtype(ref_dtype):
                categories = pd.Categorical(ref_col.astype(str)).categories
                as_str = out[c].astype(str)
                fallback = categories[0] if len(categories) else "_nan_"
                as_str = as_str.where(as_str.isin(set(categories)), other=fallback)
                out[c] = pd.Categorical(as_str, categories=categories)
            else:
                out[c] = out[c].astype(str)

    return out


def _clip_target_distribution(
    syn: pd.DataFrame, ref: pd.DataFrame, target_col: str, lo: float = 0.01, hi: float = 0.99
) -> pd.DataFrame:
    """把合成数据目标列 clip 到参考训练集的 lo~hi 分位数范围，缓解极端值导致 R2 为负的问题。"""
    if target_col not in syn.columns or target_col not in ref.columns:
        return syn
    ref_vals = pd.to_numeric(ref[target_col], errors="coerce").dropna()
    if len(ref_vals) == 0:
        return syn
    lo_val = float(ref_vals.quantile(lo))
    hi_val = float(ref_vals.quantile(hi))
    syn = syn.copy()
    syn_num = pd.to_numeric(syn[target_col], errors="coerce")
    syn[target_col] = syn_num.clip(lower=lo_val, upper=hi_val)
    try:
        syn[target_col] = syn[target_col].round().astype(ref[target_col].dtype)
    except Exception:
        pass
    print(f"[clip_target] {target_col}: clip to [{lo_val:.4f}, {hi_val:.4f}] "
          f"(ref p{int(lo*100)}-p{int(hi*100)}), syn mean={float(syn[target_col].mean()):.4f}")
    return syn


def _write_output(df: pd.DataFrame, name: str, tier: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_{tier}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTGAN/TVAE/SMOTE external baselines")
    parser.add_argument(
        "--tier",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "full"],
        help="Data subsampling tier.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataname",
        type=str,
        default="nyc_crash",
        help="训练源域数据目录名，位于 data/ 下，例如 nyc_crash_2024_v2",
    )
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default=None,
        help="baseline 输出目录；默认 results/synthetic",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    info = _load_info(args.dataname)
    real_path = _resolve_real_train_path(args.dataname)
    out_dir = Path(args.synthetic_dir) if args.synthetic_dir else CDT_ROOT / "results" / "synthetic"
    if not out_dir.is_absolute():
        out_dir = CDT_ROOT / out_dir
    real_df_full = pd.read_csv(real_path, low_memory=False)
    real_df = _sample_by_tier(real_df_full, tier=args.tier, seed=args.seed)
    n = len(real_df)

    print(f"[data] real_train={real_path}")
    print(f"[data] tier={args.tier}, sampled_rows={n}")

    ctgan_syn = _fit_sample_ctgan(real_df, n_rows=n, seed=args.seed, tier=args.tier)
    tvae_syn = _fit_sample_tvae(real_df, n_rows=n, seed=args.seed, tier=args.tier)
    smote_syn = _fit_sample_smote(real_df, n_rows=n, seed=args.seed, info=info)

    ctgan_aligned = _coerce_to_reference_schema(ctgan_syn, real_df)
    tvae_aligned = _coerce_to_reference_schema(tvae_syn, real_df)
    smote_aligned = _coerce_to_reference_schema(smote_syn, real_df)

    target_col = info.get("target_col", "NUMBER OF PERSONS INJURED")
    ctgan_aligned = _clip_target_distribution(ctgan_aligned, real_df_full, target_col)

    p1 = _write_output(ctgan_aligned, "baseline_ctgan", args.tier, out_dir)
    p2 = _write_output(tvae_aligned, "baseline_tvae", args.tier, out_dir)
    p3 = _write_output(smote_aligned, "baseline_smote", args.tier, out_dir)

    print(f"[write] {p1}")
    print(f"[write] {p2}")
    print(f"[write] {p3}")


if __name__ == "__main__":
    main()
