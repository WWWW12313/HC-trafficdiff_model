"""
prepare_data_v7.py

v7 data preparation for Two-Stage Contextual Fusion + Causal-Logical generation.

Key ideas implemented:
1) Build absolute spatiotemporal anchors from real NYC data.
2) Build API-like physical context columns (OSM + weather) in training table.
3) Convert 6 sparse atomic casualty targets to categorical bins for multinomial diffusion.
4) Export TabDDPM-style npy files + v7 metadata.
5) Build a constrained DAG where context nodes are exogenous roots.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NYC_BOUNDS = {
    "lat_min": 40.4774,
    "lat_max": 40.9176,
    "lon_min": -74.2591,
    "lon_max": -73.7004,
}

ATOMIC_TARGETS = [
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
]


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0).astype(int)


def bin_count_4level(s: pd.Series) -> pd.Series:
    # 0, 1, 2, 3+
    x = safe_int_series(s)
    return x.clip(upper=3).astype(int)


def infer_time_period(hour: int) -> int:
    if 7 <= hour <= 9:
        return 1
    if 10 <= hour <= 15:
        return 2
    if 16 <= hour <= 19:
        return 3
    return 0


def normalize_weather_code(v) -> int:
    try:
        if pd.isna(v):
            return 1
        return int(float(v))
    except Exception:
        return 1


def normalize_speed_tag(v) -> str:
    if pd.isna(v):
        return "25 mph"
    t = str(v).strip()
    return t if t else "25 mph"


def build_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- absolute datetime ---
    dt = pd.to_datetime(
        out["CRASH DATE"].astype(str).str.strip() + " " + out["CRASH TIME"].astype(str).str.strip(),
        errors="coerce",
    )
    dt = dt.fillna(pd.Timestamp("2017-06-01 12:00:00"))

    out["CRASH_DATE_TS"] = (dt.astype("int64") // 10**9).astype(np.int64)
    out["CRASH_TIME_MIN"] = (dt.dt.hour * 60 + dt.dt.minute).astype(int)
    out["DAY_OF_WEEK"] = dt.dt.weekday.astype(int)
    out["CRASH_TIME_PERIOD"] = dt.dt.hour.map(infer_time_period).astype(int)

    # --- OSM-like context columns (from existing enriched columns if present) ---
    out["CTX_HIGHWAY"] = out.get("OSM_TYPE", "residential").fillna("residential").astype(str)
    out["CTX_ONEWAY"] = out.get("OSM_ONEWAY", 0)
    out["CTX_ONEWAY"] = pd.to_numeric(out["CTX_ONEWAY"], errors="coerce").fillna(0).astype(int)
    out["CTX_MAXSPEED"] = out.get("OSM_SPEED_TAG", out.get("REAL_SPEED_LIMIT", "25 mph")).map(normalize_speed_tag)

    if "cycleway" in out.columns:
        out["CTX_CYCLEWAY"] = out["cycleway"].fillna("no").astype(str)
    else:
        out["CTX_CYCLEWAY"] = "no"

    out["CTX_DIST_TO_INTERSECTION"] = pd.to_numeric(
        out.get("DIST_TO_SIGNAL_M", 500.0), errors="coerce"
    ).fillna(500.0).astype(float)

    out["CTX_IS_SIGNALIZED"] = pd.to_numeric(
        out.get("HAS_TRAFFIC_SIGNAL", 0), errors="coerce"
    ).fillna(0).astype(int)

    # --- Meteostat-like context columns ---
    out["CTX_TEMP"] = pd.to_numeric(out.get("TEMP_C", 15.0), errors="coerce").fillna(15.0).astype(float)
    out["CTX_PRCP"] = pd.to_numeric(out.get("prcp", 0.0), errors="coerce").fillna(0.0).astype(float)
    out["CTX_WSPD"] = pd.to_numeric(out.get("WIND_SPEED_KMH", 10.0), errors="coerce").fillna(10.0).astype(float)
    out["CTX_COCO"] = out.get("coco", 1).map(normalize_weather_code).astype(int)

    return out


def build_generated_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ATOMIC_TARGETS:
        if col not in out.columns:
            out[col] = 0
        out[col] = safe_int_series(out[col])

    for col in ATOMIC_TARGETS:
        bcol = f"{col}_BIN"
        out[bcol] = bin_count_4level(out[col])

    return out


def label_encode(train_df: pd.DataFrame, full_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, int]]:
    encoded = full_df.copy()
    mapping: Dict[str, Dict[str, int]] = {}
    cat_sizes: Dict[str, int] = {}

    for col in cols:
        values = full_df[col].fillna("<NA>").astype(str)
        cats = pd.Index(values.unique().tolist())
        c2i = {c: i for i, c in enumerate(cats)}
        encoded[col] = values.map(c2i).astype(int)
        mapping[col] = c2i
        cat_sizes[col] = len(c2i)

    return encoded, mapping, cat_sizes


def build_constrained_dag(context_roots: List[str], generated_nodes: List[str]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []

    # Root -> mid-level causes/vehicles
    cause_like = [
        "CONTRIBUTING FACTOR VEHICLE 1",
        "CONTRIBUTING FACTOR VEHICLE 2",
        "CONTRIBUTING FACTOR VEHICLE 3",
        "CONTRIBUTING FACTOR VEHICLE 4",
        "CONTRIBUTING FACTOR VEHICLE 5",
        "VEHICLE TYPE CODE 1",
        "VEHICLE TYPE CODE 2",
        "VEHICLE TYPE CODE 3",
        "VEHICLE TYPE CODE 4",
        "VEHICLE TYPE CODE 5",
        "TOTAL_VEHICLES",
        "IS_MULTI_VEHICLE",
    ]

    target_bins = [f"{c}_BIN" for c in ATOMIC_TARGETS]

    for r in context_roots:
        for c in cause_like:
            if c in generated_nodes:
                edges.append((r, c))
        for t in target_bins:
            if t in generated_nodes:
                edges.append((r, t))

    for c in cause_like:
        for t in target_bins:
            if c in generated_nodes and t in generated_nodes:
                edges.append((c, t))

    # enforce acyclic + remove illegal incoming edges to roots
    g = nx.DiGraph()
    g.add_nodes_from(context_roots + generated_nodes)
    g.add_edges_from(edges)

    for root in context_roots:
        for parent in list(g.predecessors(root)):
            g.remove_edge(parent, root)

    if not nx.is_directed_acyclic_graph(g):
        for cyc in list(nx.simple_cycles(g)):
            if len(cyc) > 1 and g.has_edge(cyc[-1], cyc[0]):
                g.remove_edge(cyc[-1], cyc[0])

    return list(g.edges())


def prepare_v7(input_csv: str, output_dir: str, val_ratio: float, test_ratio: float, seed: int) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Minimal cleaning for absolute anchors
    df["LATITUDE"] = pd.to_numeric(df.get("LATITUDE", np.nan), errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df.get("LONGITUDE", np.nan), errors="coerce")

    df = df.dropna(subset=["LATITUDE", "LONGITUDE", "CRASH DATE", "CRASH TIME"]).copy()
    df = df[
        (df["LATITUDE"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]))
        & (df["LONGITUDE"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]))
    ].copy()

    df = build_context_columns(df)
    df = build_generated_columns(df)

    context_numeric = [
        "LATITUDE", "LONGITUDE", "CRASH_DATE_TS", "CRASH_TIME_MIN",
        "DAY_OF_WEEK", "CRASH_TIME_PERIOD", "CTX_ONEWAY", "CTX_DIST_TO_INTERSECTION",
        "CTX_IS_SIGNALIZED", "CTX_TEMP", "CTX_PRCP", "CTX_WSPD", "CTX_COCO",
    ]
    context_categorical = ["CTX_HIGHWAY", "CTX_MAXSPEED", "CTX_CYCLEWAY"]

    gen_categorical = [
        "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3",
        "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5",
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
        "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4",
        "CONTRIBUTING FACTOR VEHICLE 5", "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
    ] + [f"{c}_BIN" for c in ATOMIC_TARGETS]

    # keep only available generated columns
    gen_categorical = [c for c in gen_categorical if c in df.columns]
    context_numeric = [c for c in context_numeric if c in df.columns]
    context_categorical = [c for c in context_categorical if c in df.columns]

    # split
    idx = np.arange(len(df))
    train_idx, tmp_idx = train_test_split(idx, test_size=val_ratio + test_ratio, random_state=seed, shuffle=True)
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=rel_test, random_state=seed, shuffle=True)

    df_train = df.iloc[train_idx].copy()

    # encode categorical context + generated categorical together
    all_cat_cols = context_categorical + gen_categorical
    encoded_all, col_mapping, cat_sizes = label_encode(df_train, df, all_cat_cols)

    # Build arrays
    X_ctx = np.concatenate([
        encoded_all[context_numeric].astype(float).values if context_numeric else np.zeros((len(df), 0), dtype=float),
        encoded_all[context_categorical].astype(float).values if context_categorical else np.zeros((len(df), 0), dtype=float),
    ], axis=1).astype(np.float32)

    X_num = np.zeros((len(df), 0), dtype=np.float32)
    X_cat = encoded_all[gen_categorical].astype(int).values.astype(np.int64) if gen_categorical else np.zeros((len(df), 0), dtype=np.int64)

    # dummy y for compatibility; v7 uses 6 atomic bin targets inside X_cat
    y_dummy = np.zeros((len(df),), dtype=np.float32)

    def save_split(name: str, split_idx: np.ndarray) -> None:
        np.save(out_dir / f"X_ctx_{name}.npy", X_ctx[split_idx])
        np.save(out_dir / f"X_num_{name}.npy", X_num[split_idx])
        np.save(out_dir / f"X_cat_{name}.npy", X_cat[split_idx])
        np.save(out_dir / f"y_{name}.npy", y_dummy[split_idx])

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

    # DAG with strict root constraints
    context_roots = context_numeric + context_categorical
    dag_edges = build_constrained_dag(context_roots, gen_categorical)

    info = {
        "version": "v7",
        "rows": int(len(df)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "context_numeric": context_numeric,
        "context_categorical": context_categorical,
        "context_columns": context_numeric + context_categorical,
        "num_columns": [],
        "cat_columns": gen_categorical,
        "cat_sizes": [cat_sizes[c] for c in gen_categorical],
        "target_col": "NUMBER OF PERSONS INJURED",  # compatibility placeholder
        "atomic_targets": ATOMIC_TARGETS,
        "atomic_target_bins": [f"{c}_BIN" for c in ATOMIC_TARGETS],
        "bin_decode_default": {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0},
        "dag_edges": dag_edges,
        "constraints": {
            "context_as_exogenous_root": True,
            "forbid_any_edge_into_context": True,
        },
    }

    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    with open(out_dir / "column_mapping.json", "w", encoding="utf-8") as f:
        json.dump(col_mapping, f, ensure_ascii=False, indent=2)

    with open(out_dir / "dag_v7.json", "w", encoding="utf-8") as f:
        json.dump({"edges": dag_edges}, f, ensure_ascii=False, indent=2)

    report = {
        "input_csv": input_csv,
        "output_dir": str(out_dir),
        "rows_after_clean": int(len(df)),
        "context_columns": info["context_columns"],
        "generated_cat_columns": gen_categorical,
        "dag_edge_count": len(dag_edges),
    }
    with open(out_dir / "prepare_report_v7.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== v7 data prepared ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v7 dataset for two-stage conditional causal diffusion")
    parser.add_argument("--input_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--output_dir", type=str, default="data/nyc_crash_v7")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_v7(args.input_csv, args.output_dir, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
