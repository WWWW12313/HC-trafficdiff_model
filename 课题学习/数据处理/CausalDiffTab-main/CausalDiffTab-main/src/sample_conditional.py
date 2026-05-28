"""
Hierarchical CausalDiffTab - 完整采样器
========================================
支持两种模式:
  1. unconditional: 直接从 Stage 3 模型生成全量特征 (用于评估)
    2. conditional:   DDPM Inpainting, 锁定上游条件, 生成 Stage 2 或 Stage 3

Usage:
  # 无条件采样 (评估用)
  python src/sample_conditional.py \
    --ckpt_dir ckpt/nyc_crash/stage3_full_full \
    --num_samples 5000 --device cuda:0

  # 条件采样：训练集行索引（张量空间与 TabDiffDataset 一致），重采样 y + Stage3 分类
  python src/sample_conditional.py \
    --ckpt_dir ckpt/nyc_crash/stage3_full_full \
    --condition_train_indices "0,1,2" \
    --num_samples 100 --device cuda:0

  # 外部 CSV 条件：需与 TabDiff 前向编码对齐后方可接入；当前仍会回退无条件并提示
"""

import os
import sys
import json
import pickle
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CDT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CDT_ROOT))

import src as cdt_src
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from utils_train import TabDiffDataset, make_dataset


RETIRED_CONTEXT_COLUMNS = set()
DETERMINISTIC_OSM_NUMERIC = {"DIST_TO_SIGNAL_M", "REAL_SPEED_LIMIT", "INFERRED_LANES"}
DETERMINISTIC_OSM_CATEGORICAL = {"HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "OSM_TYPE", "HAS_DIVIDER"}
H3_ROADCELL_COLUMN = "ROAD_H3_CELL"


# ============================================================
# 1. 模型加载
# ============================================================

def load_model(
    ckpt_dir: str,
    data_dir: str,
    device: str = "cuda:0",
    ckpt_name: str = None,
) -> tuple:
    """
    从 checkpoint 目录加载训练好的扩散模型。

    Returns:
        diffusion: UnifiedCtimeDiffusion
        dataset: TabDiffDataset (用于 inverse transform)
        info: dict
    """
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        raw_config = pickle.load(f)

    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    dataset = TabDiffDataset(
        os.path.basename(data_dir), data_dir, info,
        y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    d_numerical = dataset.d_numerical
    categories = dataset.categories

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (
        (categories + 1).tolist() if len(categories) > 0 else []
    )

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
        causal_weight_max=1.0,
        causal_warmup_steps=1,
    )
    diffusion.to(device)

    if ckpt_name is None:
        pt_files = [f for f in os.listdir(ckpt_dir)
                     if f.startswith("best_model_") and f.endswith(".pt")]
        if pt_files:
            ckpt_name = sorted(pt_files)[0]
        else:
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            ckpt_name = sorted(pt_files)[-1] if pt_files else None

    if ckpt_name is None:
        raise FileNotFoundError(f"No .pt checkpoint found in {ckpt_dir}")

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(f"[load] Loading checkpoint: {ckpt_name}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])

    diffusion.eval()
    print(f"[load] Model loaded: d_numerical={d_numerical}, "
          f"categories={list(categories)[:5]}... ({len(categories)} total)")

    return diffusion, dataset, info


def load_forward_dataset(
    ckpt_dir: str,
    data_dir: str,
    info: dict,
):
    """加载保留正向 transforms 的数据对象，用于把上游生成结果重新编码到下游张量空间。"""
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        raw_config = pickle.load(f)

    return make_dataset(
        data_path=data_dir,
        T=cdt_src.Transformations(
            normalization="quantile",
            num_nan_policy="mean",
            cat_nan_policy=None,
            cat_min_frequency=None,
            cat_encoding=None,
            y_policy="default",
            dequant_dist=raw_config["data"]["dequant_dist"],
            int_dequant_factor=raw_config["data"]["int_dequant_factor"],
        ),
        task_type=info["task_type"],
        change_val=False,
        concat=True,
        y_only=False,
    )


# ============================================================
# 2. 无条件采样
# ============================================================

def sample_unconditional(
    diffusion: UnifiedCtimeDiffusion,
    dataset: TabDiffDataset,
    info: dict,
    num_samples: int = 5000,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    无条件采样: 使用模型直接生成全量特征, 经 inverse transform 得到 DataFrame。
    """
    print(f"[sample] Generating {num_samples} samples (batch={batch_size})...")

    all_samples = []
    remaining = num_samples

    while remaining > 0:
        b = min(remaining, batch_size)
        syn_tensor = diffusion.sample(b)
        all_samples.append(syn_tensor)
        remaining -= b
        print(f"  generated {num_samples - remaining}/{num_samples}")

    syn_data = torch.cat(all_samples, dim=0)[:num_samples]

    syn_df = tensor_to_dataframe(syn_data, dataset, info)
    return syn_df


def tensor_to_dataframe(
    syn_data: torch.Tensor,
    dataset: TabDiffDataset,
    info: dict,
) -> pd.DataFrame:
    """
    将模型输出张量转换为 DataFrame, 应用 inverse transforms。
    复刻 Trainer.sample_synthetic 的逻辑。
    """
    from tabdiff.trainer import split_num_cat_target, recover_data

    num_inverse = dataset.num_inverse
    int_inverse = dataset.int_inverse
    cat_inverse = dataset.cat_inverse

    arr = syn_data.detach().float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        arr, info, num_inverse, int_inverse, cat_inverse,
    )

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(k): v for k, v in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    return syn_df


# ============================================================
# 3. 条件 Inpainting 采样
# ============================================================

def build_stage_indices(info: dict, column_groups: dict) -> Dict[str, List[int]]:
    """
    构建各 Stage 特征在模型内部张量中的列索引。
    regression 模式下 y prepend 到 num 部分, 故 num 有 +1 offset。
    """
    num_col_names = info["num_col_names"]
    cat_col_names = info["cat_col_names"]
    cat_sizes = info["cat_sizes"]

    s1_cont = set(column_groups.get("stage1_continuous", []))
    s1_cat = set(column_groups.get("stage1_categorical", []))
    s2_cont = set(column_groups.get("stage2_continuous", []))
    s2_cat = set(column_groups.get("stage2_categorical", []))

    is_regression = info.get("task_type", "regression") == "regression"
    y_offset = 1 if is_regression else 0

    def num_indices(target_set):
        return [i + y_offset for i, c in enumerate(num_col_names) if c in target_set]

    cat_offsets = []
    cur = 0
    for s in cat_sizes:
        cat_offsets.append(cur)
        cur += s + 1

    def cat_indices(target_set):
        idx = []
        for i, c in enumerate(cat_col_names):
            if c in target_set:
                start = cat_offsets[i]
                end = start + cat_sizes[i] + 1
                idx.extend(range(start, end))
        return idx

    cond_cont = s1_cont | s2_cont
    cond_cat = s1_cat | s2_cat

    return {
        "cond_num_idx": num_indices(cond_cont),
        "free_num_idx": num_indices(set(num_col_names) - cond_cont),
        "cond_cat_idx": cat_indices(cond_cat),
        "free_cat_idx": cat_indices(set(cat_col_names) - cond_cat),
    }


def build_stage_impute_masks(
    info: dict,
    column_groups: dict,
    impute_stage: str = "stage3",
    free_target: bool = False,
):
    """
    用于 diffusion.sample_impute：num_mask_idx / cat_mask_idx 为需要 **重新生成** 的维度。
    与 make_dataset(regression) 一致：X_num 第 0 列为目标 y（经 TabDiff 侧 Quantile），
    其余列为 num_col_names 顺序的连续特征。
    """
    num_col_names = info["num_col_names"]
    cat_col_names = info["cat_col_names"]

    if impute_stage == "stage2":
        target_num = set(column_groups.get("stage2_continuous", []))
        target_cat = set(column_groups.get("stage2_categorical", []))
        y_offset = 1 if info.get("task_type", "regression") == "regression" else 0
        num_mask_idx = [i + y_offset for i, c in enumerate(num_col_names) if c in target_num]
        if free_target and y_offset:
            num_mask_idx = [0] + num_mask_idx
        cat_mask_idx = [i for i, c in enumerate(cat_col_names) if c in target_cat]
    elif impute_stage == "stage3":
        target_cat = set(column_groups.get("stage3_categorical", []))
        # 排除 proxy 列，改为后置计算以避免 proxy 泄漏
        PROXY_COLS = {
            "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
            "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
            "NUMBER_OF_CYCLIST_INJURED_BIN", "NUMBER_OF_CYCLIST_KILLED_BIN",
            "NUMBER_OF_MOTORIST_INJURED_BIN", "NUMBER_OF_MOTORIST_KILLED_BIN",
        }
        target_cat = target_cat - PROXY_COLS
        cat_mask_idx = [i for i, c in enumerate(cat_col_names) if c in target_cat]
        num_mask_idx = [0]
    else:
        raise ValueError(f"Unsupported impute_stage={impute_stage!r}")

    return num_mask_idx, cat_mask_idx


def _compute_proxy_columns(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """
    采样后后置计算 proxy 列，替代扩散模型直接生成。
    消除 proxy 泄漏：TOTAL_VEHICLES / IS_MULTI_VEHICLE 由 vehicle flags 确定；
    injury bins 由目标 y 和事故特征启发式推断。
    """
    df = df.copy()

    # ── Vehicle-derived proxies ──
    vehicle_cols = [
        "is_emergency", "is_taxi", "is_bus", "is_truck",
        "is_other_vehicle", "is_pickup", "is_van",
        "is_motorcycle", "is_bicycle", "is_suv", "is_sedan",
    ]
    present_vehicle = [c for c in vehicle_cols if c in df.columns]
    if present_vehicle:
        for c in present_vehicle:
            df[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0.0) > 0.5).astype(int)
        active_count = df[present_vehicle].sum(axis=1).astype(int)
        # 至少一辆车
        zero_vehicle = active_count <= 0
        if zero_vehicle.any():
            fallback = "is_sedan" if "is_sedan" in present_vehicle else present_vehicle[-1]
            df.loc[zero_vehicle, fallback] = 1
            active_count = df[present_vehicle].sum(axis=1).astype(int)
        df["TOTAL_VEHICLES"] = active_count.clip(1, 5).astype(int)
        df["IS_MULTI_VEHICLE"] = (df["TOTAL_VEHICLES"] >= 2).astype(int)

    # ── Injury BIN proxies (heuristic, based on target y and accident features) ──
    target_col = info.get("target_col", "NUMBER OF PERSONS INJURED")
    y = pd.to_numeric(df.get(target_col, pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    # 基础规则：y <= 0 时所有 injury bins = 0
    any_injury = (y > 0).astype(int)

    # Pedestrian
    if "NUMBER_OF_PEDESTRIANS_INJURED_BIN" in df.columns:
        ped_rel = pd.to_numeric(df.get("is_pedestrian_related", 0), errors="coerce").fillna(0)
        df["NUMBER_OF_PEDESTRIANS_INJURED_BIN"] = (any_injury & (ped_rel > 0.5)).astype(int)
    if "NUMBER_OF_PEDESTRIANS_KILLED_BIN" in df.columns:
        # 死亡概率更低：仅在 y > 2 且与行人相关时
        df["NUMBER_OF_PEDESTRIANS_KILLED_BIN"] = ((y > 2) & (pd.to_numeric(df.get("is_pedestrian_related", 0), errors="coerce").fillna(0) > 0.5)).astype(int)

    # Cyclist
    if "NUMBER_OF_CYCLIST_INJURED_BIN" in df.columns:
        bike = pd.to_numeric(df.get("is_bicycle", 0), errors="coerce").fillna(0)
        df["NUMBER_OF_CYCLIST_INJURED_BIN"] = (any_injury & (bike > 0.5)).astype(int)
    if "NUMBER_OF_CYCLIST_KILLED_BIN" in df.columns:
        df["NUMBER_OF_CYCLIST_KILLED_BIN"] = ((y > 2) & (pd.to_numeric(df.get("is_bicycle", 0), errors="coerce").fillna(0) > 0.5)).astype(int)

    # Motorist (default: likely injured if any accident)
    if "NUMBER_OF_MOTORIST_INJURED_BIN" in df.columns:
        df["NUMBER_OF_MOTORIST_INJURED_BIN"] = any_injury.astype(int)
    if "NUMBER_OF_MOTORIST_KILLED_BIN" in df.columns:
        df["NUMBER_OF_MOTORIST_KILLED_BIN"] = (y > 2).astype(int)

    return df


def _freeze_impute_columns(
    num_mask_idx: List[int],
    cat_mask_idx: List[int],
    info: dict,
    frozen_num: set[str],
    frozen_cat: set[str],
) -> tuple[List[int], List[int]]:
    y_offset = 1 if info.get("task_type", "regression") == "regression" else 0
    frozen_num_idx = {
        i + y_offset for i, col in enumerate(info["num_col_names"]) if col in frozen_num
    }
    frozen_cat_idx = {
        i for i, col in enumerate(info["cat_col_names"]) if col in frozen_cat
    }
    return (
        [idx for idx in num_mask_idx if idx not in frozen_num_idx],
        [idx for idx in cat_mask_idx if idx not in frozen_cat_idx],
    )


def _infer_data_year(*paths: str) -> Optional[str]:
    for path in paths:
        if not path:
            continue
        match = re.search(r"20\d{2}", str(path))
        if match:
            return match.group(0)
    return None


def _resolve_osm_resource(user_path: Optional[str], year: Optional[str], filename: str) -> Optional[str]:
    if user_path and os.path.exists(user_path):
        return user_path
    candidates = []
    if year:
        candidates.append(CDT_ROOT / "raw_data" / "osm" / year / filename)
    candidates.append(CDT_ROOT / "raw_data" / "osm" / filename)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else None


def _compat_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("yes", "true", "1", "on")


def _edge_value(edge_data: dict, key: str, default=None):
    value = edge_data.get(key, default)
    return value[0] if isinstance(value, list) else value


def _infer_lanes(raw_lanes, highway_type: str) -> int:
    try:
        return int(float(str(raw_lanes)))
    except (ValueError, TypeError):
        pass
    highway = str(highway_type).lower()
    if "motorway" in highway or "trunk" in highway:
        return 3
    if "primary" in highway:
        return 2
    return 1


def _infer_speed_limit_mph(raw_speed, highway_type: str) -> float:
    text = str(raw_speed or "").lower()
    values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if values:
        speed = float(np.median(values))
        if "km" in text or "kph" in text:
            speed *= 0.621371
        return float(np.clip(round(speed / 5.0) * 5.0, 5.0, 70.0))

    highway = str(highway_type).lower()
    if "motorway" in highway:
        return 50.0
    if "trunk" in highway:
        return 40.0
    if "primary" in highway or "secondary" in highway:
        return 30.0
    return 25.0


def enrich_with_osm_context(
    df: pd.DataFrame,
    info: dict,
    stage_data_dir: str,
    road_graphml: Optional[str] = None,
    road_signals: Optional[str] = None,
) -> pd.DataFrame:
    """Fill deterministic OSM context from generated LATITUDE/LONGITUDE."""
    required_num = DETERMINISTIC_OSM_NUMERIC & set(info.get("num_col_names", []))
    required_cat = DETERMINISTIC_OSM_CATEGORICAL & set(info.get("cat_col_names", []))
    if not required_num and not required_cat:
        return df
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        print("[osm_context] LATITUDE/LONGITUDE missing; skip deterministic OSM lookup")
        return df

    year = _infer_data_year(stage_data_dir)
    graph_path = _resolve_osm_resource(road_graphml, year, "nyc_drive_graph.graphml")
    if not graph_path or not os.path.exists(graph_path):
        print(f"[osm_context] graphml missing; skip deterministic OSM lookup: {graph_path}")
        return df
    signal_path = _resolve_osm_resource(road_signals, year, "nyc_traffic_signals.geojson")
    try:
        from src.road_snap import enrich_road_context

        return enrich_road_context(
            df,
            graphml_path=graph_path,
            signals_path=signal_path,
            columns=required_num | required_cat,
            overwrite=True,
            verbose=True,
        )
    except ImportError as exc:
        print(f"[osm_context] dependency missing; skip deterministic OSM lookup: {exc}")
    except Exception as exc:
        print(f"[osm_context] deterministic OSM lookup failed; keep generated defaults: {exc}")
    return df


def _support_key_value(value) -> str:
    if pd.isna(value):
        return "__NA__"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "__NA__"
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip().lower()


def _support_label_value(value, col_name: str, label_mappings: dict) -> str:
    raw = _support_key_value(value)
    mapping = label_mappings.get(col_name, {})
    mapped = mapping.get(raw, mapping.get(str(value), value))
    return _support_key_value(mapped)


def repair_stage1_spatial_support(
    stage1_df: pd.DataFrame,
    reference_csv: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Project generated Stage1 coordinates onto empirical crash-road support.

    Continuous latitude/longitude samples can match marginal ranges while landing
    away from the intersection lattice where real crashes occur. This repair keeps
    generated time/season anchors and samples a real crash coordinate from the
    matching temporal bucket before deterministic OSM lookup.
    """
    required = {"LATITUDE", "LONGITUDE"}
    key_cols = ["SEASON", "DAY_OF_WEEK", "TIME_PERIOD"]
    if not required.issubset(stage1_df.columns) or not os.path.exists(reference_csv):
        return stage1_df

    usecols = ["LATITUDE", "LONGITUDE"] + [col for col in key_cols if col in stage1_df.columns]
    try:
        ref = pd.read_csv(reference_csv, usecols=usecols, low_memory=False)
    except Exception as exc:
        print(f"[stage1_support] failed to load reference coordinates; keep generated coords: {exc}")
        return stage1_df

    ref = ref.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    if len(ref) == 0:
        return stage1_df

    out = stage1_df.copy()
    rng = np.random.default_rng(seed)
    active_keys = [col for col in key_cols if col in out.columns and col in ref.columns]
    label_mappings: dict = {}
    info_path = os.path.join(os.path.dirname(reference_csv), "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                label_mappings = json.load(f).get("cat_label_mappings", {})
        except Exception as exc:
            print(f"[stage1_support] failed to load category label mappings: {exc}")

    def build_groups(cols: list[str]) -> dict[tuple[str, ...], np.ndarray]:
        if not cols:
            return {(): ref.index.to_numpy(dtype=int)}
        keys = ref[cols].apply(lambda row: tuple(_support_label_value(row[col], col, label_mappings) for col in cols), axis=1)
        return {key: idx.to_numpy(dtype=int) for key, idx in keys.groupby(keys).groups.items()}

    groups_exact = build_groups(active_keys)
    groups_coarse = build_groups([col for col in ["SEASON", "TIME_PERIOD"] if col in active_keys])
    all_indices = ref.index.to_numpy(dtype=int)

    chosen: list[int] = []
    exact_hits = 0
    coarse_hits = 0
    for _, row in out.iterrows():
        exact_key = tuple(_support_label_value(row[col], col, label_mappings) for col in active_keys)
        candidates = groups_exact.get(exact_key)
        if candidates is not None and len(candidates) > 0:
            exact_hits += 1
        else:
            coarse_cols = [col for col in ["SEASON", "TIME_PERIOD"] if col in active_keys]
            coarse_key = tuple(_support_label_value(row[col], col, label_mappings) for col in coarse_cols)
            candidates = groups_coarse.get(coarse_key)
            if candidates is not None and len(candidates) > 0:
                coarse_hits += 1
            else:
                candidates = all_indices
        chosen.append(int(rng.choice(candidates)))

    sampled = ref.iloc[chosen].reset_index(drop=True)
    before_lat = pd.to_numeric(out["LATITUDE"], errors="coerce").to_numpy(dtype=float)
    before_lon = pd.to_numeric(out["LONGITUDE"], errors="coerce").to_numpy(dtype=float)
    out["LATITUDE"] = sampled["LATITUDE"].to_numpy(dtype=float)
    out["LONGITUDE"] = sampled["LONGITUDE"].to_numpy(dtype=float)
    moved_m = np.sqrt((before_lat - out["LATITUDE"].to_numpy(dtype=float)) ** 2 + (before_lon - out["LONGITUDE"].to_numpy(dtype=float)) ** 2) * 111_000.0
    print(
        "[stage1_support] repaired LATITUDE/LONGITUDE with empirical crash-road anchors: "
        f"exact={exact_hits}/{len(out)}, coarse={coarse_hits}/{len(out)}, "
        f"mean_move={float(np.nanmean(moved_m)):.1f}m"
    )
    return out


def repair_stage1_h3_roadcell_support(
    stage1_df: pd.DataFrame,
    reference_csv: str,
    resolution: int = 8,
    max_ring: int = 2,
    min_bucket_size: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Decode generated ROAD_H3_CELL to empirical crash-road anchors."""
    if H3_ROADCELL_COLUMN not in stage1_df.columns or not os.path.exists(reference_csv):
        return repair_stage1_spatial_support(stage1_df, reference_csv=reference_csv, seed=seed)
    try:
        import h3
    except ImportError as exc:
        print(f"[h3_roadcell] h3 missing; fallback to empirical spatial repair: {exc}")
        return repair_stage1_spatial_support(stage1_df, reference_csv=reference_csv, seed=seed)

    key_cols = ["SEASON", "TIME_PERIOD"]
    usecols = ["LATITUDE", "LONGITUDE", H3_ROADCELL_COLUMN] + [c for c in key_cols if c in stage1_df.columns]
    try:
        ref = pd.read_csv(reference_csv, usecols=lambda c: c in set(usecols), low_memory=False)
    except Exception as exc:
        print(f"[h3_roadcell] failed to load reference anchors; fallback: {exc}")
        return repair_stage1_spatial_support(stage1_df, reference_csv=reference_csv, seed=seed)

    ref = ref.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    if len(ref) == 0:
        return stage1_df
    if H3_ROADCELL_COLUMN not in ref.columns:
        ref[H3_ROADCELL_COLUMN] = [
            h3.latlng_to_cell(float(lat), float(lon), resolution)
            for lat, lon in zip(ref["LATITUDE"], ref["LONGITUDE"])
        ]

    out = stage1_df.copy()
    rng = np.random.default_rng(seed)
    active_keys = [col for col in key_cols if col in out.columns and col in ref.columns]
    label_mappings: dict = {}
    info_path = os.path.join(os.path.dirname(reference_csv), "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                label_mappings = json.load(f).get("cat_label_mappings", {})
        except Exception as exc:
            print(f"[h3_roadcell] failed to load category label mappings: {exc}")

    def key_from_row(row, cols):
        return tuple(_support_label_value(row[col], col, label_mappings) for col in cols)

    if active_keys:
        condition_keys = ref[active_keys].apply(lambda row: key_from_row(row, active_keys), axis=1)
    else:
        condition_keys = pd.Series([tuple()] * len(ref), index=ref.index)

    ref_cell_keys = ref[H3_ROADCELL_COLUMN].map(
        lambda value: _support_label_value(value, H3_ROADCELL_COLUMN, label_mappings)
    )
    cell_condition: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    for (cell, key), idx in ref.groupby([ref_cell_keys, condition_keys], dropna=False).groups.items():
        cell_condition[(str(cell), tuple(key))] = np.asarray(list(idx), dtype=int)
    cell_only = {
        str(cell): np.asarray(list(idx), dtype=int)
        for cell, idx in ref.groupby(ref_cell_keys, dropna=False).groups.items()
    }
    condition_only = {
        tuple(key): np.asarray(list(idx), dtype=int)
        for key, idx in condition_keys.groupby(condition_keys).groups.items()
    }
    global_pool = ref.index.to_numpy(dtype=int)

    stats = {
        "same_cell_condition": 0,
        "neighbor_cell_condition": 0,
        "same_cell": 0,
        "condition_only": 0,
        "global": 0,
        "invalid_cell": 0,
        "n_reference_cells": int(ref_cell_keys.nunique()),
    }
    chosen: list[int] = []
    for _, row in out.iterrows():
        cell = _support_label_value(row.get(H3_ROADCELL_COLUMN, ""), H3_ROADCELL_COLUMN, label_mappings)
        key = key_from_row(row, active_keys) if active_keys else tuple()
        pool = cell_condition.get((cell, key))
        source = "same_cell_condition"
        if not cell or cell == "__INVALID__":
            stats["invalid_cell"] += 1
            pool = None
        if pool is None or len(pool) < min_bucket_size:
            pool = None
            try:
                for ring in range(1, max_ring + 1):
                    candidates: list[int] = []
                    for neighbor in h3.grid_disk(cell, ring):
                        arr = cell_condition.get((neighbor, key))
                        if arr is not None:
                            candidates.extend(arr.tolist())
                    if len(candidates) >= min_bucket_size:
                        pool = np.asarray(candidates, dtype=int)
                        source = "neighbor_cell_condition"
                        break
            except Exception:
                pool = None
        if pool is None or len(pool) == 0:
            pool = cell_only.get(cell)
            source = "same_cell"
        if pool is None or len(pool) == 0:
            pool = condition_only.get(key)
            source = "condition_only"
        if pool is None or len(pool) == 0:
            pool = global_pool
            source = "global"
        chosen.append(int(rng.choice(pool)))
        stats[source] += 1

    sampled = ref.iloc[chosen].reset_index(drop=True)
    out["ROAD_H3_ANCHOR_CELL"] = sampled[H3_ROADCELL_COLUMN].to_numpy()
    out[H3_ROADCELL_COLUMN] = sampled[H3_ROADCELL_COLUMN].to_numpy()
    out["LATITUDE"] = sampled["LATITUDE"].to_numpy(dtype=float)
    out["LONGITUDE"] = sampled["LONGITUDE"].to_numpy(dtype=float)
    print(f"[h3_roadcell] decoded ROAD_H3_CELL anchors: {stats}")
    return out


def _load_raw_feature_defaults(data_dir: str) -> dict:
    x_num = np.load(os.path.join(data_dir, "X_num_train.npy"), allow_pickle=True)
    x_cat = np.load(os.path.join(data_dir, "X_cat_train.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True)

    cat_defaults = []
    for column_idx in range(x_cat.shape[1]):
        values, counts = np.unique(x_cat[:, column_idx], return_counts=True)
        cat_defaults.append(int(values[np.argmax(counts)]))

    return {
        "y_mean": float(np.nanmean(y)),
        "num_mean": np.nanmean(x_num, axis=0).astype(np.float32),
        "cat_mode": np.asarray(cat_defaults, dtype=np.int64),
    }


def _build_cat_value_maps(info: dict) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for col, mapping in info.get("cat_label_mappings", {}).items():
        inv: Dict[str, int] = {}
        for key, value in mapping.items():
            inv[str(value)] = int(key)
        maps[col] = inv
    return maps


def _encode_cat_value(value, label_to_code: Dict[str, int], default: int) -> int:
    if pd.isna(value):
        return default
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return default
        if float(value).is_integer():
            return int(value)
    text = str(value)
    if text in label_to_code:
        return label_to_code[text]
    try:
        return int(float(text))
    except ValueError:
        return default


def build_condition_tensors(
    condition_df: pd.DataFrame,
    encoded_dataset,
    info: dict,
    data_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    defaults = _load_raw_feature_defaults(data_dir)
    label_maps = _build_cat_value_maps(info)

    n = len(condition_df)
    num_cols = info["num_col_names"]
    cat_cols = info["cat_col_names"]
    target_col = info["target_col"]

    raw_num = np.tile(
        np.concatenate([[defaults["y_mean"]], defaults["num_mean"]]).astype(np.float32),
        (n, 1),
    )
    raw_cat = np.tile(defaults["cat_mode"], (n, 1))

    if target_col in condition_df.columns:
        target_values = pd.to_numeric(condition_df[target_col], errors="coerce").to_numpy(dtype=np.float32)
        valid = ~np.isnan(target_values)
        raw_num[valid, 0] = target_values[valid]

    for offset, col in enumerate(num_cols, start=1):
        if col not in condition_df.columns:
            continue
        values = pd.to_numeric(condition_df[col], errors="coerce").to_numpy(dtype=np.float32)
        valid = ~np.isnan(values)
        raw_num[valid, offset] = values[valid]

    for idx, col in enumerate(cat_cols):
        if col not in condition_df.columns:
            continue
        default_value = int(defaults["cat_mode"][idx])
        mapping = label_maps.get(col, {})
        raw_cat[:, idx] = np.asarray(
            [_encode_cat_value(value, mapping, default_value) for value in condition_df[col].tolist()],
            dtype=np.int64,
        )

    enc_num = raw_num.copy()
    if encoded_dataset.int_transform is not None:
        enc_num = encoded_dataset.int_transform.transform(enc_num)
    if encoded_dataset.num_transform is not None:
        enc_num = encoded_dataset.num_transform.transform(enc_num)

    enc_cat = raw_cat.copy()
    if encoded_dataset.cat_transform is not None:
        enc_cat = encoded_dataset.cat_transform.transform(enc_cat)

    return torch.tensor(enc_num).float(), torch.tensor(enc_cat).long()


@torch.no_grad()
def sample_impute_from_conditions(
    diffusion: UnifiedCtimeDiffusion,
    encoded_dataset,
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    num_mask_idx: List[int],
    cat_mask_idx: List[int],
    resample_rounds: int = 1,
    impute_condition: str = "x_0",
    w_num: float = 0.0,
    w_cat: float = 0.0,
) -> torch.Tensor:
    device = diffusion.device
    x_num = x_num.to(device).clone().float()
    x_cat = x_cat.to(device).clone().long()

    if num_mask_idx:
        avg_m = torch.tensor(
            encoded_dataset.X_num["train"][:, num_mask_idx].mean(axis=0),
            device=device,
            dtype=torch.float32,
        )
        for idx, value in zip(num_mask_idx, avg_m):
            x_num[:, idx] = value

    mi = diffusion.mask_index
    for j in cat_mask_idx:
        x_cat[:, j] = mi[j]

    try:
        out = diffusion.sample_impute(
            x_num,
            x_cat,
            num_mask_idx,
            cat_mask_idx,
            resample_rounds,
            impute_condition,
            w_num,
            w_cat,
        )
    finally:
        _reset_impute_state(diffusion)
    return out


@torch.no_grad()
def run_hierarchical_chain_sampling(
    stage1_ckpt_dir: str,
    stage1_data_dir: str,
    stage2_ckpt_dir: str,
    stage2_data_dir: str,
    stage3_ckpt_dir: str,
    stage3_data_dir: str,
    num_samples: int = 5000,
    batch_size: int = 500,
    device: str = "cuda:0",
    output_csv: str = None,
    do_postprocess: bool = True,
    impute_resample_rounds: int = 1,
    impute_condition: str = "x_0",
    road_graphml: str = None,
    road_signals: str = None,
    causal_guidance_scale: float = 0.0,
    snap_max_dist_m: float = 300.0,
    recompute_osm_after_snap: bool = True,
):
    print("=" * 60)
    print("Hierarchical CausalDiffTab - Chain Sampler")
    print("  Mode: stage1 -> stage2 -> stage3")
    print(f"  Samples: {num_samples}")
    print(f"  Device: {device}")
    print("=" * 60)

    column_groups_json = str(CDT_ROOT / "data" / "processed" / "column_groups.json")
    with open(column_groups_json, "r", encoding="utf-8") as f:
        groups = json.load(f)

    stage1_diffusion, stage1_dataset, stage1_info = load_model(stage1_ckpt_dir, stage1_data_dir, device)
    stage2_diffusion, stage2_dataset, stage2_info = load_model(stage2_ckpt_dir, stage2_data_dir, device)
    stage3_diffusion, stage3_dataset, stage3_info = load_model(stage3_ckpt_dir, stage3_data_dir, device)
    if causal_guidance_scale != 0.0:
        stage3_diffusion.causal_guidance_scale = causal_guidance_scale
        print(f"[causal_guidance] Stage3 scale={causal_guidance_scale}")

    stage2_encoded = load_forward_dataset(stage2_ckpt_dir, stage2_data_dir, stage2_info)
    stage3_encoded = load_forward_dataset(stage3_ckpt_dir, stage3_data_dir, stage3_info)

    stage1_df = sample_unconditional(
        stage1_diffusion,
        stage1_dataset,
        stage1_info,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    stage1_keep = stage1_info["num_col_names"] + stage1_info["cat_col_names"]
    stage1_df = stage1_df[stage1_keep].copy()
    stage1_df = repair_stage1_h3_roadcell_support(
        stage1_df,
        reference_csv=os.path.join(stage3_data_dir, "train.csv"),
        resolution=8,
        seed=42,
    )
    stage1_df = enrich_with_osm_context(
        stage1_df,
        stage2_info,
        stage2_data_dir,
        road_graphml=road_graphml,
        road_signals=road_signals,
    )

    stage2_num_mask, stage2_cat_mask = build_stage_impute_masks(
        stage2_info,
        groups,
        impute_stage="stage2",
        free_target=True,
    )
    stage2_num_mask, stage2_cat_mask = _freeze_impute_columns(
        stage2_num_mask,
        stage2_cat_mask,
        stage2_info,
        DETERMINISTIC_OSM_NUMERIC,
        DETERMINISTIC_OSM_CATEGORICAL,
    )
    print(
        "[stage2] frozen deterministic OSM columns: "
        f"num={sorted(DETERMINISTIC_OSM_NUMERIC & set(stage2_info['num_col_names']))}, "
        f"cat={sorted(DETERMINISTIC_OSM_CATEGORICAL & set(stage2_info['cat_col_names']))}"
    )
    stage2_chunks: List[pd.DataFrame] = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        chunk_df = stage1_df.iloc[start:end].reset_index(drop=True)
        x_num, x_cat = build_condition_tensors(chunk_df, stage2_encoded, stage2_info, stage2_data_dir)
        syn_tensor = sample_impute_from_conditions(
            stage2_diffusion,
            stage2_encoded,
            x_num,
            x_cat,
            stage2_num_mask,
            stage2_cat_mask,
            resample_rounds=impute_resample_rounds,
            impute_condition=impute_condition,
        )
        stage2_chunk = tensor_to_dataframe(syn_tensor, stage2_dataset, stage2_info)
        keep_cols = [
            col for col in stage2_info["num_col_names"] + stage2_info["cat_col_names"]
            if col not in RETIRED_CONTEXT_COLUMNS
        ]
        stage2_chunks.append(stage2_chunk[keep_cols].copy())
    stage2_df = pd.concat(stage2_chunks, ignore_index=True)

    stage3_num_mask, stage3_cat_mask = build_stage_impute_masks(
        stage3_info,
        groups,
        impute_stage="stage3",
    )
    stage3_chunks: List[pd.DataFrame] = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        chunk_df = stage2_df.iloc[start:end].reset_index(drop=True)
        x_num, x_cat = build_condition_tensors(chunk_df, stage3_encoded, stage3_info, stage3_data_dir)
        syn_tensor = sample_impute_from_conditions(
            stage3_diffusion,
            stage3_encoded,
            x_num,
            x_cat,
            stage3_num_mask,
            stage3_cat_mask,
            resample_rounds=impute_resample_rounds,
            impute_condition=impute_condition,
        )
        stage3_chunks.append(tensor_to_dataframe(syn_tensor, stage3_dataset, stage3_info))
    syn_df = pd.concat(stage3_chunks, ignore_index=True)
    syn_df = syn_df.drop(columns=list(RETIRED_CONTEXT_COLUMNS), errors="ignore")
    # 后置计算 proxy 列（消除 proxy 泄漏）
    syn_df = _compute_proxy_columns(syn_df, stage3_info)

    if not output_csv:
        output_dir = os.path.join(str(CDT_ROOT), "result", "nyc_crash", "sampled")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "samples.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    syn_df.to_csv(output_csv, index=False)
    print(f"\n[saved] Raw samples: {output_csv} ({len(syn_df)} rows)")

    base, ext = os.path.splitext(output_csv)
    stage1_csv = f"{base}_stage1_chain{ext}"
    stage2_csv = f"{base}_stage2_chain{ext}"
    stage1_df.to_csv(stage1_csv, index=False)
    stage2_df.to_csv(stage2_csv, index=False)
    print(f"[saved] Stage1 chain context: {stage1_csv}")
    print(f"[saved] Stage2 chain context: {stage2_csv}")

    if do_postprocess:
        from src.postprocess_samples import postprocess
        physical_csv = f"{base}_physical{ext}"
        print(f"\n[postprocess] Restoring physical values...")
        ref_train_csv = os.path.join(stage3_data_dir, "train.csv") if os.path.exists(os.path.join(stage3_data_dir, "train.csv")) else None
        postprocess(
            samples_csv=output_csv,
            output_csv=physical_csv,
            processed_csv=ref_train_csv,
            info_json=os.path.join(stage3_data_dir, "info.json"),
            road_graphml=road_graphml,
            road_signals=road_signals,
            snap_max_dist_m=snap_max_dist_m,
            recompute_osm_after_snap=recompute_osm_after_snap,
        )

    return syn_df


def build_stage3_impute_masks(info: dict, column_groups: dict):
    return build_stage_impute_masks(info, column_groups, impute_stage="stage3")


def _reset_impute_state(diffusion: UnifiedCtimeDiffusion) -> None:
    diffusion.w_num = 0.0
    diffusion.w_cat = 0.0
    diffusion.num_mask_idx = []
    diffusion.cat_mask_idx = []


def parse_train_indices(spec: str) -> List[int]:
    """逗号分隔，或指向每行一个整数的文本文件。"""
    p = Path(spec)
    if p.is_file():
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return [int(x.strip()) for x in lines if x.strip()]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def sample_impute_stage3(
    diffusion: UnifiedCtimeDiffusion,
    dataset: TabDiffDataset,
    info: dict,
    column_groups: dict,
    train_row_indices: List[int],
    resample_rounds: int = 1,
    impute_condition: str = "x_0",
    w_num: float = 0.0,
    w_cat: float = 0.0,
) -> torch.Tensor:
    """
    以训练集中指定行的 Stage1+2 为条件，仅对目标 y 与 Stage3 分类列做 sample_impute。
    条件行必须与 TabDiffDataset 内部张量一致（已含 TabDiff Quantile + Ordinal 编码）。
    """
    return sample_impute_by_stage(
        diffusion,
        dataset,
        info,
        column_groups,
        train_row_indices,
        impute_stage="stage3",
        resample_rounds=resample_rounds,
        impute_condition=impute_condition,
        w_num=w_num,
        w_cat=w_cat,
    )


@torch.no_grad()
def sample_impute_by_stage(
    diffusion: UnifiedCtimeDiffusion,
    dataset: TabDiffDataset,
    info: dict,
    column_groups: dict,
    train_row_indices: List[int],
    impute_stage: str,
    resample_rounds: int = 1,
    impute_condition: str = "x_0",
    w_num: float = 0.0,
    w_cat: float = 0.0,
) -> torch.Tensor:
    """
    按 stage 执行 sample_impute。

    stage2: 冻结 Stage1，重采样天气/OSM 上下文。
    stage3: 冻结 Stage1+2，重采样 y 与 Stage3 分类列。
    """
    device = diffusion.device
    d_n = dataset.d_numerical
    n = len(dataset)
    idx = [i % n for i in train_row_indices]
    X = dataset.X[idx].to(device).float()
    x_num = X[:, :d_n].clone()
    x_cat = X[:, d_n:].long()

    num_mask_idx, cat_mask_idx = build_stage_impute_masks(info, column_groups, impute_stage)

    x_train_num = dataset.X[:, :d_n].float()
    if num_mask_idx:
        avg_m = x_train_num[:, num_mask_idx].mean(dim=0).to(device)
        for k, ji in enumerate(num_mask_idx):
            x_num[:, ji] = avg_m[k]

    mi = diffusion.mask_index
    for j in cat_mask_idx:
        x_cat[:, j] = mi[j]

    try:
        out = diffusion.sample_impute(
            x_num,
            x_cat,
            num_mask_idx,
            cat_mask_idx,
            resample_rounds,
            impute_condition,
            w_num,
            w_cat,
        )
    finally:
        _reset_impute_state(diffusion)
    return out


@torch.no_grad()
def sample_conditional(
    diffusion: UnifiedCtimeDiffusion,
    condition_x0_num: torch.Tensor,
    cond_num_idx: List[int],
    free_num_idx: List[int],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    DDPM Inpainting 条件采样:
    反向去噪过程中, 每步对条件列 (Stage 1/2) 做前向加噪替换,
    仅让 Stage 3 列自由去噪。
    """
    b = condition_x0_num.shape[0]
    dtype = torch.float32
    num_timesteps = diffusion.num_timesteps

    t = torch.linspace(0, 1, num_timesteps, dtype=dtype, device=device)
    t = t[:, None]

    sigma_num_cur = diffusion.num_schedule.total_noise(t)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(t)
    sigma_num_next = torch.zeros_like(sigma_num_cur)
    sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur)
    sigma_cat_next[1:] = sigma_cat_cur[:-1]

    t_hat = t
    sigma_num_hat = sigma_num_cur
    sigma_cat_hat = sigma_cat_cur

    z_norm = torch.randn(
        (b, diffusion.num_numerical_features), device=device
    ) * sigma_num_cur[-1]

    has_cat = len(diffusion.num_classes) > 0
    z_cat = torch.zeros((b, 0), device=device).float()
    if has_cat:
        z_cat = diffusion._sample_masked_prior(b, len(diffusion.num_classes))

    pbar = tqdm(reversed(range(num_timesteps)), total=num_timesteps,
                desc="Conditional Inpainting")

    for i in pbar:
        z_norm, z_cat, q_xs = diffusion.edm_update(
            z_norm, z_cat, i,
            t[i], t[i - 1] if i > 0 else None, t_hat[i],
            sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i],
            sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
        )

        if i > 0 and cond_num_idx:
            sigma_cond = sigma_num_next[i]
            eps_num = torch.randn_like(condition_x0_num)
            x_cond_noisy = condition_x0_num + sigma_cond * eps_num
            z_norm[:, cond_num_idx] = x_cond_noisy[:, cond_num_idx]
        elif i == 0 and cond_num_idx:
            z_norm[:, cond_num_idx] = condition_x0_num[:, cond_num_idx]

    sample = torch.cat([z_norm, z_cat], dim=1).cpu()
    return sample


# ============================================================
# 4. 端到端管线
# ============================================================

def run_sampling(
    ckpt_dir: str,
    data_dir: str = None,
    condition_csv: str = None,
    condition_train_indices: str = None,
    num_samples: int = 5000,
    batch_size: int = 500,
    device: str = "cuda:0",
    output_csv: str = None,
    do_postprocess: bool = True,
    impute_resample_rounds: int = 1,
    impute_condition: str = "x_0",
    road_graphml: str = None,
    road_signals: str = None,
    snap_max_dist_m: float = 300.0,
    recompute_osm_after_snap: bool = True,
    impute_stage: str = "stage3",
    causal_guidance_scale: float = 0.0,
    macro_guidance_scale: float = 0.0,
    macro_guidance_mode: str = "absolute",
    macro_guidance_adaptive_drift_threshold: float = 2.0,
):
    """
    完整采样管线:
    - 无附加条件: 无条件 sample()
    - condition_train_indices: 训练集行号（与 TabDiffDataset 一致），对 y + Stage3 做 sample_impute
    - condition_csv: 预留；未与 TabDiff 前向编码对齐前仍回退无条件（见下方说明）
    - do_postprocess: 自动调用后处理还原物理值
    """
    data_dir = data_dir or str(CDT_ROOT / "data" / "nyc_crash")

    if condition_train_indices:
        mode_s = f"impute_{impute_stage} (train row indices)"
    elif condition_csv is not None:
        mode_s = "condition_csv -> fallback unconditional"
    else:
        mode_s = "unconditional"
    print("=" * 60)
    print("Hierarchical CausalDiffTab - Sampler")
    print(f"  Mode: {mode_s}")
    print(f"  Samples: {num_samples}")
    print(f"  Device: {device}")
    print("=" * 60)

    diffusion, dataset, info = load_model(ckpt_dir, data_dir, device)
    if causal_guidance_scale != 0.0:
        diffusion.causal_guidance_scale = causal_guidance_scale
        print(f"[causal_guidance] scale={causal_guidance_scale}")
    
    # Inference-time Macro Guidance
    if macro_guidance_scale != 0.0:
        # Auto-configure macro guidance params if missing (checkpoints don't save them)
        if diffusion.macro_injury_idx is None:
            diffusion.macro_injury_idx = 0  # y is prepended to num features
            print(f"[macro_guidance] auto-set macro_injury_idx=0")
        if diffusion.macro_group_indices is None:
            cat_names = info.get("cat_col_names", [])
            group_cols = ["SEASON", "WEATHER_CONDITION", "OSM_TYPE"]
            diffusion.macro_group_indices = []
            for col in group_cols:
                if col in cat_names:
                    diffusion.macro_group_indices.append(cat_names.index(col))
            if diffusion.macro_group_indices:
                print(f"[macro_guidance] auto-set macro_group_indices={diffusion.macro_group_indices} from {group_cols}")
            else:
                print(f"[macro_guidance] WARNING: no group columns found in cat_names, skipping")
        
        if diffusion.macro_group_indices:
            group_means_path = CDT_ROOT / "data" / "processed" / "target_group_means.json"
            if group_means_path.exists():
                with open(group_means_path, "r", encoding="utf-8") as f:
                    gm_data = json.load(f)
                diffusion.macro_guidance_group_means = {r["group_key"]: r["mean"] for r in gm_data.get("group_means", [])}
                diffusion.macro_guidance_global_mean = gm_data.get("global_mean", 0.0)
                diffusion.macro_guidance_global_std = gm_data.get("global_std", 1.0)
                diffusion.macro_guidance_scale = macro_guidance_scale
                diffusion.macro_guidance_mode = macro_guidance_mode
                diffusion.macro_guidance_adaptive_drift_threshold = macro_guidance_adaptive_drift_threshold
                print(f"[macro_guidance] scale={macro_guidance_scale}, mode={macro_guidance_mode}, groups={len(diffusion.macro_guidance_group_means)}, start_step={diffusion.macro_guidance_start_step}")
            else:
                print(f"[macro_guidance] WARNING: {group_means_path} not found, skipping")

    column_groups_json = str(CDT_ROOT / "data" / "processed" / "column_groups.json")
    with open(column_groups_json, "r", encoding="utf-8") as f:
        groups = json.load(f)

    if condition_train_indices:
        base_idx = parse_train_indices(condition_train_indices)
        if not base_idx:
            raise ValueError("condition_train_indices 解析结果为空")
        expanded = [base_idx[i % len(base_idx)] for i in range(num_samples)]
        print(f"[impute] {impute_stage} sample_impute, base indices={base_idx}, total rows={num_samples}, "
              f"batch_size={batch_size}")
        chunks = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            sub = expanded[start:end]
            t_out = sample_impute_by_stage(
                diffusion,
                dataset,
                info,
                groups,
                sub,
                impute_stage=impute_stage,
                resample_rounds=impute_resample_rounds,
                impute_condition=impute_condition,
            )
            chunks.append(t_out)
        syn_tensor = torch.cat(chunks, dim=0)
        syn_df = tensor_to_dataframe(syn_tensor, dataset, info)
        syn_df = _compute_proxy_columns(syn_df, info)
    elif condition_csv is not None:
        stage_idx = build_stage_indices(info, groups)
        print(f"[indices] cond_num: {len(stage_idx['cond_num_idx'])} dims, "
              f"free_num: {len(stage_idx['free_num_idx'])} dims")
        print(f"[indices] cond_cat: {len(stage_idx['cond_cat_idx'])} dims, "
              f"free_cat: {len(stage_idx['free_cat_idx'])} dims")
        print("[warn] condition_csv 尚未与 TabDiff 前向编码对齐，回退无条件 sample()。")
        syn_df = sample_unconditional(
            diffusion, dataset, info,
            num_samples=num_samples, batch_size=batch_size,
        )
        syn_df = _compute_proxy_columns(syn_df, info)
    else:
        syn_df = sample_unconditional(
            diffusion, dataset, info,
            num_samples=num_samples, batch_size=batch_size,
        )
        syn_df = _compute_proxy_columns(syn_df, info)

    if not output_csv:
        output_dir = os.path.join(str(CDT_ROOT), "result", "nyc_crash", "sampled")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "samples.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    syn_df.to_csv(output_csv, index=False)
    print(f"\n[saved] Raw samples: {output_csv} ({len(syn_df)} rows)")

    if do_postprocess:
        from src.postprocess_samples import postprocess
        base, ext = os.path.splitext(output_csv)
        physical_csv = f"{base}_physical{ext}"
        print(f"\n[postprocess] Restoring physical values...")
        # 用 data_dir/train.csv 作为分类映射参照（含新 schema 列名）
        ref_train_csv = os.path.join(data_dir, "train.csv") if data_dir and os.path.exists(os.path.join(data_dir, "train.csv")) else None
        postprocess(
            samples_csv=output_csv,
            output_csv=physical_csv,
            processed_csv=ref_train_csv,
            info_json=os.path.join(data_dir, "info.json") if data_dir else None,
            road_graphml=road_graphml,
            road_signals=road_signals,
            snap_max_dist_m=snap_max_dist_m,
            recompute_osm_after_snap=recompute_osm_after_snap,
        )

    return syn_df


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical CausalDiffTab - Complete Sampler"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--condition_csv", type=str, default=None)
    parser.add_argument(
        "--condition_train_indices",
        type=str,
        default=None,
        help="训练集行索引：逗号分隔，或 txt 文件每行一个整数；与 TabDiffDataset 张量一致",
    )
    parser.add_argument("--impute_resample_rounds", type=int, default=1)
    parser.add_argument("--impute_condition", type=str, default="x_0", choices=["x_0", "x_t"])
    parser.add_argument("--impute_stage", type=str, default="stage3", choices=["stage2", "stage3"],
                        help="condition_train_indices 模式下要重采样的层：stage2=天气/OSM，stage3=事故结果")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--causal_guidance_scale", type=float, default=0.0,
                        help="Causal Guidance scale (CFG-style). 0=disabled. Typical values: 0.5-2.0")
    parser.add_argument("--macro_guidance_scale", type=float, default=0.0,
                        help="Inference-time Macro Guidance scale. 0=disabled. Typical values: 0.5-2.0")
    parser.add_argument("--macro_guidance_mode", type=str, default="absolute",
                        choices=["absolute", "relative", "adaptive", "annealed"],
                        help="Guidance mode: absolute (default), relative, adaptive, annealed")
    parser.add_argument("--macro_guidance_adaptive_drift_threshold", type=float, default=2.0,
                        help="Drift threshold for adaptive mode (higher = more tolerance)")

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_sampling(
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        condition_csv=args.condition_csv,
        condition_train_indices=args.condition_train_indices,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_csv=args.output_csv,
        do_postprocess=not args.no_postprocess,
        impute_resample_rounds=args.impute_resample_rounds,
        impute_condition=args.impute_condition,
        impute_stage=args.impute_stage,
        causal_guidance_scale=args.causal_guidance_scale,
        macro_guidance_scale=args.macro_guidance_scale,
        macro_guidance_mode=args.macro_guidance_mode,
        macro_guidance_adaptive_drift_threshold=args.macro_guidance_adaptive_drift_threshold,
    )


if __name__ == "__main__":
    main()
