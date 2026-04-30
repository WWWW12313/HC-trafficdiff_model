"""
Hierarchical CausalDiffTab - 数据预处理模块
============================================
将 NYC Motor Vehicle Collisions 2017 原始/富化数据，
按照分层生成架构(Spatial → Temporal → Crash Typology)的需求，
进行多粒度特征工程与变量类型划分。

输入: nyc_2017_pristine_v9.csv (已融合 OSM + 天气的富化表)
输出: 预处理后的 DataFrame + continuous_cols / categorical_cols 定义
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 常量与映射表
# ============================================================

VEHICLE_TYPE_GROUPS: Dict[str, List[str]] = {
    "is_sedan": ["Sedan", "SEDAN", "4 dr sedan", "2 dr sedan"],
    "is_suv": ["Station Wagon/Sport Utility Vehicle", "SPORT UTILITY / STATION WAGON"],
    "is_taxi": ["Taxi", "TAXI", "Livery Vehicle", "LIVERY VEHICLE"],
    "is_truck": [
        "Tractor Truck Diesel", "Tractor Truck Gasoline",
        "Box Truck", "BOX TRUCK", "Flat Bed",
        "LARGE COM VEH(6 OR MORE TIRES)", "SMALL COM VEH(4 TIRES)",
        "Carry All", "Dump", "Garbage or Refuse", "Tanker", "Concrete Mixer",
    ],
    "is_bus": ["Bus", "BUS", "School Bus", "OMNIBUS", "VAN/SHUTTLE/BUS"],
    "is_motorcycle": [
        "Motorcycle", "MOTORCYCLE", "Motorbike", "Moped",
        "E-Scooter", "Motorscooter",
    ],
    "is_bicycle": ["Bicycle", "Bike", "E-Bike", "BICYCLE"],
}

VEHICLE_OTHER_INDICATOR_NAME = "is_other_vehicle"

CONTRIBUTING_FACTOR_GROUPS: Dict[str, List[str]] = {
    "is_distracted": [
        "Driver Inattention/Distraction",
        "Cell Phone (hand-Held)",
        "Cell Phone (hands-free)",
        "Using On Board Electronic Device",
        "Texting",
        "Listening/Using Headphones",
        "Outside Car Distraction",
    ],
    "is_speeding": [
        "Unsafe Speed",
        "Aggressive Driving/Road Rage",
    ],
    "is_failure_to_yield": [
        "Failure to Yield Right-of-Way",
        "Failure to Keep Right",
        "Traffic Control Disregarded",
    ],
    "is_following_too_closely": [
        "Following Too Closely",
        "Unsafe Lane Changing",
        "Passing or Lane Usage Improper",
        "Passing Too Closely",
    ],
    "is_drunk_driving": [
        "Alcohol Involvement",
        "Drugs (illegal)",
        "Drugs (Illegal)",
        "Prescription Medication",
    ],
    "is_fatigue": [
        "Fatigued/Drowsy",
        "Lost Consciousness",
        "Fell Asleep",
    ],
    "is_view_obstructed": [
        "View Obstructed/Limited",
        "Glare",
        "Obstruction/Debris",
        "Windshield Inadequate",
    ],
    "is_vehicle_defect": [
        "Brakes Defective",
        "Steering Failure",
        "Tire Failure/Inadequate",
        "Accelerator Defective",
        "Headlights Defective",
        "Other Lighting Defects",
        "Tow Hitch Defective",
    ],
    "is_backing_unsafely": [
        "Backing Unsafely",
        "Turning Improperly",
    ],
    "is_pedestrian_related": [
        "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion",
        "Pedestrians in Roadway (not intersection)",
    ],
    "is_inexperience": [
        "Driver Inexperience",
    ],
    "is_pavement_slippery": [
        "Pavement Slippery",
    ],
}

VEHICLE_CODE_COLS = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
CONTRIBUTING_FACTOR_COLS = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)]

# 时段划分 (24h)
TIME_PERIOD_MAP = {
    range(5, 8): "dawn",        # 05:00-07:59  黎明
    range(8, 12): "morning",    # 08:00-11:59  上午
    range(12, 17): "afternoon", # 12:00-16:59  下午
    range(17, 21): "evening",   # 17:00-20:59  傍晚
}
# 21:00-04:59 归为 night


def _get_time_period(hour: int) -> str:
    for hour_range, label in TIME_PERIOD_MAP.items():
        if hour in hour_range:
            return label
    return "night"


SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# ============================================================
# 三阶段特征分组 (Hierarchical Generation Pipeline)
# Stage 1: 时空锚点 / 外生变量 (扩散模型首先联合生成)
# Stage 2: 离线上下文 / 环境变量 (不由扩散模型生成，基于 Stage 1 的
#           时空锚点查询本地离线数据库：Open-Meteo 天气、OSM 路网；
#           BOROUGH 也在此阶段通过 geopandas Point-in-Polygon 反查补全)
# Stage 3: 事故微观特征 / 内生变量 (以 Stage 1+2 为条件)
# ============================================================

STAGE1_CONTINUOUS = ["LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS"]
STAGE1_CATEGORICAL = ["SEASON", "DAY_OF_WEEK", "TIME_PERIOD"]

STAGE2_CONTINUOUS = [
    "TEMP_C", "prcp", "WIND_SPEED_KMH",
    "DIST_TO_SIGNAL_M", "REAL_SPEED_LIMIT", "INFERRED_LANES",
]
STAGE2_CATEGORICAL = [
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "HAS_DIVIDER",
    "coco", "WEATHER_CONDITION", "OSM_TYPE",
]

VEHICLE_TYPE_INDICATOR_NAMES = list(VEHICLE_TYPE_GROUPS.keys()) + [VEHICLE_OTHER_INDICATOR_NAME]
FACTOR_INDICATOR_NAMES = list(CONTRIBUTING_FACTOR_GROUPS.keys())

# Phase 2：crash_type 列（事故形态四类指示变量，均为 binary）
CRASH_TYPE_INDICATOR_NAMES = [
    "is_rear_end",
    "is_lane_change_related",
    "is_pedestrian_involved",
    "is_cyclist_involved",
]

INJURY_BIN_COLS = [
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN", "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN", "NUMBER_OF_MOTORIST_KILLED_BIN",
]

# ============================================================
# 2. 核心处理函数
# ============================================================


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """读取 CSV 并做基础清洗"""
    df = pd.read_csv(csv_path, low_memory=False)
    n_before = len(df)

    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    df = df[(df["LATITUDE"] != 0) & (df["LONGITUDE"] != 0)]

    nyc_lat_bounds = (40.45, 40.95)
    nyc_lon_bounds = (-74.30, -73.65)
    df = df[
        (df["LATITUDE"].between(*nyc_lat_bounds))
        & (df["LONGITUDE"].between(*nyc_lon_bounds))
    ]

    # BOROUGH 完全由经纬度通过 Point-in-Polygon 推导，不参与扩散训练；
    # 将在 Stage 2 离线后处理中通过 geopandas 反查补全。
    df.drop(columns=["BOROUGH"], inplace=True, errors="ignore")

    n_after = len(df)
    print(f"[load] {n_before} -> {n_after} rows (dropped {n_before - n_after} invalid/out-of-bounds)")
    return df.reset_index(drop=True)


# ---------- A. 空间特征 ----------

def process_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """保留 LATITUDE/LONGITUDE，后续统一做归一化"""
    df["LATITUDE"] = df["LATITUDE"].astype(np.float64)
    df["LONGITUDE"] = df["LONGITUDE"].astype(np.float64)
    return df


# ---------- B. 时间特征 ----------

def process_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 CRASH DATE / CRASH TIME 中拆出多粒度时间特征:
      - SEASON          (categorical)
      - DAY_OF_WEEK     (categorical)
      - TIME_PERIOD     (categorical)
      - CRASH_TIME_SIN  (continuous, 周期编码)
      - CRASH_TIME_COS  (continuous, 周期编码)
    """
    crash_dt = pd.to_datetime(
        df["CRASH DATE"].astype(str) + " " + df["CRASH TIME"].astype(str),
        errors="coerce",
    )
    df["_crash_dt"] = crash_dt

    df["SEASON"] = crash_dt.dt.month.map(SEASON_MAP)

    df["DAY_OF_WEEK"] = crash_dt.dt.dayofweek  # 0=Mon ... 6=Sun

    hours = crash_dt.dt.hour
    df["TIME_PERIOD"] = hours.apply(_get_time_period)

    minutes_of_day = hours * 60 + crash_dt.dt.minute  # 0-1439
    frac = minutes_of_day / 1440.0  # 归一化到 [0, 1)
    df["CRASH_TIME_SIN"] = np.sin(2 * np.pi * frac)
    df["CRASH_TIME_COS"] = np.cos(2 * np.pi * frac)

    df.drop(columns=["_crash_dt"], inplace=True, errors="ignore")
    return df


# ---------- C. 事故类型二元拆分 ----------

def _build_binary_indicators(
    df: pd.DataFrame,
    source_cols: List[str],
    group_map: Dict[str, List[str]],
    prefix: str = "",
) -> pd.DataFrame:
    """
    将多个高基数类别列聚合成 is_XXX 二元指示列。
    只要 source_cols 中任一列匹配到 group 关键词，该行的 is_XXX = 1。
    """
    combined = df[source_cols].fillna("").astype(str)
    for indicator_name, keywords in group_map.items():
        col_name = f"{prefix}{indicator_name}" if prefix else indicator_name
        kw_lower = {k.lower() for k in keywords}
        mask = combined.apply(
            lambda row: any(
                cell.strip().lower() in kw_lower for cell in row
            ),
            axis=1,
        )
        df[col_name] = mask.astype(np.int8)
    return df


def _add_other_vehicle_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """将未归入主车辆类别的长尾车辆类型合并为 is_other_vehicle。"""
    existing_cols = [c for c in VEHICLE_CODE_COLS if c in df.columns]
    if not existing_cols:
        df[VEHICLE_OTHER_INDICATOR_NAME] = 0
        return df

    main_keywords = {
        keyword.lower()
        for keywords in VEHICLE_TYPE_GROUPS.values()
        for keyword in keywords
    }
    combined = df[existing_cols].fillna("").astype(str)
    mask = combined.apply(
        lambda row: any(
            (cell := value.strip().lower()) and cell not in main_keywords
            for value in row
        ),
        axis=1,
    )
    df[VEHICLE_OTHER_INDICATOR_NAME] = mask.astype(np.int8)
    return df


def process_crash_typology(df: pd.DataFrame) -> pd.DataFrame:
    """对车辆类型和事故原因分别做 is_XXX 二元降维"""
    df = _build_binary_indicators(
        df, VEHICLE_CODE_COLS, VEHICLE_TYPE_GROUPS, prefix=""
    )
    df = _add_other_vehicle_indicator(df)
    df = _build_binary_indicators(
        df, CONTRIBUTING_FACTOR_COLS, CONTRIBUTING_FACTOR_GROUPS, prefix=""
    )

    for col_base in [
        "NUMBER OF PEDESTRIANS INJURED",
        "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST INJURED",
        "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST INJURED",
        "NUMBER OF MOTORIST KILLED",
    ]:
        if col_base in df.columns:
            bin_col = col_base.replace(" ", "_") + "_BIN"
            vals = pd.to_numeric(df[col_base], errors="coerce").fillna(0)
            df[bin_col] = (vals > 0).astype(np.int8)

    if "TOTAL_VEHICLES" not in df.columns:
        non_empty = df[VEHICLE_CODE_COLS].notna() & (df[VEHICLE_CODE_COLS] != "")
        df["TOTAL_VEHICLES"] = non_empty.sum(axis=1)
    if "IS_MULTI_VEHICLE" not in df.columns:
        df["IS_MULTI_VEHICLE"] = (
            pd.to_numeric(df["TOTAL_VEHICLES"], errors="coerce").fillna(0) > 1
        ).astype(np.int8)

    return df


# ---------- D. 离线物理上下文 (如果 v9 富化表已有则直接选用) ----------

def process_offline_context(df: pd.DataFrame) -> pd.DataFrame:
    """对已有的天气/道路上下文列做数值标准化或类别编码"""
    ctx_num_cols = [
        "TEMP_C", "prcp", "WIND_SPEED_KMH",
        "DIST_TO_SIGNAL_M", "REAL_SPEED_LIMIT", "INFERRED_LANES",
    ]
    for col in ctx_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ctx_bin_cols = ["HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "HAS_DIVIDER"]
    for col in ctx_bin_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)

    return df


# ============================================================
# 3. 列分组定义
# ============================================================


def define_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    返回按三阶段组织的变量分组:
      Stage 1: 时空锚点 (扩散模型首先生成)
      Stage 2: 离线上下文 (基于 Stage 1 查询离线数据库)
      Stage 3: 事故微观特征 (以 Stage 1+2 为条件生成)
    """
    _exist = lambda cols: [c for c in cols if c in df.columns]

    # ---- Stage 1: 时空锚点 / 外生变量 ----
    s1_cont = _exist(STAGE1_CONTINUOUS)
    s1_cat = _exist(STAGE1_CATEGORICAL)

    # ---- Stage 2: 离线上下文 / 环境变量 ----
    s2_cont = _exist(STAGE2_CONTINUOUS)
    s2_cat = _exist(STAGE2_CATEGORICAL)

    # ---- Stage 3: 事故微观特征 / 内生变量 ----
    vehicle_binary = _exist(VEHICLE_TYPE_INDICATOR_NAMES)
    factor_binary = _exist(FACTOR_INDICATOR_NAMES)
    injury_binary = _exist(INJURY_BIN_COLS)
    misc_s3 = _exist(["TOTAL_VEHICLES", "IS_MULTI_VEHICLE"])
    s3_cat = vehicle_binary + factor_binary + injury_binary + misc_s3

    # ---- 汇总 ----
    all_continuous = s1_cont + s2_cont
    all_categorical = s1_cat + s2_cat + s3_cat

    groups = {
        "stage1_continuous": s1_cont,
        "stage1_categorical": s1_cat,
        "stage2_continuous": s2_cont,
        "stage2_categorical": s2_cat,
        "stage3_categorical": s3_cat,
        "vehicle_binary": vehicle_binary,
        "factor_binary": factor_binary,
        "injury_binary": injury_binary,
        "stage1_features": s1_cont + s1_cat,
        "stage2_features": s2_cont + s2_cat,
        "stage3_features": s3_cat,
        "continuous_cols": all_continuous,
        "categorical_cols": all_categorical,
    }

    return groups


# ============================================================
# 4. 归一化
# ============================================================


def normalize_continuous(
    df: pd.DataFrame,
    continuous_cols: List[str],
    method: str = "quantile",
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, object, List[str]]:
    """
    对连续列做归一化，自动剔除全 NaN 或零方差列。
    method='quantile' 使用 QuantileTransformer(output_distribution='normal')，
    确保输出完全服从标准正态分布，与 EDM 扩散模型的高斯先验完美对齐。
    返回 (df, scaler, dropped_cols)
    """
    if method == "quantile":
        scaler = QuantileTransformer(
            output_distribution="normal", random_state=42, n_quantiles=1000
        )
    elif method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    valid_cols = [c for c in continuous_cols if c in df.columns]

    dropped = []
    for col in valid_cols[:]:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            print(f"  [drop] {col}: all NaN")
            dropped.append(col)
            valid_cols.remove(col)
        elif series.std() == 0:
            print(f"  [drop] {col}: zero variance (const={series.iloc[0]})")
            dropped.append(col)
            valid_cols.remove(col)

    df[valid_cols] = df[valid_cols].fillna(df[valid_cols].median())
    df[valid_cols] = scaler.fit_transform(df[valid_cols])

    # Persist scaler for inverse_transform during sampling
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, "continuous_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler": scaler, "columns": valid_cols, "method": method}, f)
        print(f"  [save] Scaler ({method}) -> {scaler_path}")

    return df, scaler, dropped


# ============================================================
# 5. 导出为 CausalDiffTab 格式 (npy split)
# ============================================================


def export_for_causaldifftab(
    df: pd.DataFrame,
    column_groups: Dict[str, List[str]],
    output_dir: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> None:
    """
    将预处理后的 DataFrame 按 CausalDiffTab 的 npy 格式导出:
      X_num_{train/val/test}.npy
      X_cat_{train/val/test}.npy
      y_{train/val/test}.npy
      info.json
    """
    os.makedirs(output_dir, exist_ok=True)

    cont_cols = column_groups["continuous_cols"]
    cat_cols = column_groups["categorical_cols"]

    valid_cont = [c for c in cont_cols if c in df.columns]
    valid_cat = [c for c in cat_cols if c in df.columns]

    X_num_all = df[valid_cont].values.astype(np.float32)

    cat_encoded = df[valid_cat].copy()
    for col in valid_cat:
        cat_encoded[col] = cat_encoded[col].astype("category").cat.codes
    X_cat_all = cat_encoded.values.astype(np.int64)

    # y = NUMBER OF PERSONS INJURED (与之前 v7 保持一致)
    if "NUMBER OF PERSONS INJURED" in df.columns:
        y_all = pd.to_numeric(
            df["NUMBER OF PERSONS INJURED"], errors="coerce"
        ).fillna(0).values.astype(np.float32)
    else:
        y_all = np.zeros(len(df), dtype=np.float32)

    # 划分 train / val / test
    idx = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, random_state=seed
    )

    for split_name, split_idx in [
        ("train", idx_train), ("val", idx_val), ("test", idx_test)
    ]:
        np.save(os.path.join(output_dir, f"X_num_{split_name}.npy"), X_num_all[split_idx])
        np.save(os.path.join(output_dir, f"X_cat_{split_name}.npy"), X_cat_all[split_idx])
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y_all[split_idx])

    # category sizes
    cat_sizes = []
    for col in valid_cat:
        cat_sizes.append(int(df[col].astype("category").cat.categories.size))

    info = {
        "task_type": "regression",
        "n_num_features": len(valid_cont),
        "n_cat_features": len(valid_cat),
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "n_classes": None,
        "num_col_names": valid_cont,
        "cat_col_names": valid_cat,
        "cat_sizes": cat_sizes,
        "column_groups": {
            k: v for k, v in column_groups.items()
            if k not in ("continuous_cols", "categorical_cols")
        },
        "target_col": "NUMBER OF PERSONS INJURED",
    }

    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[export] Saved to {output_dir}/")
    print(f"  X_num: {X_num_all.shape[1]} cols, X_cat: {X_cat_all.shape[1]} cols")
    print(f"  train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")


# ============================================================
# 6. 主流程
# ============================================================


def run_preprocessing(
    input_csv: str,
    output_dir: str,
    norm_method: str = "quantile",
    export_npy: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    完整预处理流水线:
      1. 加载并清洗
      2. 空间特征处理
      3. 时间特征拆解
      4. 事故类型二元拆分
      5. 离线上下文处理
      6. 列分组定义
      7. 连续变量归一化
      8. (可选) 导出 npy
    """
    print("=" * 60)
    print("Hierarchical CausalDiffTab - Data Preprocessor")
    print("=" * 60)

    # Step 1
    df = load_raw_data(input_csv)

    # Step 2
    df = process_spatial_features(df)

    # Step 3
    df = process_temporal_features(df)

    # Step 4: 车辆类型 + 事故原因二元降维
    df = process_crash_typology(df)

    # Step 5
    df = process_offline_context(df)

    # Step 6
    column_groups = define_column_groups(df)

    print("\n--- Column Groups ---")
    for group_name, cols in column_groups.items():
        print(f"  {group_name}: {len(cols)} cols -> {cols[:5]}{'...' if len(cols) > 5 else ''}")

    # Step 7 (归一化连续列，自动剔除无效列)
    df_normalized = df.copy()
    df_normalized, scaler, dropped_cont = normalize_continuous(
        df_normalized, column_groups["continuous_cols"],
        method=norm_method, output_dir=output_dir,
    )

    # 同步移除被剔除的列
    if dropped_cont:
        column_groups["continuous_cols"] = [
            c for c in column_groups["continuous_cols"] if c not in dropped_cont
        ]
        for grp in ["stage1_continuous", "stage2_continuous"]:
            column_groups[grp] = [
                c for c in column_groups[grp] if c not in dropped_cont
            ]
        column_groups["stage1_features"] = [
            c for c in column_groups["stage1_features"] if c not in dropped_cont
        ]
        column_groups["stage2_features"] = [
            c for c in column_groups["stage2_features"] if c not in dropped_cont
        ]
        print(f"  Dropped {len(dropped_cont)} degenerate continuous cols: {dropped_cont}")

    # 同样检查分类列的零方差
    dropped_cat = []
    for col in column_groups["categorical_cols"][:]:
        if col in df_normalized.columns:
            n_unique = df_normalized[col].nunique()
            if n_unique <= 1:
                print(f"  [drop] {col}: only {n_unique} unique value(s)")
                dropped_cat.append(col)
    if dropped_cat:
        column_groups["categorical_cols"] = [
            c for c in column_groups["categorical_cols"] if c not in dropped_cat
        ]
        for grp in ["stage1_categorical", "stage2_categorical", "stage3_categorical",
                     "vehicle_binary", "factor_binary", "injury_binary"]:
            column_groups[grp] = [
                c for c in column_groups.get(grp, []) if c not in dropped_cat
            ]
        for sf in ["stage1_features", "stage2_features", "stage3_features"]:
            column_groups[sf] = [
                c for c in column_groups.get(sf, []) if c not in dropped_cat
            ]
        print(f"  Dropped {len(dropped_cat)} degenerate categorical cols: {dropped_cat}")

    # 保存中间产物
    processed_csv = os.path.join(output_dir, "processed_hierarchical.csv")
    os.makedirs(output_dir, exist_ok=True)
    df_normalized.to_csv(processed_csv, index=False)
    print(f"\n[save] Processed CSV -> {processed_csv}")

    # 保存列分组 JSON
    groups_json = os.path.join(output_dir, "column_groups.json")
    with open(groups_json, "w", encoding="utf-8") as f:
        json.dump(column_groups, f, indent=2, ensure_ascii=False)
    print(f"[save] Column groups -> {groups_json}")

    # Step 8
    if export_npy:
        npy_dir = os.path.join(output_dir, "npy")
        export_for_causaldifftab(
            df_normalized, column_groups, npy_dir, seed=seed
        )

    return df_normalized, column_groups


# ============================================================
# 7. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical CausalDiffTab Data Preprocessor"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "tab-ddpm-main" / "data" / "processed" / "nyc_2017_pristine_v9.csv"
        ),
        help="Path to the enriched v9 CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "processed"),
        help="Directory to save processed outputs",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="quantile",
        choices=["quantile", "standard", "minmax"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_export_npy", action="store_true")

    args = parser.parse_args()

    df, groups = run_preprocessing(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        norm_method=args.norm_method,
        export_npy=not args.no_export_npy,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("continuous_cols =", groups["continuous_cols"])
    print("categorical_cols =", groups["categorical_cols"])
    print(f"\nTotal features: {len(groups['continuous_cols'])} continuous + {len(groups['categorical_cols'])} categorical")
    print("=" * 60)


if __name__ == "__main__":
    main()
