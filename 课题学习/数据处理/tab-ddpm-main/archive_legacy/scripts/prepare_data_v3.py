"""
prepare_data_v3.py  —  v6 迭代特征工程
========================================
从 nyc_2017_pristine_v8.csv 源数据出发，构建适用于 CausalDDPM v6 的 TabDDPM 标准数据集。

与 v2 (nyc_crash_c4) 的关键差异:
  1. 连续特征: 7→3（仅 LATITUDE, LONGITUDE, SPEED_LIMIT；天气/OSM 后处理补全）
  2. 时间特征: 3→4（新增 DAY_OF_WEEK）
  3. 多车车辆类型: 恢复 VEHICLE_TYPE_CODE_3/4/5（含 UNSPECIFIED 类别）
  4. 移除 OSM/天气相关列（后处理通过 API 补全）

输出: data/nyc_crash_v3/{X_num,X_cat,y}_{train,val,test}.npy + info.json + column_mapping.json + feature_engineering_report.json
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================
# 全局映射常量
# =============================================

# 车型合并映射（高基数 → 7 类 + Other + None）
VEHICLE_TYPE_MAP = {
    'Sedan': 'Sedan',
    'Station Wagon/Sport Utility Vehicle': 'SUV',
    'Taxi': 'Taxi',
    'Pick-up Truck': 'Pickup',
    'Box Truck': 'Truck',
    'Bus': 'Bus',
    'Bike': 'Bike',
}

# v3 新增: 车型 3/4/5 保留 Top-5 + OTHER + UNSPECIFIED
VEHICLE_345_TOP_N = 5

# IS_XXX 二元标签关键词映射（与 causal_factor_mapper.py 一致）
CAUSAL_FACTOR_KEYWORDS = {
    "IS_ALCOHOL_INVOLVED": ["alcohol involvement", "drugs (illegal)", "prescription medication"],
    "IS_SPEEDING": ["unsafe speed"],
    "IS_DISTRACTED": [
        "driver inattention/distraction", "cell phone (hand-held)",
        "cell phone (hands-free)", "passenger distraction",
        "outside car distraction", "using on board navigation device",
        "texting", "eating or drinking",
    ],
    "IS_FOLLOWING_TOO_CLOSE": ["following too closely", "passing too closely"],
    "IS_FAILURE_TO_YIELD": ["failure to yield right-of-way"],
    "IS_IMPROPER_LANE_USE": ["passing or lane usage improper", "unsafe lane changing"],
    "IS_BACKING_UNSAFE": ["backing unsafely"],
    "IS_IMPROPER_TURNING": ["turning improperly"],
    "IS_TRAFFIC_SIGNAL_VIOLATION": [
        "traffic control disregarded",
        "traffic control device improper/non-working",
        "red light violation",
    ],
    "IS_INEXPERIENCED_DRIVER": ["driver inexperience"],
    "IS_FATIGUED": ["fell asleep", "fatigued/drowsy"],
    "IS_POOR_ROAD_CONDITION": ["pavement slippery", "pavement defective", "obstruction/debris"],
    "IS_VISION_OBSCURED": ["view obstructed/limited", "glare"],
    "IS_PEDESTRIAN_CYCLIST_ERROR": ["pedestrian/bicyclist/other pedestrian error/confusion"],
    "IS_AGGRESSIVE_DRIVING": ["aggressive driving/road rage"],
    "IS_VEHICLE_DEFECT": [
        "brakes defective", "steering failure", "tire failure/inadequate",
        "accelerator defective", "headlights defective",
        "windshield inadequate", "other lighting defects", "tow hitch defective",
    ],
    "IS_OVERSIZED_VEHICLE": ["oversized vehicle"],
    "IS_OTHER_VEHICULAR": ["other vehicular"],
    "IS_ANIMAL_RELATED": ["animals action"],
    "IS_DRIVERLESS": ["driverless/runaway vehicle"],
}

SPECIAL_LABELS = {
    "IS_UNSPECIFIED": ["unspecified"],
    "IS_NONE_INVOLVED": ["none_involved"],
}

# CAUSE_XXX 特征: Top-4 + Unspecified
CAUSE_EXCLUDE = {
    "unspecified", "other vehicular", "unknown", "not applicable", "none", "",
    "none_involved", "1",
}


def _get_datetime(df):
    """安全提取 CRASH DATE + CRASH TIME → datetime Series"""
    date_col = next((c for c in df.columns if c.strip().upper() == "CRASH DATE"), None)
    time_col = next((c for c in df.columns if c.strip().upper() == "CRASH TIME"), None)
    if date_col and time_col:
        return pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    if date_col:
        return pd.to_datetime(df[date_col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


# =============================================
# A1. 时间特征派生
# =============================================
def add_time_features(df):
    """从 CRASH DATE + CRASH TIME 派生 4 个分类时间特征；删除原始时间列。"""
    out = df.copy()
    dt = _get_datetime(out)

    hour = dt.dt.hour.fillna(0).astype(int)
    month = dt.dt.month.fillna(1).astype(int)
    weekday = dt.dt.weekday.fillna(0).astype(int)  # 0=Mon ... 6=Sun

    # 季节: Spring(3-5)=1, Summer(6-8)=2, Autumn(9-11)=3, Winter(12,1,2)=0
    out["CRASH_SEASON"] = ((month % 12) // 3).astype(str)

    # 周末
    out["IS_WEEKEND"] = weekday.isin([5, 6]).astype(int).astype(str)

    # v3 新增: 周几（0=Mon ~ 6=Sun）
    out["DAY_OF_WEEK"] = weekday.astype(str)

    # 时段: Morning_Rush(7-9)=1, Midday(10-15)=2, Evening_Rush(16-19)=3, Night(20-6)=0
    bins = [-1, 6, 9, 15, 19, 23]
    labels = ["0", "1", "2", "3", "0"]  # Night wraps
    # 使用小时直接映射
    period = pd.Series("0", index=out.index)
    period[(hour >= 7) & (hour <= 9)] = "1"    # Morning Rush
    period[(hour >= 10) & (hour <= 15)] = "2"  # Midday
    period[(hour >= 16) & (hour <= 19)] = "3"  # Evening Rush
    # else: Night (0-6, 20-23) -> "0"
    out["CRASH_TIME_PERIOD"] = period

    # 删除原始时间列
    for c in ["CRASH DATE", "CRASH TIME", "CRASH DATETIME", "CRASH_FULL_TIME"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    print(f"  ⏰ 时间特征已添加: CRASH_SEASON, IS_WEEKEND, DAY_OF_WEEK, CRASH_TIME_PERIOD")
    return out


# =============================================
# A2. IS_XXX 二元标签生成
# =============================================
def add_is_labels(df):
    """从 CONTRIBUTING FACTOR VEHICLE 1-5 生成 22 个 IS_XXX 二元标签。"""
    factor_cols = [c for c in df.columns if c.startswith("CONTRIBUTING FACTOR VEHICLE")]
    if not factor_cols:
        print("  ⚠️ 未找到 CONTRIBUTING FACTOR 列，跳过 IS_XXX 标签生成")
        return df

    out = df.copy()

    # 主标签
    for label_name, keywords in CAUSAL_FACTOR_KEYWORDS.items():
        matches = pd.Series(False, index=out.index)
        for col in factor_cols:
            col_lower = out[col].fillna("").astype(str).str.lower()
            for kw in keywords:
                matches = matches | col_lower.str.contains(kw.lower(), na=False)
        out[label_name] = matches.astype(int).astype(str)

    # 特殊标签
    for label_name, keywords in SPECIAL_LABELS.items():
        matches = pd.Series(False, index=out.index)
        for col in factor_cols:
            col_lower = out[col].fillna("").astype(str).str.lower()
            for kw in keywords:
                matches = matches | col_lower.str.contains(kw.lower(), na=False)
        out[label_name] = matches.astype(int).astype(str)

    print(f"  🏷️ 已生成 {len(CAUSAL_FACTOR_KEYWORDS) + len(SPECIAL_LABELS)} 个 IS_XXX 标签")
    return out


# =============================================
# A2b. CAUSE_XXX 特征生成
# =============================================
def add_cause_features(df):
    """从 CONTRIBUTING FACTOR VEHICLE 1-5 生成 Top-4 + Unspecified CAUSE_XXX 特征。"""
    factor_cols = [c for c in df.columns if c.startswith("CONTRIBUTING FACTOR VEHICLE")]
    if not factor_cols:
        print("  ⚠️ 未找到 CONTRIBUTING FACTOR 列，跳过 CAUSE_XXX 生成")
        return df, []

    out = df.copy()

    # 收集所有原因并计频
    all_reasons = []
    for col in factor_cols:
        vals = out[col].dropna().astype(str).str.strip()
        vals = vals[~vals.str.lower().isin(CAUSE_EXCLUDE)]
        all_reasons.extend(vals.tolist())

    from collections import Counter
    reason_counts = Counter(all_reasons)
    top_reasons = [r for r, _ in reason_counts.most_common(4)]
    selected_reasons = top_reasons + ["Unspecified"]

    cause_cols = []
    for i, reason in enumerate(selected_reasons, 1):
        col_name = f"CAUSE_{i:03d}"
        cause_cols.append(col_name)
        if reason == "Unspecified":
            hit = pd.Series(False, index=out.index)
            for fc in factor_cols:
                hit = hit | (out[fc].fillna("").astype(str).str.lower() == "unspecified")
        else:
            hit = pd.Series(False, index=out.index)
            for fc in factor_cols:
                hit = hit | (out[fc].fillna("").astype(str).str.strip() == reason)
        out[col_name] = np.where(hit, "1", "0")

    print(f"  📋 CAUSE_XXX 特征: {dict(zip(cause_cols, selected_reasons))}")
    return out, cause_cols


# =============================================
# A2c. 车辆类型处理
# =============================================
def process_vehicle_types(df):
    """处理 VEHICLE TYPE CODE 1-5:
       - CODE 1 & 2: 合并至 8 类(Sedan/SUV/Taxi/Pickup/Truck/Bus/Bike/Other) + None
       - CODE 3/4/5: Top-5 + OTHER + UNSPECIFIED
    """
    out = df.copy()

    # CODE 1 & 2: 标准合并
    for col in ["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: (
                    "None"
                    if pd.isna(x) or str(x).strip() in ("", "nan", "None_Involved")
                    else VEHICLE_TYPE_MAP.get(str(x).strip(), "Other")
                )
            )

    # CODE 3/4/5: v3 新增 — 恢复并合并
    for col in ["VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4", "VEHICLE TYPE CODE 5"]:
        if col not in out.columns:
            continue

        # 非空非 UNSPECIFIED 的值
        valid_mask = out[col].notna() & (out[col].astype(str).str.strip() != "")
        vals = out.loc[valid_mask, col].astype(str).str.strip()

        # 频次 Top-5
        top5 = vals.value_counts().head(VEHICLE_345_TOP_N).index.tolist()
        print(f"  🚗 {col} Top-{VEHICLE_345_TOP_N}: {top5}")

        def _map_345(x):
            if pd.isna(x) or str(x).strip() == "":
                return "UNSPECIFIED"
            s = str(x).strip()
            if s in top5:
                return VEHICLE_TYPE_MAP.get(s, s)
            return "OTHER"

        out[col] = out[col].apply(_map_345)

    # 统一转 str
    for col in [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    vt_cols = [c for c in out.columns if c.startswith("VEHICLE TYPE CODE")]
    print(f"  🚗 车辆类型列: {vt_cols}")
    return out


# =============================================
# A3. 特征合并与删除
# =============================================
def consolidate_features_v3(df, cause_cols):
    """v3 特征合并: 聚焦减轻模型学习负担。

    v3 vs v2 差异:
      - 移除 OSM/天气列（后处理 API 补全）
      - 保留 SPEED_LIMIT 作为连续特征
      - 恢复 VEHICLE TYPE CODE 3/4/5
    """
    out = df.copy()

    # --- 1. 大量删除: ID / 伤亡子列 / 地理文本 / 天气 / OSM ---
    drop_cols = [
        # ID
        "COLLISION_ID",
        # 地理文本
        "LOCATION", "ON STREET NAME", "CROSS STREET NAME",
        "OFF STREET NAME", "ZIP CODE", "BOROUGH",
        # 天气 (后处理 API 补全)
        "TEMP_C", "prcp", "WIND_SPEED_KMH", "coco",
        "WEATHER_CONDITION", "REAL_WEATHER", "VISIBILITY_KM",
        # OSM (后处理 API 补全)
        "OSM_TYPE", "OSM_SPEED_TAG", "OSM_LANES_TAG", "OSM_ONEWAY",
        "DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL", "HAS_DIVIDER",
        "INFERRED_LANES",
        # 伤亡子列 (仅保留目标 NUMBER OF PERSONS INJURED)
        "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED",
        # 原始原因列 (已提取为 IS_XXX + CAUSE_XXX)
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
        "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4",
        "CONTRIBUTING FACTOR VEHICLE 5",
    ]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)

    # --- 2. TOTAL_VEHICLES / INFERRED_LANES → 分类 ---
    if "TOTAL_VEHICLES" in out.columns:
        out["TOTAL_VEHICLES"] = out["TOTAL_VEHICLES"].clip(upper=4).astype(int).astype(str)
    # INFERRED_LANES 已删除(OSM)，但若保留则做同样处理

    # --- 3. 二元列 → str ---
    binary_like = []
    is_cols = [c for c in out.columns if c.startswith("IS_")]
    binary_like.extend(is_cols)
    for col in binary_like:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).astype(str)

    # --- 4. CAUSE_XXX → 已经是 "0"/"1" str ---
    # (在 add_cause_features 中已处理)

    # --- 5. IS_MULTI_VEHICLE → str ---
    if "IS_MULTI_VEHICLE" in out.columns:
        out["IS_MULTI_VEHICLE"] = pd.to_numeric(out["IS_MULTI_VEHICLE"], errors="coerce").fillna(0).astype(int).astype(str)

    # --- 6. REAL_SPEED_LIMIT → 保留为连续特征 ---
    if "REAL_SPEED_LIMIT" in out.columns:
        out["REAL_SPEED_LIMIT"] = pd.to_numeric(out["REAL_SPEED_LIMIT"], errors="coerce").fillna(25.0).astype(float)

    return out


# =============================================
# A5. 主函数: 分割并保存
# =============================================
def prepare_v3_dataset(csv_path, output_dir, val_ratio=0.1, test_ratio=0.1, seed=42):
    """完整的 v3 特征工程管线。"""
    print("=" * 80)
    print(f"📂 v3 特征工程 | 输入: {csv_path}")
    print(f"📂 输出目录: {output_dir}")
    print("=" * 80)

    # 加载
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"  📊 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # Step 1: 时间特征
    print("\n--- Step 1: 时间特征 ---")
    df = add_time_features(df)

    # Step 2: IS_XXX 二元标签
    print("\n--- Step 2: IS_XXX 标签 ---")
    df = add_is_labels(df)

    # Step 3: CAUSE_XXX
    print("\n--- Step 3: CAUSE_XXX ---")
    df, cause_cols = add_cause_features(df)

    # Step 4: 车辆类型处理
    print("\n--- Step 4: 车辆类型 ---")
    df = process_vehicle_types(df)

    # Step 5: 特征合并
    print("\n--- Step 5: 特征合并 ---")
    df = consolidate_features_v3(df, cause_cols)

    # Step 6: 分离目标列
    target_col = "NUMBER OF PERSONS INJURED"
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在！可用列: {list(df.columns)}")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).values.astype(np.float32)
    df = df.drop(columns=[target_col])

    # Step 7: 分离连续/分类特征
    # 连续: LATITUDE, LONGITUDE, REAL_SPEED_LIMIT
    num_cols = ["LATITUDE", "LONGITUDE", "REAL_SPEED_LIMIT"]
    num_cols = [c for c in num_cols if c in df.columns]

    # 分类: 所有其他列
    cat_cols = [c for c in df.columns if c not in num_cols]

    # 确保分类列全部为 str
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # 连续列填充 + 转 float
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    print(f"\n--- 特征集概要 ---")
    print(f"  连续特征 ({len(num_cols)}): {num_cols}")
    print(f"  分类特征 ({len(cat_cols)}): {cat_cols}")
    print(f"  目标: {target_col}")

    # Step 8: 列名映射 (label -> index)
    column_mapping = {}
    cat_sizes = []
    for col in cat_cols:
        unique_vals = sorted(df[col].unique())
        label2idx = {v: i for i, v in enumerate(unique_vals)}
        column_mapping[col] = label2idx
        cat_sizes.append(len(unique_vals))

    # 编码分类特征为整数
    X_cat_encoded = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        mapping = column_mapping[col]
        X_cat_encoded[:, j] = df[col].map(mapping).values

    X_num = df[num_cols].values.astype(np.float32) if num_cols else None

    # Step 9: 划分 train/val/test
    n = len(df)
    indices = np.arange(n)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size

    idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=seed)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_size, random_state=seed)

    print(f"\n  划分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # Step 10: 保存
    os.makedirs(output_dir, exist_ok=True)

    for split, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if X_num is not None:
            np.save(os.path.join(output_dir, f"X_num_{split}.npy"), X_num[idx])
        np.save(os.path.join(output_dir, f"X_cat_{split}.npy"), X_cat_encoded[idx])
        np.save(os.path.join(output_dir, f"y_{split}.npy"), y[idx])

    # info.json
    info = {
        "task_type": "regression",
        "name": "nyc_crash_v3",
        "id": "nyc_crash_v3-2017",
        "train_size": len(idx_train),
        "val_size": len(idx_val),
        "test_size": len(idx_test),
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "num_columns": num_cols,
        "cat_columns": cat_cols,
        "cat_sizes": cat_sizes,
        "target_col": target_col,
    }
    with open(os.path.join(output_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # column_mapping.json
    with open(os.path.join(output_dir, "column_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(column_mapping, f, ensure_ascii=False, indent=2)

    # feature_engineering_report.json
    report = {
        "source_file": str(csv_path),
        "output_dir": str(output_dir),
        "total_samples": n,
        "splits": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
        "num_columns": num_cols,
        "cat_columns": cat_cols,
        "cat_sizes": {col: size for col, size in zip(cat_cols, cat_sizes)},
        "cat_distributions": {},
        "v2_v3_differences": {
            "num_features": "7 → 3 (移除天气/OSM，后处理补全)",
            "time_features": "3 → 4 (新增 DAY_OF_WEEK)",
            "vehicle_types": "2 → 5 (恢复 CODE 3/4/5 含 UNSPECIFIED)",
            "removed_for_postprocess": [
                "TEMP_C", "prcp", "WIND_SPEED_KMH", "REAL_WEATHER",
                "OSM_TYPE", "OSM_SPEED_TAG", "OSM_LANES_TAG", "OSM_ONEWAY",
                "DIST_TO_SIGNAL_M", "HAS_TRAFFIC_SIGNAL", "HAS_DIVIDER",
                "INFERRED_LANES", "coco", "WEATHER_CONDITION",
            ],
        },
    }

    # 各分类列的分布
    for col in cat_cols:
        dist = df[col].value_counts(normalize=True).to_dict()
        report["cat_distributions"][col] = {str(k): round(v, 4) for k, v in dist.items()}

    with open(os.path.join(output_dir, "feature_engineering_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ v3 数据集已保存到 {output_dir}")
    print(f"   info.json:                    {len(num_cols)} 连续 + {len(cat_cols)} 分类")
    print(f"   column_mapping.json:          {len(column_mapping)} 列映射")
    print(f"   feature_engineering_report.json: 完成")
    return info


def main():
    parser = argparse.ArgumentParser(description="v3 特征工程: 构建 nyc_crash_v3 数据集")
    parser.add_argument(
        "--csv", type=str,
        default="nyc_2017_pristine_v8.csv",
        help="源 CSV 文件路径",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/nyc_crash_v3",
        help="输出目录",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_v3_dataset(args.csv, args.output, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
