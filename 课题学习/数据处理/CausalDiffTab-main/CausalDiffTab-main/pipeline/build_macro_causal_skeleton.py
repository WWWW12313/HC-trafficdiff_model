"""
宏观因果骨架构建器（7 条主骨架边，三档软约束）
===============================================
将 222 边密集因果图替换为 7 组宏观骨架边，适配 2024→2025 迁移学习场景。

骨架设计原则（对应 2026-04-26 提示词 2.5 节）：
  保留 7 条主骨架:
    1. 时间 -> 事故类型         [强, 1.0]
    2. 时间 -> 伤亡粗结果       [弱, 0.5]
    3. 空间 -> 路网属性         [强, 1.0]
    4. 天气/路网 -> 事故类型    [弱, 0.5]
    5. 车辆类型 -> 事故类型     [强, 1.0]
    6. 车辆类型 -> 伤亡结果     [弱, 0.5]
    7. 事故类型 -> 伤亡结果     [强, 1.0]

新版字段（38 列）顺序：
  连续 (0-8):   LATITUDE LONGITUDE CRASH_TIME_SIN CRASH_TIME_COS
                TEMP_C prcp WIND_SPEED_KMH DIST_TO_SIGNAL_M INFERRED_LANES
  离散 (9-37):  SEASON IS_WEEKEND IS_AM_PEAK IS_PM_PEAK
                HAS_TRAFFIC_SIGNAL OSM_ONEWAY WEATHER_CONDITION OSM_TYPE
                is_rear_end is_lane_change_related is_pedestrian_involved is_cyclist_involved
                is_sedan is_suv is_taxi is_truck is_pickup is_bus is_van is_motorcycle is_bicycle
                NUMBER_OF_PEDESTRIANS_INJURED_BIN NUMBER_OF_PEDESTRIANS_KILLED_BIN
                NUMBER_OF_CYCLIST_INJURED_BIN NUMBER_OF_CYCLIST_KILLED_BIN
                NUMBER_OF_MOTORIST_INJURED_BIN NUMBER_OF_MOTORIST_KILLED_BIN
                TOTAL_VEHICLES IS_MULTI_VEHICLE

用法:
  python pipeline/build_macro_causal_skeleton.py
  python pipeline/build_macro_causal_skeleton.py --out_prefix configs/causal_matrix_macro
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

CDT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# 列定义（必须与 column_groups.json 中 continuous_cols + categorical_cols 一致）
# ──────────────────────────────────────────────────────────────────────────────

CONTINUOUS = [
    "LATITUDE", "LONGITUDE", "CRASH_TIME_SIN", "CRASH_TIME_COS",
    "TEMP_C", "prcp", "WIND_SPEED_KMH", "DIST_TO_SIGNAL_M", "INFERRED_LANES",
]

CATEGORICAL = [
    "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK",
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "WEATHER_CONDITION", "OSM_TYPE",
    "is_rear_end", "is_lane_change_related", "is_pedestrian_involved", "is_cyclist_involved",
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
    "is_bus", "is_van", "is_motorcycle", "is_bicycle",
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",     "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",    "NUMBER_OF_MOTORIST_KILLED_BIN",
    "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
]

ALL_COLS = CONTINUOUS + CATEGORICAL   # 38 列
COL_IDX  = {c: i for i, c in enumerate(ALL_COLS)}
N        = len(ALL_COLS)              # 38

# ──────────────────────────────────────────────────────────────────────────────
# 节点组定义
# ──────────────────────────────────────────────────────────────────────────────

def _idx(*names) -> List[int]:
    return [COL_IDX[n] for n in names if n in COL_IDX]


# 时间节点（含连续圆周编码 + 离散语义）
TIME_NODES = _idx(
    "CRASH_TIME_SIN", "CRASH_TIME_COS",   # 连续
    "SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK",  # 离散
)
# 时间语义节点（用于弱边，避免低信噪比的连续编码产生杂连接）
TIME_SEMANTIC_NODES = _idx("SEASON", "IS_WEEKEND", "IS_AM_PEAK", "IS_PM_PEAK")

# 空间节点
SPATIAL_NODES = _idx("LATITUDE", "LONGITUDE")

# 天气节点
WEATHER_NODES = _idx("TEMP_C", "prcp", "WIND_SPEED_KMH", "WEATHER_CONDITION")

# 路网属性节点
ROAD_NODES = _idx(
    "DIST_TO_SIGNAL_M", "INFERRED_LANES",
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY", "OSM_TYPE",
)

# 宏观事故类型节点
ACCIDENT_TYPE_NODES = _idx(
    "is_rear_end", "is_lane_change_related",
    "is_pedestrian_involved", "is_cyclist_involved",
)

# 车辆类型节点
VEHICLE_TYPE_NODES = _idx(
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
    "is_bus", "is_van", "is_motorcycle", "is_bicycle",
)

# 伤亡结果节点
INJURY_NODES = _idx(
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",     "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",    "NUMBER_OF_MOTORIST_KILLED_BIN",
)

# 规模节点
SCALE_NODES = _idx("TOTAL_VEHICLES", "IS_MULTI_VEHICLE")


# ──────────────────────────────────────────────────────────────────────────────
# 骨架边构建
# ──────────────────────────────────────────────────────────────────────────────

STRONG = 1.0   # 强保留边：核心主链条，跨年份高稳定性
WEAK   = 0.5   # 弱约束边：有合理依据但不是核心链条，作为 soft 正则化
# 删除边 = 0.0（即不出现在矩阵中）


def _add_group_edges(
    W: np.ndarray,
    sources: List[int],
    targets: List[int],
    strength: float,
) -> None:
    """将 sources×targets 的边批量写入邻接矩阵 W。"""
    for s in sources:
        for t in targets:
            if s != t:                  # 不允许自环
                W[s, t] = max(W[s, t], strength)   # 取较大值（避免覆盖强边）


def build_macro_skeleton() -> Tuple[np.ndarray, List[dict]]:
    """
    构建 38×38 宏观因果骨架邻接矩阵。
    返回 (W, edge_list)。
    """
    W = np.zeros((N, N), dtype=np.float32)
    edge_list = []

    # ── 骨架 1: 时间 -> 事故类型  [强] ───────────────────────────────────────
    _add_group_edges(W, TIME_NODES, ACCIDENT_TYPE_NODES, STRONG)
    edge_list.append({
        "backbone": 1,
        "description": "时间 -> 事故类型",
        "sources": [ALL_COLS[i] for i in TIME_NODES],
        "targets": [ALL_COLS[i] for i in ACCIDENT_TYPE_NODES],
        "strength": STRONG,
        "tier": "强保留",
        "rationale": "一天中不同时段、周末、高峰期决定驾驶行为模式，直接影响事故类型",
    })

    # ── 骨架 2: 时间语义 -> 伤亡粗结果  [弱] ─────────────────────────────────
    _add_group_edges(W, TIME_SEMANTIC_NODES, INJURY_NODES, WEAK)
    edge_list.append({
        "backbone": 2,
        "description": "时间语义 -> 伤亡结果（粗）",
        "sources": [ALL_COLS[i] for i in TIME_SEMANTIC_NODES],
        "targets": [ALL_COLS[i] for i in INJURY_NODES],
        "strength": WEAK,
        "tier": "弱约束",
        "rationale": "时段影响车速和道路密度，间接影响伤亡严重程度；相关性较弱",
    })

    # ── 骨架 3: 空间 -> 路网属性  [强] ────────────────────────────────────────
    _add_group_edges(W, SPATIAL_NODES, ROAD_NODES, STRONG)
    edge_list.append({
        "backbone": 3,
        "description": "空间 -> 路网属性",
        "sources": [ALL_COLS[i] for i in SPATIAL_NODES],
        "targets": [ALL_COLS[i] for i in ROAD_NODES],
        "strength": STRONG,
        "tier": "强保留",
        "rationale": "经纬度直接决定所在路段的信号灯、车道数、道路类型等基础设施属性",
    })

    # ── 骨架 4: 天气/路网 -> 事故类型  [弱] ──────────────────────────────────
    _add_group_edges(W, WEATHER_NODES, ACCIDENT_TYPE_NODES, WEAK)
    _add_group_edges(W, ROAD_NODES,    ACCIDENT_TYPE_NODES, WEAK)
    edge_list.append({
        "backbone": 4,
        "description": "天气/路网 -> 事故类型",
        "sources": [ALL_COLS[i] for i in WEATHER_NODES + ROAD_NODES],
        "targets": [ALL_COLS[i] for i in ACCIDENT_TYPE_NODES],
        "strength": WEAK,
        "tier": "弱约束",
        "rationale": "雨雪天气和信号灯密度等路网条件影响事故类型分布；但跨年份变化较小",
    })

    # ── 骨架 5: 车辆类型 -> 事故类型  [强] ────────────────────────────────────
    _add_group_edges(W, VEHICLE_TYPE_NODES, ACCIDENT_TYPE_NODES, STRONG)
    # 车辆类型也直接影响事故规模
    _add_group_edges(W, VEHICLE_TYPE_NODES, SCALE_NODES, STRONG)
    edge_list.append({
        "backbone": 5,
        "description": "车辆类型 -> 事故类型 + 规模",
        "sources": [ALL_COLS[i] for i in VEHICLE_TYPE_NODES],
        "targets": [ALL_COLS[i] for i in ACCIDENT_TYPE_NODES + SCALE_NODES],
        "strength": STRONG,
        "tier": "强保留",
        "rationale": "车辆类型直接影响事故模式（自行车→涉骑行事故，货车→多车碰撞等）",
    })

    # ── 骨架 6: 车辆类型 -> 伤亡结果  [弱] ────────────────────────────────────
    _add_group_edges(W, VEHICLE_TYPE_NODES, INJURY_NODES, WEAK)
    edge_list.append({
        "backbone": 6,
        "description": "车辆类型 -> 伤亡结果",
        "sources": [ALL_COLS[i] for i in VEHICLE_TYPE_NODES],
        "targets": [ALL_COLS[i] for i in INJURY_NODES],
        "strength": WEAK,
        "tier": "弱约束",
        "rationale": "摩托车/自行车涉及时伤亡率更高；但间接效应，通过事故类型中介传导",
    })

    # ── 骨架 7: 事故类型 -> 伤亡结果  [强] ────────────────────────────────────
    _add_group_edges(W, ACCIDENT_TYPE_NODES, INJURY_NODES, STRONG)
    edge_list.append({
        "backbone": 7,
        "description": "事故类型 -> 伤亡结果",
        "sources": [ALL_COLS[i] for i in ACCIDENT_TYPE_NODES],
        "targets": [ALL_COLS[i] for i in INJURY_NODES],
        "strength": STRONG,
        "tier": "强保留",
        "rationale": "涉行人/涉骑行事故直接决定行人/骑行者伤亡指标",
    })

    # ── 确保对角线为 0 ────────────────────────────────────────────────────────
    np.fill_diagonal(W, 0.0)

    return W, edge_list


# ──────────────────────────────────────────────────────────────────────────────
# 统计和验证
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(W: np.ndarray, edge_list: List[dict]) -> None:
    total_edges = int((W > 0).sum())
    strong_edges = int((W >= STRONG - 1e-6).sum())
    weak_edges   = int(((W > 0) & (W < STRONG - 1e-6)).sum())
    print("\n" + "=" * 60)
    print("  宏观因果骨架统计")
    print("=" * 60)
    print(f"  矩阵尺寸   : {W.shape[0]} × {W.shape[1]}")
    print(f"  总边数     : {total_edges}")
    print(f"  强保留边   : {strong_edges} (权重={STRONG})")
    print(f"  弱约束边   : {weak_edges}  (权重={WEAK})")
    print(f"  最大可能边 : {N*(N-1)}")
    print(f"  稀疏度     : {total_edges / (N*(N-1)):.3%}")
    print()
    for e in edge_list:
        n_src = len(e["sources"])
        n_tgt = len(e["targets"])
        print(f"  骨架{e['backbone']}: {e['description']}")
        print(f"    [{e['tier']}] {n_src} 源 × {n_tgt} 目标 = {n_src*n_tgt} 条边（权重 {e['strength']}）")
    print("=" * 60)

    # 快速检查：是否有骨架以外的孤立节点（对 Stage3 不应出现孤立）
    stage3_idx = set(ACCIDENT_TYPE_NODES + VEHICLE_TYPE_NODES + INJURY_NODES + SCALE_NODES)
    isolated = [ALL_COLS[i] for i in stage3_idx
                if W[:, i].sum() == 0 and W[i, :].sum() == 0]
    if isolated:
        print(f"\n  ⚠ Stage3 孤立节点: {isolated}")
    else:
        print("  ✓ 无 Stage3 孤立节点")


# ──────────────────────────────────────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────────────────────────────────────

def save_skeleton(
    W: np.ndarray,
    edge_list: List[dict],
    out_prefix: str,
) -> None:
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. NPY（供 prepare_dataset.py 使用）
    npy_path = out_path.with_suffix(".npy")
    np.save(str(npy_path), W)
    print(f"\n[save] {npy_path}")

    # 2. CSV（可读版本）
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(W, index=ALL_COLS, columns=ALL_COLS).to_csv(str(csv_path))
    print(f"[save] {csv_path}")

    # 3. JSON（边表）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "generated_at":   ts,
        "description":    "宏观因果骨架（7组），适配 2024→2025 迁移学习",
        "matrix_shape":   list(W.shape),
        "feature_names":  ALL_COLS,
        "n_features":     N,
        "n_continuous":   len(CONTINUOUS),
        "n_categorical":  len(CATEGORICAL),
        "total_edges":    int((W > 0).sum()),
        "strong_edges":   int((W >= STRONG - 1e-6).sum()),
        "weak_edges":     int(((W > 0) & (W < STRONG - 1e-6)).sum()),
        "strength_tiers": {"strong": STRONG, "weak": WEAK, "deleted": 0.0},
        "backbones":      edge_list,
        "removed_vs_v1":  (
            "旧版 222 边密集因果图（列级超密连接 + 细粒度行为因子）"
            "→ 本版 7 组宏观骨架（soft 权重，仅保留跨年份稳定主链条）"
        ),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {json_path}")

    # 4. 同时保存一份带时间戳的副本
    ts_npy = out_path.parent / f"{out_path.stem}_{ts}.npy"
    ts_csv = out_path.parent / f"{out_path.stem}_{ts}.csv"
    np.save(str(ts_npy), W)
    pd.DataFrame(W, index=ALL_COLS, columns=ALL_COLS).to_csv(str(ts_csv))
    print(f"[save] {ts_npy}  (timestamped copy)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建宏观因果骨架（7 组主骨架边，三档软约束）"
    )
    parser.add_argument(
        "--out_prefix",
        default=str(CDT_ROOT / "configs" / "causal_matrix_macro_skeleton"),
        help="输出路径前缀（自动添加 .npy/.csv/.json 后缀）",
    )
    parser.add_argument(
        "--verify_col_groups",
        action="store_true",
        help="与 data/processed/column_groups.json 做列名一致性校验",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  宏观因果骨架构建器")
    print(f"  列数: {N} ({len(CONTINUOUS)} 连续 + {len(CATEGORICAL)} 离散)")
    print("=" * 60)

    # 可选：校验与 column_groups.json 的一致性
    if args.verify_col_groups:
        cg_path = CDT_ROOT / "data" / "processed" / "column_groups.json"
        if cg_path.exists():
            with open(cg_path, encoding="utf-8") as f:
                cg = json.load(f)
            cg_all = cg["continuous_cols"] + cg["categorical_cols"]
            diff = set(ALL_COLS) ^ set(cg_all)
            if diff:
                print(f"⚠ 与 column_groups.json 存在差异: {diff}")
            else:
                print("✓ 与 column_groups.json 列名完全一致")

    W, edge_list = build_macro_skeleton()
    print_summary(W, edge_list)
    save_skeleton(W, edge_list, args.out_prefix)
    print("\n✓ 宏观因果骨架构建完毕")
    print(f"  输出前缀: {args.out_prefix}")


if __name__ == "__main__":
    main()
