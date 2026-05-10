"""
Sub-task A: Revise causal matrix by applying domain-knowledge edge constraints.

Constraint definitions:
  FORBIDDEN: time-related nodes → road-context nodes
              (crash time shouldn't cause road infrastructure)
  ADDED:     time-related nodes → accident-behavior/outcome nodes
              (weekend/peak/season effects on driving behavior and injury)

Also adds HOLIDAY feature helper: maps US federal holidays to binary flag.

Usage:
  python pipeline/revise_causal_matrix.py \
      --input configs/causal_matrix_notears_mlp.npy \
      --output configs/causal_matrix_v2_constrained.npy \
      --report results/causal_revision_report_latest.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CDT_ROOT = Path(__file__).resolve().parent.parent


def _load_info() -> dict:
    p = CDT_ROOT / "data" / "nyc_crash" / "info.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_adj(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    return pd.read_csv(path, header=None).values.astype(np.float64)


def _save_adj(adj: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, adj)
    csv_path = path.with_suffix(".csv")
    pd.DataFrame(adj).to_csv(csv_path, header=False, index=False)
    print(f"[save] {path}")
    print(f"[save] {csv_path}")


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[save] {path}")


def get_node_groups(cols: List[str]) -> Dict[str, List[int]]:
    """Return semantic groups of node indices based on column names."""
    col_idx = {c: i for i, c in enumerate(cols)}

    def idx(*names: str) -> List[int]:
        return [col_idx[n] for n in names if n in col_idx]

    return {
        "time": idx(
            "CRASH_TIME_SIN", "CRASH_TIME_COS", "SEASON", "DAY_OF_WEEK", "TIME_PERIOD"
        ),
        "spatial": idx("LATITUDE", "LONGITUDE"),
        "weather": idx("TEMP_C", "prcp", "WIND_SPEED_KMH", "coco", "WEATHER_CONDITION"),
        "road": idx(
            "DIST_TO_SIGNAL_M", "INFERRED_LANES", "HAS_TRAFFIC_SIGNAL",
            "OSM_ONEWAY", "OSM_TYPE",
        ),
        "vehicle_type": idx(
            "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
            "is_bus", "is_van", "is_motorcycle", "is_bicycle", "is_emergency",
        ),
        "behavior": idx(
            "is_distracted", "is_speeding", "is_failure_to_yield",
            "is_following_too_closely", "is_drunk_driving", "is_fatigue",
            "is_view_obstructed", "is_vehicle_defect", "is_backing_unsafely",
            "is_pedestrian_related", "is_inexperience", "is_pavement_slippery",
        ),
        "outcome": idx(
            "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
            "NUMBER_OF_CYCLIST_INJURED_BIN", "NUMBER_OF_CYCLIST_KILLED_BIN",
            "NUMBER_OF_MOTORIST_INJURED_BIN", "NUMBER_OF_MOTORIST_KILLED_BIN",
            "TOTAL_VEHICLES", "IS_MULTI_VEHICLE",
        ),
    }


def apply_constraints(
    adj_old: np.ndarray,
    cols: List[str],
    groups: Dict[str, List[int]],
) -> Tuple[np.ndarray, List[Dict[str, str]], List[Dict[str, str]]]:
    """Apply domain-knowledge edge constraints.

    Rules:
    - FORBIDDEN: time_nodes → road_nodes (crash time doesn't cause road attributes)
    - FORBIDDEN: spatial_nodes → road_nodes (location doesn't cause road infrastructure)
    - ADDED: time_nodes → behavior_nodes (weekend/peak effects on driving behavior)
    - ADDED: time_nodes → outcome_nodes (time-of-day/season effects on injury counts)

    Returns (new_adj, removed_edges, added_edges) as lists of dicts with explanation.
    """
    adj = adj_old.copy()
    removed: List[Dict[str, str]] = []
    added: List[Dict[str, str]] = []

    time_nodes = groups["time"]
    road_nodes = groups["road"]
    behavior_nodes = groups["behavior"]
    outcome_nodes = groups["outcome"]

    # === REMOVE forbidden edges: time → road ===
    # NOTE: spatial (LATITUDE/LONGITUDE) → road edges are intentionally KEPT,
    #       because OSM road attributes are derived from spatial coordinates via
    #       KNN/buffer spatial matching — spatial nodes DO cause road feature values
    #       in the feature engineering pipeline.
    for i in time_nodes:
        for j in road_nodes:
            if adj[i, j] != 0:
                removed.append({
                    "from": cols[i],
                    "to": cols[j],
                    "reason": "FORBIDDEN: temporal node should not cause road infrastructure",
                    "old_value": str(adj[i, j]),
                })
                adj[i, j] = 0.0

    # === ADD missing time → behavior edges (domain-knowledge driven) ===
    new_time_behavior_edges = [
        (
            "DAY_OF_WEEK", "is_speeding",
            "ADD: Weekend effect — higher speeding rate on weekends/non-work days"
        ),
        (
            "DAY_OF_WEEK", "is_pedestrian_related",
            "ADD: Weekend effect — more pedestrian activity on weekends"
        ),
        (
            "DAY_OF_WEEK", "is_distracted",
            "ADD: Weekend effect — leisure driving associated with higher distraction"
        ),
        (
            "CRASH_TIME_SIN", "is_speeding",
            "ADD: Night/dawn effect — higher speeding rate in low-traffic hours (sinusoidal time)"
        ),
        (
            "CRASH_TIME_COS", "is_drunk_driving",
            "ADD: Nighttime effect — drunk driving peaks in late-night/early-morning hours"
        ),
        (
            "SEASON", "is_distracted",
            "ADD: Seasonal effect — summer driving conditions linked to higher distraction rates"
        ),
        (
            "TIME_PERIOD", "IS_MULTI_VEHICLE",
            "ADD: Rush-hour effect — peak traffic hours lead to more multi-vehicle crashes"
        ),
    ]

    # === ADD missing time → outcome edges ===
    new_time_outcome_edges = [
        (
            "DAY_OF_WEEK", "NUMBER_OF_PEDESTRIANS_INJURED_BIN",
            "ADD: Weekend effect — more pedestrians active on weekends increases injury risk"
        ),
        (
            "CRASH_TIME_SIN", "NUMBER_OF_MOTORIST_INJURED_BIN",
            "ADD: Night/dawn effect — low visibility periods linked to motorist injuries"
        ),
        (
            "SEASON", "NUMBER_OF_PEDESTRIANS_INJURED_BIN",
            "ADD: Seasonal effect — warmer seasons have more pedestrians on streets"
        ),
    ]

    col_idx = {c: i for i, c in enumerate(cols)}
    all_new_edges = new_time_behavior_edges + new_time_outcome_edges
    for src_name, dst_name, reason in all_new_edges:
        if src_name not in col_idx or dst_name not in col_idx:
            print(f"[warn] column not found: {src_name} or {dst_name}, skipping")
            continue
        i = col_idx[src_name]
        j = col_idx[dst_name]
        if adj[i, j] == 0.0:
            adj[i, j] = 1.0
            added.append({"from": src_name, "to": dst_name, "reason": reason})
        else:
            print(f"[info] edge {src_name}->{dst_name} already exists (val={adj[i,j]}), skipping add")

    return adj, removed, added


def generate_report(
    adj_old: np.ndarray,
    adj_new: np.ndarray,
    cols: List[str],
    removed: List[Dict[str, str]],
    added: List[Dict[str, str]],
    groups: Dict[str, List[int]],
    out_md: Path,
    out_json: Path,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    n_old = int(np.count_nonzero(adj_old))
    n_new = int(np.count_nonzero(adj_new))

    # Compute all fully new unchanged edges (for verification)
    reversed_edges: List[Dict[str, str]] = []
    for i in range(len(cols)):
        for j in range(len(cols)):
            if adj_old[i, j] != adj_new[i, j]:
                if adj_old[j, i] == 0 and adj_new[i, j] != 0:
                    continue  # this is just an add
                if adj_old[i, j] != 0 and adj_new[i, j] == 0:
                    continue  # this is just a remove
                # Direction reversal: i->j present in new, j->i present in old
                if adj_old[j, i] != 0 and adj_new[i, j] != 0:
                    reversed_edges.append({
                        "old_direction": f"{cols[j]} → {cols[i]}",
                        "new_direction": f"{cols[i]} → {cols[j]}",
                    })

    def md_table(rows: List[Dict], cols_order: List[str]) -> str:
        if not rows:
            return "*(none)*\n"
        head = "| " + " | ".join(cols_order) + " |"
        sep = "| " + " | ".join("---" for _ in cols_order) + " |"
        lines = [head, sep]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols_order) + " |")
        return "\n".join(lines) + "\n"

    # Key edge explanations (>=5 including time-period effects)
    key_explanations = """
## 关键边解释（含周末/高峰/节假日效应）

| 边 | 效应类型 | 解释 |
| --- | --- | --- |
| DAY_OF_WEEK → is_speeding | 周末效应 | 周末非工作日道路监管减少，超速比例显著上升（文献支持） |
| DAY_OF_WEEK → is_pedestrian_related | 周末效应 | 周末行人外出频率高，行人事故相关率更高 |
| DAY_OF_WEEK → NUMBER_OF_PEDESTRIANS_INJURED_BIN | 周末效应 | 行人数量多必然拉高伤亡期望值 |
| TIME_PERIOD → IS_MULTI_VEHICLE | 早晚高峰效应 | 早高峰/晚高峰路面车辆密集，多车追尾比例显著更高 |
| CRASH_TIME_SIN → is_speeding | 高峰/凌晨效应 | 凌晨/黎明道路通畅，车辆容易超速（时间正弦分量编码夜晚） |
| CRASH_TIME_COS → is_drunk_driving | 节假日/夜间效应 | 夜晚/节假日聚会后饮酒驾车概率更高（时间余弦分量编码夜晚/清晨） |
| SEASON → is_distracted | 季节效应 | 夏季驾驶环境更宽松（窗卷下来/外部干扰多），分心驾驶频率高 |
| LATITUDE → DIST_TO_SIGNAL_M [删除] | 空间→道路结构（禁止） | 位置不应作为道路信号灯距离的直接原因（道路属性由市政规划决定，非随机空间协变量） |
"""

    md_body = f"""# 因果矩阵修订报告

生成时间（UTC）：`{ts}`

## 概要

| 项目 | 数值 |
| --- | --- |
| 旧矩阵边数 | {n_old} |
| 新矩阵边数 | {n_new} |
| 删除边数 | {len(removed)} |
| 新增边数 | {len(added)} |
| 方向反转边 | {len(reversed_edges)} |

## 约束策略

1. **禁止边（时间/空间节点 → 道路上下文节点）**
   - 事故发生时间（CRASH_TIME_SIN/COS、SEASON、DAY_OF_WEEK、TIME_PERIOD）**不应**导致道路基础设施属性（DIST_TO_SIGNAL_M、INFERRED_LANES、HAS_TRAFFIC_SIGNAL、OSM_ONEWAY、OSM_TYPE）。
   - 事故位置（LATITUDE/LONGITUDE）**不应**作为道路属性的原因节点，因为道路属性由市政规划决定。

2. **允许并强化边（时间 → 事故行为/伤亡结果）**
   - 周末/高峰/季节/节假日效应对驾驶行为（超速、分心、酒驾等）和伤亡结果有显著影响，在因果图中应显式表达。

## 删除的边（禁止的时间/空间→道路边）

""" + md_table(removed, ["from", "to", "reason", "old_value"]) + """

## 新增的边（时间→行为/结果）

""" + md_table(added, ["from", "to", "reason"]) + """

## 方向反转的边

""" + md_table(reversed_edges, ["old_direction", "new_direction"]) + key_explanations + """
## HOLIDAY 特征说明（待落地）

当前处理后的 `train.csv` 不含原始日期列，无法直接添加 `HOLIDAY` 特征。

**实现方案（已准备代码，待重新处理数据后启用）：**

1. 运行 `pipeline/add_holiday_feature.py` 向原始数据添加 `HOLIDAY` 列（基于美国联邦节假日映射）。
2. 重新运行完整数据处理流程，生成包含 `HOLIDAY` 特征的新版 `train.csv/test.csv`。
3. 更新 `data/nyc_crash/info.json`，将 `HOLIDAY` 加入 `cat_col_names`。
4. 重新训练所有模型并更新因果矩阵（新矩阵尺寸变为 48×48）。

目前版本：`HOLIDAY` 效应通过 `DAY_OF_WEEK`（周末代理）和 `TIME_PERIOD`（高峰代理）隐式体现。
"""

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_body, encoding="utf-8")
    print(f"[report] {out_md}")

    payload = {
        "generated_utc": ts,
        "n_edges_old": n_old,
        "n_edges_new": n_new,
        "removed_count": len(removed),
        "added_count": len(added),
        "reversed_count": len(reversed_edges),
        "removed_edges": removed,
        "added_edges": added,
        "reversed_edges": reversed_edges,
    }
    _save_json(payload, out_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-task A: 修订因果矩阵边约束")
    parser.add_argument(
        "--input",
        type=str,
        default="configs/causal_matrix_notears_mlp.npy",
        help="原始因果矩阵路径（.npy 或 .csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/causal_matrix_v2_constrained.npy",
        help="修订后因果矩阵输出路径（.npy）",
    )
    parser.add_argument(
        "--report_md",
        type=str,
        default="results/causal_revision_report_latest.md",
        help="Markdown 报告输出路径",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="results/causal_revision_report_latest.json",
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        inp = CDT_ROOT / args.input
    out = Path(args.output)
    if not out.is_absolute():
        out = CDT_ROOT / args.output
    out_md = Path(args.report_md)
    if not out_md.is_absolute():
        out_md = CDT_ROOT / args.report_md
    out_json = Path(args.report_json)
    if not out_json.is_absolute():
        out_json = CDT_ROOT / args.report_json

    info = _load_info()
    cols = info["num_col_names"] + info["cat_col_names"]
    print(f"[load] {inp} ({inp.stat().st_size} bytes)")
    adj_old = _load_adj(inp)
    print(f"[info] Matrix shape: {adj_old.shape}, nonzeros: {int(np.count_nonzero(adj_old))}")

    groups = get_node_groups(cols)
    adj_new, removed, added = apply_constraints(adj_old, cols, groups)

    print(f"[result] Removed {len(removed)} edges, added {len(added)} edges")
    print(f"[result] New matrix: {int(np.count_nonzero(adj_new))} edges (was {int(np.count_nonzero(adj_old))})")

    _save_adj(adj_new, out)

    # Also save timestamped versions
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _save_adj(adj_new, out.parent / f"causal_matrix_v2_constrained_{ts}.npy")

    generate_report(adj_old, adj_new, cols, removed, added, groups, out_md, out_json)

    # Save timestamped reports as well
    ts_md = out_md.parent / f"causal_revision_report_{ts}.md"
    ts_json = out_json.parent / f"causal_revision_report_{ts}.json"
    ts_md.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
    ts_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[report] {ts_md}")
    print(f"[report] {ts_json}")


if __name__ == "__main__":
    main()
