# 因果宏观结构图 v2（约束修订版）

> 生成日期：2026-04-15  
> 对应矩阵：`configs/causal_matrix_v2_constrained.npy`（222 条边）  
> 原始矩阵：`configs/causal_matrix_notears_mlp.npy`（214 条边）  
> 新增 8 条 / 删除 0 条（约束修复后空间→道路边已保留，详见 [tab/causal_macro_edges_v2.md](../tab/causal_macro_edges_v2.md)）

---

## 节点分组说明

| 组别 ID | 名称 | 节点数 | 代表变量 |
|---------|------|--------|---------|
| T | 时间（Time） | 5 | CRASH_TIME_SIN/COS, SEASON, DAY_OF_WEEK, TIME_PERIOD |
| V | 车辆类型（Vehicle） | 10 | is_sedan, is_suv, is_taxi … |
| W | 天气/环境（Weather） | 5 | TEMP_C, prcp, WIND_SPEED_KMH, coco, WEATHER_CONDITION |
| R | 道路属性（Road） | 7 | LATITUDE, LONGITUDE, DIST_TO_SIGNAL_M, INFERRED_LANES … |
| B | 驾驶行为（Behavior） | 12 | is_distracted, is_speeding, is_failure_to_yield … |
| H | 事故结果（Harm/Outcome） | 8 | pedestrian/cyclist/motorist injured/killed bins, TOTAL_VEHICLES, IS_MULTI_VEHICLE |

> **注意**：v2 约束修订中，`LATITUDE`/`LONGITUDE` 被归入"空间（Spatial）"子组，但在 OSM 特征工程语义下它们 *应当* 可以指向道路属性节点（R 组），因为 OSM 属性正是通过空间坐标映射得到的。  
> **已修复（2026-04-15）**：`revise_causal_matrix.py` 约束规则已修正，仅禁止 `time→road`，`spatial→road` 边保留，矩阵重新生成为 222 条边。

---

## Mermaid 宏观因果图

```mermaid
flowchart LR
  subgraph T["⏰ 时间 (Time)"]
    T1["CRASH_TIME_SIN"]
    T2["CRASH_TIME_COS"]
    T3["SEASON"]
    T4["DAY_OF_WEEK"]
    T5["TIME_PERIOD"]
  end

  subgraph W["🌦 天气/环境 (Weather)"]
    W1["TEMP_C"]
    W2["prcp"]
    W3["WIND_SPEED_KMH"]
    W4["coco"]
    W5["WEATHER_CONDITION"]
  end

  subgraph R["🛣 空间/道路 (Spatial+Road)"]
    R1["LATITUDE"]
    R2["LONGITUDE"]
    R3["DIST_TO_SIGNAL_M"]
    R4["INFERRED_LANES"]
    R5["HAS_TRAFFIC_SIGNAL"]
    R6["OSM_ONEWAY"]
    R7["OSM_TYPE"]
  end

  subgraph V["🚗 车辆类型 (Vehicle)"]
    V1["is_sedan / is_suv / is_taxi"]
    V2["is_truck / is_pickup / is_bus"]
    V3["is_van / is_motorcycle"]
    V4["is_bicycle / is_emergency"]
  end

  subgraph B["⚠️ 驾驶行为 (Behavior)"]
    B1["is_distracted"]
    B2["is_speeding"]
    B3["is_failure_to_yield"]
    B4["is_following_too_closely"]
    B5["is_drunk_driving"]
    B6["is_fatigue / is_view_obstructed"]
    B7["is_vehicle_defect / is_backing_unsafely"]
    B8["is_pedestrian_related / is_inexperience"]
    B9["is_pavement_slippery"]
  end

  subgraph H["🚑 事故结果 (Harm/Outcome)"]
    H1["NUMBER_OF_PEDESTRIANS_INJURED_BIN"]
    H2["NUMBER_OF_PEDESTRIANS_KILLED_BIN"]
    H3["NUMBER_OF_CYCLIST_INJURED/KILLED_BIN"]
    H4["NUMBER_OF_MOTORIST_INJURED_BIN"]
    H5["NUMBER_OF_MOTORIST_KILLED_BIN"]
    H6["TOTAL_VEHICLES / IS_MULTI_VEHICLE"]
  end

  %% ── 时间 → 行为（v2 新增，实线绿色）──
  T4 -->|"🆕 weekend→speeding"| B2
  T4 -->|"🆕 weekend→pedestrian"| B8
  T4 -->|"🆕 weekend→distracted"| B1
  T1 -->|"🆕 night→speeding"| B2
  T2 -->|"🆕 night→drunk"| B5
  T3 -->|"🆕 summer→distracted"| B1

  %% ── 时间 → 结果（v2 新增，实线绿色）──
  T4 -->|"🆕 weekend→ped_injured"| H1
  T5 -->|"🆕 rushhour→multi_vehicle"| H6
  T1 -->|"🆕 night→motorist_injured"| H4

  %% ── 空间 → 道路（OSM 映射，虚线橙色=v2 误删，建议恢复）──
  R1 -->|"✅ 已恢复：OSM 映射"| R3
  R1 -.->|"OSM 映射"| R4
  R1 -.->|"OSM 映射"| R5
  R1 -.->|"OSM 映射"| R6
  R1 -.->|"OSM 映射"| R7
  R2 -.->|"OSM 映射"| R3
  R2 -.->|"OSM 映射"| R4

  %% ── 天气 → 行为（原始保留）──
  W -->|"wet/ice→slippery"| B9
  W -->|"weather→behavior"| B

  %% ── 道路 → 行为（原始保留）──
  R -->|"road_context→behavior"| B

  %% ── 行为 → 结果（原始保留）──
  B -->|"behavior→injury"| H

  %% ── 车辆类型 → 行为（原始保留）──
  V -->|"vehicle_type→behavior"| B

  %% ── 天气 → 结果（原始保留）──
  W -->|"weather→outcome"| H

  classDef time fill:#1a3a5c,color:#8ecfff,stroke:#4488cc
  classDef weather fill:#1a3a2a,color:#80e880,stroke:#44aa44
  classDef road fill:#3a2a0a,color:#ffcc80,stroke:#cc8800
  classDef vehicle fill:#2a1a3a,color:#cc88ff,stroke:#8844cc
  classDef behavior fill:#3a1a1a,color:#ff8888,stroke:#cc4444
  classDef harm fill:#3a2a3a,color:#ff80ff,stroke:#aa44aa

  class T1,T2,T3,T4,T5 time
  class W1,W2,W3,W4,W5 weather
  class R1,R2,R3,R4,R5,R6,R7 road
  class V1,V2,V3,V4 vehicle
  class B1,B2,B3,B4,B5,B6,B7,B8,B9 behavior
  class H1,H2,H3,H4,H5,H6 harm
```

---

## 图例说明

| 线型 | 含义 |
|------|------|
| `-->` 实线（绿色标注 🆕） | v2 新增边 |
| `-.->` 虚线（橙色标注 ⚠️） | v2 中被错误禁止、建议恢复的边 |
| `-->` 实线（无标注） | 原始 NOTEARS 矩阵保留边 |

---

*对应详细边表见 → [tab/causal_macro_edges_v2.md](../tab/causal_macro_edges_v2.md)*  
*对应宏观缩略图见 → [graph/causal_macro_thumb_v2.png](causal_macro_thumb_v2.png)*  
*对应详细节点图见 → [graph/causal_graph_v2_detailed.png](causal_graph_v2_detailed.png)*
