# 因果宏观边表 v2（约束修订版）

> 生成日期：2026-04-15  
> 对应矩阵：`configs/causal_matrix_v2_constrained.npy`（222 条边）  
> 对应图见：[graph/causal_macro_graph_v2.md](../graph/causal_macro_graph_v2.md)

---

## 表 1：宏观分组 → 具体变量

| 组别 | 简称 | 具体变量 |
|------|------|---------|
| 时间（Time） | T | CRASH_TIME_SIN, CRASH_TIME_COS, SEASON, DAY_OF_WEEK, TIME_PERIOD |
| 天气/环境（Weather） | W | TEMP_C, prcp, WIND_SPEED_KMH, coco, WEATHER_CONDITION |
| 空间（Spatial） | R_sp | LATITUDE, LONGITUDE |
| 道路属性（Road） | R_rd | DIST_TO_SIGNAL_M, INFERRED_LANES, HAS_TRAFFIC_SIGNAL, OSM_ONEWAY, OSM_TYPE |
| 车辆类型（Vehicle） | V | is_sedan, is_suv, is_taxi, is_truck, is_pickup, is_bus, is_van, is_motorcycle, is_bicycle, is_emergency |
| 驾驶行为（Behavior） | B | is_distracted, is_speeding, is_failure_to_yield, is_following_too_closely, is_drunk_driving, is_fatigue, is_view_obstructed, is_vehicle_defect, is_backing_unsafely, is_pedestrian_related, is_inexperience, is_pavement_slippery |
| 事故结果（Harm/Outcome） | H | NUMBER_OF_PEDESTRIANS_INJURED_BIN, NUMBER_OF_PEDESTRIANS_KILLED_BIN, NUMBER_OF_CYCLIST_INJURED_BIN, NUMBER_OF_CYCLIST_KILLED_BIN, NUMBER_OF_MOTORIST_INJURED_BIN, NUMBER_OF_MOTORIST_KILLED_BIN, TOTAL_VEHICLES, IS_MULTI_VEHICLE |

---

## 表 2：v2 新增边（8 条）

| # | 源节点（from） | 目标节点（to） | 宏观方向 | 语义解释 |
|---|---------------|---------------|---------|---------|
| 1 | DAY_OF_WEEK | is_speeding | T → B | 周末效应——非工作日道路监管相对宽松，超速比例显著上升（文献支持） |
| 2 | DAY_OF_WEEK | is_pedestrian_related | T → B | 周末效应——周末行人外出频率高，行人事故相关率更高 |
| 3 | DAY_OF_WEEK | is_distracted | T → B | 周末效应——休闲驾驶与更高分心驾驶率相关 |
| 4 | CRASH_TIME_SIN | is_speeding | T → B | 夜间/黎明效应——低流量时段车辆容易超速（正弦时间编码夜晚） |
| 5 | SEASON | is_distracted | T → B | 季节效应——夏季驾驶环境干扰更多，分心驾驶频率高 |
| 6 | TIME_PERIOD | IS_MULTI_VEHICLE | T → H | 早晚高峰效应——高峰时段路面车辆密集，多车碰撞比例显著更高 |
| 7 | DAY_OF_WEEK | NUMBER_OF_PEDESTRIANS_INJURED_BIN | T → H | 周末效应——行人数量多拉高行人伤亡期望值 |
| 8 | CRASH_TIME_SIN | NUMBER_OF_MOTORIST_INJURED_BIN | T → H | 夜间/黎明效应——低能见度时段与驾驶员伤亡相关 |

---

## 表 3：v2 约束修订记录（已修复，2026-04-15）

| # | 源节点（from） | 目标节点（to） | 宏观方向 | 原始误删理由 | 修复后状态 |
|---|---------------|---------------|---------|------------|----------|
| 1 | LATITUDE | DIST_TO_SIGNAL_M | R_sp → R_rd | 旧逻辑："位置不应决定道路基础设施属性" | **✅ 已恢复**：OSM 属性通过空间坐标 KNN 映射得到，该边属于特征工程生成机制 |

> **根因（已修复）**：原 `revise_causal_matrix.py` 使用 `time_and_spatial = groups["time"] + groups["spatial"]`，合并禁止时间→道路与空间→道路，误删了 OSM 特征工程语义下合理的边。  
> **修复内容**：禁止规则改为仅作用于 `groups["time"]`，不再包含 `groups["spatial"]`。矩阵已重新生成，当前边数 **222 条**（新增 8 条，删除 0 条）。

---

## 表 4：宏观组对关系汇总

| 源组 | 目标组 | 关系类型 | 典型边数（v2） | 说明 |
|------|--------|---------|--------------|------|
| T（时间） | B（行为） | **因果（新增/增强）** | 6 条 | 周末/夜间/季节效应直接影响驾驶行为 |
| T（时间） | H（结果） | **因果（新增）** | 3 条 | 时段效应通过直接或间接路径影响伤亡结果 |
| W（天气） | B（行为） | **因果（保留）** | 多条 | 天气影响路面状况和驾驶行为 |
| R_sp（空间） | R_rd（道路） | **映射关系（待恢复）** | 7 条 | 坐标决定最近 OSM 特征（KNN 空间映射） |
| V（车辆） | B（行为） | **关联（保留）** | 多条 | 车辆类型与事故类型/驾驶行为模式相关 |
| B（行为） | H（结果） | **因果（核心路径，保留）** | 多条 | 驾驶行为直接导致伤亡结果，为模型最核心因果链 |
| R_rd（道路） | B（行为） | **上下文（保留）** | 多条 | 道路基础设施影响驾驶行为选择 |
| W（天气） | H（结果） | **因果（保留）** | 多条 | 恶劣天气直接提高事故伤亡风险 |

---

## 表 5：论文/报告推荐表述

> 针对空间→道路边的学术描述建议（供写作参考）

**建议表述（中文）：**
> "空间坐标（经纬度）在特征工程层面通过空间近邻匹配（KNN/OSM buffer query）决定了该事故点的道路基础设施属性（如交通信号灯距离、车道数、路口类型），因此在因果图中存在空间节点→道路属性节点的有向边，这一边反映的是**特征工程的生成机制**，而非"位置本身导致道路结构"的物理因果。"

**建议表述（英文）：**
> "In the spatial feature engineering process, raw GPS coordinates are used to derive road infrastructure attributes (signal distance, lane count, OSM road type) via spatial nearest-neighbor matching. Therefore, the directed edges from spatial nodes (LATITUDE, LONGITUDE) to road attribute nodes reflect the *computational provenance* of these features rather than a physical causal claim that geographic location causes road infrastructure."

---

## 附：可视化文件索引

| 文件 | 类型 | 说明 |
|------|------|------|
| [graph/causal_macro_graph_v2.md](../graph/causal_macro_graph_v2.md) | Mermaid 宏观图 | 5 组宏观结构 + v2 修改标注 |
| [graph/causal_graph_v2_detailed.png](../graph/causal_graph_v2_detailed.png) | 渲染 PNG | 47 节点完整因果图，6 色分组，绿色高亮新增边 |

---

## 导出规范（Convention）

> **本文件夹的使用约定：**
> - `graph/`：仅存放图形文件（Mermaid `.md` 或渲染 `.png`），每个文件一张图
> - `tab/`：存放与图形对应的边表、变量分组表、解释表，与 `graph/` 文件一一对应
> - 命名规则：`{topic}_{version}_{YYYYMMDD}.{ext}`
> - 每个 `graph/` 文件应在文末链接其对应的 `tab/` 文件，反之亦然
