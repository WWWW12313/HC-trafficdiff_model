# 当前正式研究主线声明

> 文件用途：作为整个项目的唯一权威主线参照，所有后续代码修改、配置选择、结果对比均以此为准。  
> 最后更新：2026-05-16

---

## 一、主线标识

| 维度 | 当前正式选择 | 已弃用 / 保留参考 |
|------|-------------|-----------------|
| 迁移口径 | **2024 → 2025** | 旧版 2017 域内 / postcovid 2017→2025 |
| 空间表示 | **LATITUDE / LONGITUDE（经纬度）** | H3 r8 road-cell（保留代码，暂不入主线） |
| 时间表示 | **SEASON / IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK + CRASH_TIME_SIN/COS** | 旧版细粒度 TIME_PERIOD / CRASH_HOUR |
| 事故类型 | **多列 0/1 指示列**（is_rear_end / is_pedestrian_related 等） | 旧版 CONTRIBUTING_FACTOR_VEHICLE_1 原始字符串 |
| 车辆类型 | **8列 0/1**（is_sedan/is_suv/is_taxi/is_truck/is_bus/is_motorcycle/is_bicycle/is_other_vehicle） | 旧版 VEHICLE_TYPE_CODE_1 字符串 / is_pickup / is_van 独立列 |
| 因果图 | **宏观软因果骨架（macro soft DAG）**，7 组约束，约 226 条边 | 旧版 222 边细粒度 v2 因果图（保留 CSV 供对比） |
| mask 策略 | **soft mask（causal_matrix_macro_soft.npy）**，λ=0.3 | 旧版 hard mask / v2_constrained / notears 原始 |
| 主线实验 ID | **macro_soft_2024** | ours_stage2_causal / our_model_no_h3（已作为消融基准） |
| 层级生成 | **Stage1（时空锚点）→ Stage3（事故微观变量）** | Stage2（天气/OSM 上下文）暂作辅助 |
| 评价重心 | **跨年份迁移稳健性 + 结构真实性** | R² / 单指标 TSTR 不再作为主结论依据 |

---

## 二、字段规范（当前主线）

### 2.1 时间字段

```python
SEASON          # 0-3, 春/夏/秋/冬
IS_WEEKEND      # 0/1
IS_AM_PEAK      # 0/1，07:00-09:00
IS_PM_PEAK      # 0/1，17:00-19:00
CRASH_TIME_SIN  # 连续时间周期编码（sin）
CRASH_TIME_COS  # 连续时间周期编码（cos）
```

旧版 `TIME_PERIOD`（细粒度时段）和 `CRASH_HOUR`（整数小时）**不再作为主训练特征**，可保留但不进入主线评测。

### 2.2 空间字段

```python
LATITUDE    # float，保留原始精度
LONGITUDE   # float，保留原始精度
```

**不使用 `ROAD_H3_CELL`**（H3 r8 编码）。H3 相关代码保留在 `scripts/apply_h3_roadcell_projection.py` 供消融。

### 2.3 事故类型字段（0/1 指示列）

当前实际使用（以 `build_2025_like_2017.py` 和 `data_processor.py` 为准）：

```python
is_rear_end
is_turning_conflict
is_pedestrian_related
is_cyclist_related
is_single_vehicle
is_multi_vehicle
IS_MULTI_VEHICLE   # 与 is_multi_vehicle 可能冗余，以 data_processor 实际输出列名为准
```

### 2.4 车辆类型字段（0/1 指示列）

```python
is_sedan
is_suv
is_taxi
is_truck
is_bus
is_motorcycle
is_bicycle
is_other_vehicle   # 2026-04-29 后新增，取代旧版 is_pickup / is_van 独立列
```

> ⚠️ **注意**：2026-04-29 引入 `is_other_vehicle` 后，schema 发生变化，旧 checkpoint 不可直接复用，需重新生成数据集并重训。

### 2.5 伤亡目标

```python
NUMBER_OF_PERSONS_INJURED  # 有界整数 0-5，兼作回归目标和分类标签
```

---

## 三、当前正式因果图版本

| 矩阵文件 | 状态 | 说明 |
|---------|------|------|
| `configs/causal_matrix_macro_soft.csv` | ✅ **当前主线** | 宏观软骨架，7组约束，~226边 |
| `configs/causal_matrix_macro_soft.json` | ✅ **当前主线** | 同上，JSON 格式 |
| `configs/causal_matrix_macro_skeleton.csv` | 参考 | 无软化权重的骨架版本 |
| `configs/causal_matrix_v2_constrained.csv` | ⚠️ 旧版 | 222边细粒度，仍存档供对比 |
| `configs/causal_matrix_notears.json` | ⚠️ 旧版 | NOTEARS 原始输出，未经人工修正 |

---

## 四、当前正式实验配置

```yaml
# 05_experiment_configs/experiments/macro_soft_2024.yaml
model_name: macro_soft_2024
experiment_id: macro_soft_2024
lambda_causal: 0.3
use_causal_masks: true
hierarchical: true
description: 宏观软因果骨架 + 分层生成，lambda=0.3，2024训练/2025迁移
```

> 对比消融：`our_model_no_h3.yaml`（无 H3 基准）、`ablation_no_causal.yaml`（去除因果约束）、`ablation_no_hierarchy.yaml`（去除分层）

---

## 五、旧路线档案说明

以下内容**已归档，不再作为主线推进**，保留仅供参考和消融对比：

| 旧路线 | 对应文件/实验 ID | 弃用原因 |
|--------|---------------|---------|
| 2017 域内全年 | `our_model.yaml` / `ours_full_model.yaml` | 迁移场景转移到 2024→2025 |
| v2 细粒度因果图（222边） | `causal_matrix_v2_constrained.*` | 细粒度跨年份不稳定，改用宏观软图 |
| H3 road-cell 主线 | `apply_h3_roadcell_projection.py` | 结果不优于经纬度，不进主线 |
| 训练期语义 CE 多头 | `--semantic_heads` 开关 | 持续损害 TSTR（见 2026-04-29 日志） |
| hard mask（v2_constrained_soft） | `causal_matrix_v2_constrained_soft.*` | 被 macro soft mask 替代 |
| postcovid 2017→2025 | `evaluate_postcovid_transfer.py` | 迁移口径已改为 2024→2025 |
