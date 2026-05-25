# 因果图链路说明

> 当前正式因果矩阵：`causal_matrix_macro_soft.csv / .json`  
> mask 策略：soft mask，λ=0.3  
> 图规模：7 组宏观约束，约 226 条边

---

## 因果图构建流程

```
Step 1: NOTEARS-MLP 自动发现
  → src/causal_discovery_notears.py
  输出：causal_matrix_notears_mlp.npy / .json（原始发现，未修正）

Step 2: 人工领域知识修正
  → pipeline/revise_causal_matrix.py
  操作：删除无语义边，添加时间行为因果边（共修改 ~8 条）
  输出：causal_matrix_v2_constrained.* （旧版 222 边细粒度图）

Step 3: 宏观化 + 软化（当前主线路径）
  → pipeline/build_macro_causal_skeleton.py
  操作：将 47 维变量按宏观语义分为 7 组，构建组间骨架
  输出：causal_matrix_macro_skeleton.* （骨架，硬权重 0/1）

  → pipeline/_zs_make_soft_mask.py
  → pipeline/_zs_expand_soft_masks.py
  操作：将骨架软化（边缘概率 + domain prior），扩展到完整变量维度
  输出：causal_matrix_macro_soft.* ← 当前主线使用

Step 4: 注入训练
  → src/prepare_dataset.py（domain_rules 注入）
  → src/train_hierarchical.py（作为注意力 mask 注入 diffusion loss）
```

---

## 当前各因果矩阵文件说明

| 文件 | 版本 | 状态 | 边数/约束 | 说明 |
|------|------|------|---------|------|
| `causal_matrix_macro_soft.csv` | 宏观软版本 | ✅ **当前主线** | ~226 | 7 组宏观约束 + 软化权重，λ=0.3 |
| `causal_matrix_macro_soft.json` | 宏观软版本 | ✅ **当前主线** | 同上 | JSON 格式，便于可视化 |
| `causal_matrix_macro_skeleton.csv` | 骨架版本 | 参考 | — | 无软化权重，硬骨架 |
| `causal_matrix_v2_constrained.csv` | v2 细粒度 | ⚠️ 旧版 | 222 | 人工修正后的 47×47 细粒度图 |
| `causal_matrix_notears.json` | NOTEARS 原始 | ⚠️ 旧版 | — | 未经人工修正的自动发现结果 |
| `causal_matrix_notears_mlp.json` | NOTEARS-MLP | ⚠️ 旧版 | — | MLP 版 NOTEARS 原始输出 |
| `causal_matrix_v2_constrained_soft.csv` | v2 软化版 | ⚠️ 旧版 | — | 旧版细粒度图的软化尝试，被宏观版替代 |
| `joint_specs.json` | 联合规格 | 辅助 | — | 变量分组规格文件 |

---

## 各因果图脚本职责

| 脚本 | 职责 |
|------|------|
| `causal_discovery_notears.py` | 运行 NOTEARS / NOTEARS-MLP 自动发现 |
| `build_macro_causal_skeleton.py` | 构建 7 组宏观骨架（当前主线关键步骤） |
| `revise_causal_matrix.py` | 人工修正工具（添加/删除边）|
| `_zs_make_soft_mask.py` | 将骨架转换为软化权重矩阵 |
| `_zs_expand_soft_masks.py` | 将组级软 mask 扩展到完整变量维度 |
| `analyze_drift.py` | 分析 2024→2025 变量分布漂移（辅助诊断） |
| `src/prepare_dataset.py` (01_data_pipeline) | Domain rules 注入点，影响 mask 生成 |

---

## 7 组宏观约束语义

当前宏观软图约束的 7 大组：

| 组号 | 组名 | 核心变量 |
|------|------|---------|
| G1 | 时间锚点 | SEASON / IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK |
| G2 | 空间位置 | LATITUDE / LONGITUDE |
| G3 | 天气环境 | TEMPERATURE / PRECIPITATION / VISIBILITY |
| G4 | 道路特征 | SPEED_LIMIT / ROAD_TYPE / NUM_LANES |
| G5 | 车辆类型 | is_sedan / is_suv / is_taxi / … |
| G6 | 事故类型 | is_rear_end / is_pedestrian_related / … |
| G7 | 伤亡结果 | NUMBER_OF_PERSONS_INJURED |

约束方向示例：G1(时间) → G4(道路) → G6(事故类型) → G7(伤亡)

---

## 在训练中使用 soft mask

`train_hierarchical.py` 中的相关入口：
```python
# lambda_causal=0.3 时加载 causal_matrix_macro_soft.npy
# 作为注意力 mask 注入扩散模型的 loss 函数
# 而非 hard masking 注意力权重
```

> ⚠️ **重要**：soft mask 是结构正则化手段，目标是**提升结构真实性和迁移稳健性**，不是为了提升 R²。

---

## 下一步因果图改进方向

1. **domain rules 重写**（`src/prepare_dataset.py`）：当前 domain rules 基于旧版 47 变量 schema，需要适配 `is_other_vehicle` 引入后的新 schema。
2. **2024→2025 漂移感知**：利用 `analyze_drift.py` 分析哪些变量在 2024→2025 间分布漂移最大，进一步弱化不稳定边的权重。
3. **可视化**：用 `causal_matrix_macro_soft.json` 生成 DAG 可视化图，加入论文附录。
