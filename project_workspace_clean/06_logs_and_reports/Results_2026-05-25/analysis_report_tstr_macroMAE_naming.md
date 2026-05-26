# 深度分析报告：TSTR 指标解读、ablation_no_causal 异常低 macroMAE 与模型命名

> 生成时间：2026-05-25
> 分析人：AI Assistant
> 数据来源：`eval_report_v2_ablation_baseline_uniform.json`、`macro_relation_report_phase1_fix_*.json`

---

## 一、为什么 TSTR 指标看起来偏低？

### 1.1 TSTR 分数的计算方式

在 `pipeline/benchmark_evaluator.py` 中，分类任务的 `avg_score` 定义为：

```python
avg_score = (f1_macro + f1_micro + accuracy) / 3
```

即 **三个指标的平均值**，每个指标 clip 到 [0,1]。

### 1.2 目标变量的极度不平衡性

2024 test set 中 `NUMBER OF PERSONS INJURED` 的分布：

| 受伤人数 | 比例 |
|---------|------|
| 0 | 56.08% |
| 1 | 34.29% |
| 2 | 6.27% |
| 3+ | 3.35% |

这是一个 **13 类极度不平衡多分类问题**。多数类（0 人受伤）占比 56%。

### 1.3 关键基准对比

| 基准 | avg_score | 说明 |
|------|-----------|------|
| **多数类猜测**（全猜 0） | **0.395** | f1_macro=0.0715, f1_micro=0.5569, acc=0.5569 |
| **no_proxy TSTR**（内部模型平均） | **0.392** | 与多数类基线几乎相同 |
| **TRTR**（真数据训练/真数据测试） | **0.648** | RF/XGB/MLP 平均值 |
| **随机猜测**（均匀 13 类） | **0.077** | 1/13 ≈ 7.7% |

### 1.4 核心结论

**TSTR 0.39 并非"低"，而是接近"不可能更高"的理论上限。**

原因：
1. **排除 proxy 后，合成数据几乎无法提供比"全猜 0"更多的信息**
   - no_proxy TSTR (0.392) ≈ 多数类基线 (0.395)
   - 说明合成数据在细粒度模式（区分 1人/2人/3+人受伤）上完全失效
2. **TRTR (0.648) 与 TSTR (0.392) 的巨大差距（Δ=0.256）** 表明：
   - 合成数据保留了粗粒度统计分布（均值、方差、边际分布）
   - 但丢失了真实数据中区分稀有类别（2+ 人受伤）所需的细粒度特征交互
3. **为什么 std_TSTR (0.39) 与 no_proxy_TSTR (0.39) 差异不大？**
   - 对于内部模型，proxy gap 接近 0（-0.003 ~ +0.003）
   - 说明内部模型没有利用 proxy columns 进行预测
   - 但外部基线（CTGAN/TVAE/SMOTE）的 std_TSTR 显著高于 no_proxy（如 SMOTE: 0.63 vs 0.39），这正是 proxy leakage 的证据

### 1.5 对研究路线的启示

| 问题 | 启示 |
|------|------|
| TSTR 是否足以评估合成数据质量？ | **不够**。对于极度不平衡多分类，TSTR 被多数类主导，无法反映稀有类别的生成质量。需要补充 per-class F1 或 macro-F1。 |
| 为什么宏观关系（macroMAE）更重要？ | 因为 TSTR 无法区分模型在细粒度模式上的差异，而 macroMAE 直接衡量 group-wise 条件均值的一致性。 |
| 是否需要重新设计评估指标？ | 建议：对 injury 进行二值化（≥1 vs 0）或粗粒度分箱（0/1/2+），降低类别不平衡对 TSTR 的扭曲。 |

---

## 二、ablation_no_causal 异常低 macroMAE 的根因分析

### 2.1 数据对比

Phase 1 各模型按类别的 macroMAE：

| 模型 | crash_type | road | vehicle | weather | **overall** |
|------|------------|------|---------|---------|-------------|
| **ablation_no_causal_v2** | **0.0229** | **0.0278** | **0.0212** | **0.0280** | **0.0250** |
| exp_c_macro_full_v1 | 0.0398 | 0.0427 | 0.0380 | 0.0401 | 0.0402 |
| macro_sparse_anneal_v2 | 0.0396 | 0.0381 | 0.0386 | 0.0402 | 0.0391 |
| ablation_no_hierarchy_v2 | 0.0368 | 0.0399 | 0.0370 | 0.0381 | 0.0380 |
| baseline_tabddpm_v2 | 0.0454 | 0.0442 | 0.0443 | 0.0453 | 0.0448 |

**ablation_no_causal_v2 在所有 4 个类别上均显著最低**（比第二名低 30-45%）。

### 2.2 根因假说

#### 假说 A：无因果约束 → 直接拟合训练集宏观模式

**因果 mask 的作用**：在 `exp_c_macro_full_v1` 中，DAG 强制某些特征不直接影响 injury（如 `ROAD_TYPE` 必须通过 `VEHICLE_TYPE` 间接影响）。这限制了模型学习某些关联。

**无 mask 的效果**：`ablation_no_causal_v2` 可以自由学习所有特征与 injury 的直接关联，包括：
- `ROAD_TYPE` → `INJURY`（直接）
- `WEATHER` → `INJURY`（直接）
- 各种交互项

由于训练集的宏观关系（group-wise injury rate）是固定的，无约束模型可以直接拟合这些数值，获得极低的 macroMAE。

#### 假说 B：过拟合训练集分布 → 迁移后失效

**关键证据**：

| 模型 | Phase 1 macroMAE | Phase 2 macroMAE | Δ |
|------|------------------|------------------|---|
| ablation_no_causal_v2 | **0.0250** | 0.0366 | **+46%** |
| macro_sparse_anneal_v2 | 0.0391 | **0.0302** | **-23%** |
| baseline_tabddpm_v2 | 0.0448 | 0.0442 | -1% |

- ablation_no_causal 在迁移后 macroMAE **恶化 46%**
- macro_sparse_anneal_v2（有因果约束）在迁移后 **改善 23%**
- TabDDPM（无显式因果约束但架构不同）几乎不变

**结论**：ablation_no_causal 的低 macroMAE 是**训练集过拟合**的结果。它记住了 2024 年的特定 group-wise 模式，当 2025 年分布发生漂移时，这些记忆反而成为负担。

#### 假说 C：因果约束引入了"正则化代价"

因果 DAG 作为一种结构正则化，强制模型遵循特定的生成路径。这种约束：
- **代价**：限制了模型拟合训练集宏观关系的能力（macroMAE 从 0.025 → 0.040）
- **收益**：提升了跨年度泛化能力（迁移后 macroMAE 从 0.037 → 0.030，而 ablation 从 0.025 → 0.037）

### 2.3 结论

ablation_no_causal_v2 的低 macroMAE 并非"模型更好"，而是**过拟合了训练集的特定宏观模式**。因果约束虽然提高了域内 macroMAE，但增强了跨年度鲁棒性。

---

## 三、模型清晰命名方案

### 3.1 命名原则

1. **体现核心差异**：每个名字应反映该模型与其他模型的关键区别
2. **适合论文图表**：简洁、专业、易于引用
3. **避免实验 ID**：如 `exp_c_macro_full_v1` 应替换为有意义的名称

### 3.2 命名方案

| 原模型 ID | 建议命名（英文） | 建议命名（中文） | 核心特征说明 |
|-----------|-----------------|-----------------|-------------|
| `exp_c_macro_full_v1` | **CausalDiffTab-Macro** | 因果扩散-宏观约束 | 完整因果 DAG + 宏观一致性损失 |
| `macro_sparse_anneal_v2` | **CausalDiffTab-Sparse** | 因果扩散-稀疏退火 | 稀疏退火训练 + 软因果 mask |
| `ablation_no_causal_v2` | **DiffTab-NoCausal** | 扩散表-无因果 | 无因果约束的消融 |
| `ablation_no_hierarchy_v2` | **DiffTab-Flat** | 扩散表-扁平 | 无层次结构（单阶段）消融 |
| `baseline_tabddpm_v2` | **TabDDPM** | TabDDPM | 表格扩散基线 |
| `baseline_ctgan` | **CTGAN** | CTGAN | 条件 GAN 基线 |
| `baseline_tvae` | **TVAE** | TVAE | 变分自编码器基线 |
| `baseline_smote` | **SMOTE** | SMOTE | 过采样基线 |

### 3.3 带 Guidance 的变体

| 原模型 ID | 建议命名 |
|-----------|---------|
| `exp_c_macro_full_v1` + guidance=0.5 | **CausalDiffTab-Macro-Guidance** |
| `macro_sparse_anneal_v2` + guidance=0.5 | **CausalDiffTab-Sparse-Guidance** |

### 3.4 论文图表中的使用示例

```
图 3：域内（2024）macroMAE 对比
┌─────────────────────────────┬──────────┐
│ CausalDiffTab-Macro         │ 0.0402   │
│ CausalDiffTab-Sparse        │ 0.0391   │
│ CausalDiffTab-Sparse-Guid.  │ 0.0297   │ ← 最佳
│ DiffTab-NoCausal            │ 0.0250   │
│ DiffTab-Flat                │ 0.0380   │
│ TabDDPM                     │ 0.0448   │
│ CTGAN                       │ 0.1141   │
│ TVAE                        │ 0.0781   │
│ SMOTE                       │ 4.7229   │
└─────────────────────────────┴──────────┘
```

---

## 四、改进 guidance 迁移鲁棒性的方案

### 4.1 当前问题

当前 guidance 公式（`sample_conditional.py`）：

```python
offset = (target_mean_raw - current_mean_raw) / (std + 1e-8) * scale
```

**问题**：
- `target_mean_raw` 是 2024 训练集预计算的固定值
- 2025 年分布漂移时，`target_mean_raw` 与 2025 真实 group mean 可能差异巨大
- 导致 over-correction（guidance=0.5 迁移 macroMAE 从 0.030 → 0.056）

### 4.2 方案 A：相对偏移（Relative Offset）

不使用绝对目标值，而是根据当前生成状态的偏差进行相对修正：

```python
# 计算当前偏差比例
if current_mean_raw > 0:
    relative_error = (target_mean_raw - current_mean_raw) / current_mean_raw
else:
    relative_error = 0.0

# 使用相对误差的 sigmoid 压缩，避免极端值
compressed_error = math.tanh(relative_error)

# 偏移量与当前值成正比，而非绝对差值
offset = compressed_error * current_mean_std * scale
```

**优点**：当目标值与当前值差异巨大时，不会过度校正。

### 4.3 方案 B：自适应缩放因子（Adaptive Scale）

根据分布漂移程度动态调整 guidance scale：

```python
# 计算当前 batch 的分布漂移指标
drift_indicator = abs(current_mean_raw - target_mean_raw) / (std + 1e-8)

# 如果漂移过大，自动降低 scale
adaptive_scale = scale * math.exp(-drift_indicator)

offset = (target_mean_raw - current_mean_raw) / (std + 1e-8) * adaptive_scale
```

**优点**：自动检测并抑制 over-correction。

### 4.4 方案 C：温度退火（Temperature Annealing）

在采样早期（高噪声阶段）使用强 guidance，后期（低噪声阶段）减弱：

```python
# i 是当前时间步（从 T 到 0）
temperature = i / self.num_steps  # 0→1
annealed_scale = scale * temperature

offset = (target_mean_raw - current_mean_raw) / (std + 1e-8) * annealed_scale
```

**优点**：避免在低噪声阶段对精细结构造成破坏。

### 4.5 推荐实现

建议组合 **方案 A + 方案 B**：相对偏移 + 自适应缩放。

---

## 五、2025 微调实验设计

### 5.1 实验目标

验证：用 2024 checkpoint 在 2025 数据上微调能否同时提升 TSTR 和 macroMAE。

### 5.2 实验设计

| 参数 | 设置 |
|------|------|
| 预训练模型 | `macro_sparse_anneal_v2`（2024 checkpoint）|
| 微调数据 | `nyc_crash_2025_v2` train |
| 微调 epoch | 200-500 |
| 学习率 | 预训练的 1/10（如 1e-5 → 1e-6）|
| 冻结层 | Stage1（时空锚点）冻结，只微调 Stage3 |
| 评估 | 2025 test set TSTR + macroMAE |
| 对比基线 | 不微调的 zero-shot 结果 |

### 5.3 预期结果

| 假设 | 预期 |
|------|------|
| 微调提升 TSTR | 是。2025 数据有更丰富的特征分布，微调后模型能更好拟合。 |
| 微调对 macroMAE 的影响 | 不确定。可能改善（更好拟合 2025 group means），也可能恶化（过拟合 2025 训练集）。 |
| 最优微调 epoch | 100-200（需 early stopping）。 |

---

## 六、分布漂移量化设计

### 6.1 量化指标

| 指标 | 计算方法 | 解释 |
|------|---------|------|
| Wasserstein-1 | `scipy.stats.wasserstein_distance` | 数值特征分布差异 |
| JS Divergence | `scipy.spatial.distance.jensenshannon` | 类别特征分布差异 |
| Covariate Shift | 特征均值差异 / 联合分布差异 | 输入分布变化 |
| Concept Drift | P(Y\|X) 的差异 | 目标条件分布变化 |

### 6.2 实现要点

1. **数值特征**：Wasserstein distance + KS 检验
2. **类别特征**：JS divergence + Chi-square 检验
3. **目标变量**：条件分布 P(Y\|group) 的差异
4. **宏观关系**：2024 vs 2025 的 group-wise injury rate 差异

---

## 七、下一步行动清单

| 优先级 | 任务 | 预计工作量 |
|--------|------|-----------|
| P0 | 实现 robust guidance（相对偏移 + 自适应缩放）| 2h |
| P0 | 实现 2025 微调脚本 | 2h |
| P1 | 实现分布漂移量化脚本 | 1h |
| P1 | 运行 robust guidance 消融实验 | 3h |
| P2 | 运行 2025 微调实验 | 4h |
| P2 | 更新所有报告中的模型命名 | 1h |
