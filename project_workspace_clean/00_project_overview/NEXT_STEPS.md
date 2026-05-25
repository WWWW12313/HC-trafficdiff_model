# 下一步实验改进建议

> 基于当前正式主线（2024→2025 · 宏观软 DAG · soft mask · v2 schema）  
> 生成日期：2026-05-16（更新于 2026-05-23）  
> **本文档只提建议，不修改代码；代码修改须另行确认**

---

## 已完成事项（2026-05-16）

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | **Schema 对齐验证**：`is_other_vehicle` 引入后全链路一致 | ✅ 已完成 |
| P1 | **domain rules 重写**：`prepare_dataset.py` 适配新 schema | ✅ 已完成 |
| P2 | **因果 mask 重建**：基于新 schema 生成 macro soft mask | ✅ 已完成 |
| P3 | **主线重训**：`macro_soft_2024_v2` Stage1 + Stage3 训练 | ✅ 已完成 |

---

## 当前优先级排序

| 优先级 | 任务 | 影响范围 | 估计工作量 |
|--------|------|---------|----------|
| **P0** | **补全 v2 消融与基线 2025 迁移评估**：`macro_soft_2024_v2` / `ablation_no_causal_v2` / `ablation_no_hierarchy_v2` / `baseline_tabddpm_v2` / CTGAN / TVAE / SMOTE 的统一迁移评测 | 评测 | **0.5天** |
| P1 | **2024→2025 漂移分析**：用 `analyze_drift.py` 找出高漂移变量，调整 λ | 评测+因果 | 0.5天 |
| P2 | **评测口径统一**：确保 2024/2025 评测用完全一致的脚本和指标 | 评测 | 0.5天 |

---

## P0：Schema 对齐验证

### 问题
2026-04-29 引入 `is_other_vehicle` 后，以下文件需同步更新：
- `src/data_processor.py` ✅（已修改）
- `pipeline/build_2025_like_2017.py` ✅（已修改）
- `src/postprocess_samples.py` ✅（已修改）
- `src/prepare_dataset.py` ❓ **需确认 domain_rules 中的 vehicle 列是否已适配**
- `configs/` 中的 mask 矩阵维度 ❓ **需确认 causal_matrix_macro_soft 的变量列表是否包含 is_other_vehicle**

### 验证方法

```bash
# 检查 prepare_dataset.py 中 vehicle 相关列定义
grep -n "is_pickup\|is_van\|is_other_vehicle" src/prepare_dataset.py

# 检查 joint_specs.json 中车辆分组定义
python -c "import json; d=json.load(open('configs/joint_specs.json')); print([k for k in d if 'vehicle' in k.lower()])"

# 检查生成数据的列数是否一致
python -c "import numpy as np; print(np.load('data/nyc_crash/X_cat_train.npy').shape)"
```

---

## P1：domain_rules 重写（`src/prepare_dataset.py`）

### 当前问题
当前 domain rules 是基于旧版 47 变量 schema（含 `is_pickup`、`is_van`）写的，`is_other_vehicle` 引入后存在：
- domain rules 中可能仍引用不存在的列
- 车辆组合约束逻辑可能不正确（如 `TOTAL_VEHICLES` 与 8 列 vehicle 的一致性约束）

### 建议修改方向

```python
# 当前可能的旧 domain rule
vehicle_cols = ['is_sedan','is_suv','is_taxi','is_truck','is_bus',
                'is_motorcycle','is_bicycle','is_pickup','is_van']

# 改为新 schema
vehicle_cols = ['is_sedan','is_suv','is_taxi','is_truck','is_bus',
                'is_motorcycle','is_bicycle','is_other_vehicle']

# 对应的因果约束：至少一辆车参与（vehicle_sum >= 1）
# 这里只调整列名，不改变约束逻辑
```

### 修改文件
- 主要文件：`src/prepare_dataset.py`（domain_rules 函数）
- 同步更新：`configs/joint_specs.json`（车辆分组定义）

---

## P2：因果 Mask 重建

### 当前问题
`causal_matrix_macro_soft.npy` 的维度基于旧 schema，若 schema 变更导致变量数量变化，mask 维度会不匹配。

### 建议步骤

```bash
# Step1: 重新构建宏观骨架（基于新 schema 的 joint_specs）
python pipeline/build_macro_causal_skeleton.py --schema_version 2024v2

# Step2: 软化骨架
python pipeline/_zs_make_soft_mask.py

# Step3: 展开到完整变量维度
python pipeline/_zs_expand_soft_masks.py

# Step4: 验证维度
python -c "import numpy as np; m=np.load('configs/causal_matrix_macro_soft.npy'); print(m.shape)"
```

---

## P3：主线重训（等 P0-P2 完成后）

### 训练顺序

```bash
# 1. 重新生成数据集
python pipeline/prepare_2025_data.py

# 2. 训练 Stage1
python src/train_hierarchical.py --stage 1 --tier full --device cuda:0 \
  --no_wandb --lambda_causal 0.3 --experiment_id macro_soft_2024_v2

# 3. 训练 Stage3
python src/train_hierarchical.py --stage 3 --tier full --device cuda:0 \
  --no_wandb --lambda_causal 0.3 --experiment_id macro_soft_2024_v2

# 4. 采样 + 评测
python pipeline/evaluate_all.py --experiment_id macro_soft_2024_v2 \
  --n_samples 10000 --eval_transfer
```

> 使用新 `experiment_id`（加 `_v2` 后缀）以区分 schema 变更前后的结果。

---

## P4：2024→2025 漂移分析

### 目标
找出在 2024→2025 之间分布漂移最大的变量，将其从因果约束中弱化或移出。

```bash
python pipeline/analyze_drift.py \
  --train_csv data/nyc_crash/train.csv \
  --test_csv data/nyc_crash_2025/test.csv \
  --output results/drift_analysis_2024_2025.csv
```

### 使用结果
- 漂移大的变量（如某些特定道路类型）：降低其在 soft mask 中的权重
- 漂移小的变量（如季节、经纬度）：保持或加强约束
- 将漂移分析结果记录到当天实验日志

---

## P0：补全 v2 消融与基线 2025 迁移评估

### 问题

v2 schema 下，`macro_soft_2024_v2` 及其消融、基线模型已有 **2024 域内**合成数据与评估，但 **2025 迁移评估完全缺失**。

已存在的 v2 合成数据（`results/synthetic/`）：
- 内部模型：`macro_soft_2024_v2_full.csv`、`ablation_no_causal_v2_full.csv`、`ablation_no_hierarchy_v2_full.csv`、`baseline_tabddpm_v2_full.csv`
- 外部基线：`baseline_ctgan_full.csv`、`baseline_tvae_full.csv`、`baseline_smote_full.csv`

迁移测试集已就绪：`data/nyc_crash_2025_v2/test.csv`（16,540 行）

### 执行步骤

#### Step 1：域内评估（如尚未完成）

```bash
python pipeline/evaluate_all.py \
  --real_test data/nyc_crash_2024_v2/test.csv \
  --file_glob "*_v2_full.csv" \
  --primary_metrics_profile no_rule \
  --output_tag v2_in_domain_2024
```

#### Step 2：批量迁移评估（一键完成所有模型）

```bash
python pipeline/run_transfer_eval.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --output_tag v2_transfer_2025 \
  --primary_metrics_profile no_rule
```

> `run_transfer_eval.py` 会自动收集 `*_v2_full.csv` 和 `baseline_*_full.csv`，统一在 2025 测试集上评估。

#### Step 3：合并域内与迁移报告

```bash
python pipeline/merge_transfer_reports.py \
  --in_domain_json results/eval_report_v2_in_domain_2024.json \
  --transfer_json results/eval_report_v2_transfer_2025.json \
  --output results/transfer_comparison_v2.md
```

#### Step 4（可选）：端到端基准套件

```bash
python pipeline/run_benchmark_suite.py \
  --tier full \
  --transfer_test data/nyc_crash_2025_v2/test.csv \
  --skip_internal_train
```

### 预期产出

| 文件 | 说明 |
|------|------|
| `results/eval_report_v2_in_domain_2024.json/.md` | 域内评估报告 |
| `results/eval_report_v2_transfer_2025.json/.md` | 迁移评估报告 |
| `results/transfer_comparison_v2.md` | 域内 vs 迁移对比表 |

### 论文叙事要点

迁移评估完成后，可填充论文中 **Table X：跨年份迁移效果对比**，重点对比：
- `macro_soft_2024_v2` vs `ablation_no_causal_v2`：causal mask 对迁移稳健性的影响
- `macro_soft_2024_v2` vs `ablation_no_hierarchy_v2`：分层结构对迁移稳健性的影响
- `macro_soft_2024_v2` vs `baseline_tabddpm_v2` / CTGAN / TVAE / SMOTE：与经典基线的差距

---

## P1：2024→2025 漂移分析

### 目标
找出在 2024→2025 之间分布漂移最大的变量，将其从因果约束中弱化或移出。

```bash
python pipeline/analyze_drift.py \
  --train_csv data/nyc_crash_2024_v2/train.csv \
  --test_csv data/nyc_crash_2025_v2/test.csv \
  --output results/drift_analysis_2024_2025_v2.csv
```

### 使用结果
- 漂移大的变量（如某些特定道路类型）：降低其在 soft mask 中的权重
- 漂移小的变量（如季节、经纬度）：保持或加强约束
- 将漂移分析结果记录到当天实验日志

---

## P2：评测口径统一

### 当前需要确认

1. `evaluate_all.py` 已支持通过 `--real_test` 指定任意测试集，口径统一 ✅
2. 2024 域内测试集和 2025 迁移测试集样本数：16,739 vs 16,540，基本一致 ✅
3. TSTR 下游模型在合成数据上训练，在真实测试集上评测，口径统一 ✅

### 标准化命令（v2 专用）

```bash
# 域内评测（2024 synthetic → 2024 real）
python pipeline/evaluate_all.py \
  --real_test data/nyc_crash_2024_v2/test.csv \
  --file_glob "*_v2_full.csv" \
  --primary_metrics_profile no_rule \
  --output_tag v2_in_domain

# 迁移评测（2024 synthetic → 2025 real）
python pipeline/run_transfer_eval.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --output_tag v2_transfer \
  --primary_metrics_profile no_rule
```

---

## 暂缓的方向（需要更多探索的）

以下方向目前不优先推进，因为当前主线数据/mask 问题尚未解决：

- **H3 r8 road-cell 机制实验**：保留代码，待主线稳定后做消融
- **Stage2 天气/OSM 强制启用**：Stage2 加入分层链路后需要重新评测整体指标
- **语义 CE 替代方案**：2026-04-29 日志建议改用"诊断层面语义一致性指标"，待主线稳定后实施
- **采样数量扩展**（>10000）：当前 10000 已足够，不优先
