# 评测链路说明

> 主评测入口：`evaluate_all.py`  
> 当前主线评测口径：**2024 域内 + 2025 迁移**  
> 评价重心：跨年份迁移稳健性 + 结构真实性（不以 R² 为主指标）

---

## 评测链路总览

```
合成数据（results/synth_macro_soft_2024_*.csv）
        ↓ postprocess_samples.py（坐标反归一化，不做硬修复）
        ↓ evaluate_all.py
评测结果（results/eval/）
  ├─ 2024 域内：tstr_avg_score / wass_num / mean_js_cat / tstr_r2
  └─ 2025 迁移：同上指标，测试集换为 2025 年数据
```

---

## 当前正式评测指标

### 主指标

| 指标 | 含义 | 方向 |
|------|------|------|
| `tstr_avg_score` | TSTR 下游分类平均得分（XGBoost/MLP/LR） | ↑ 越大越好 |
| `wass_num` | 连续变量 Wasserstein 距离 | ↓ 越小越好 |
| `mean_js_cat` | 类别变量 JS 散度均值 | ↓ 越小越好 |

### 辅助指标（不作为主结论依据）

| 指标 | 说明 |
|------|------|
| `tstr_r2` | TSTR 回归 R²（伤亡人数预测），跨年份本身不稳定，不作主指标 |
| SHD | 结构汉明距离（因果图结构准确性） |
| `structured_eval` | 事故类型/车辆组合语义一致性 |

---

## 各评测脚本职责

| 脚本 | 职责 | 对应场景 |
|------|------|---------|
| `evaluate_all.py` | **主评测入口**；单轮评估指定目录下的合成 CSV | 当前主线 |
| `run_transfer_eval.py` | **批量迁移评估**；自动聚合内部模型+外部基线，统一在迁移测试集上评估 | 2025 迁移评估 |
| `run_external_baselines.py` | 生成 CTGAN/TVAE/SMOTE 合成数据 | 外部基线对比 |
| `evaluate_postcovid_transfer.py` | ⚠️ 旧路线：postcovid 2017→2025 | 已归档，非主线 |
| `evaluate_joint_metrics.py` | 联合分布指标（多变量联合 JS/Wasserstein） | 辅助 |
| `run_all_experiments.py` | 批量运行多个 experiment_id 的训练+采样 | 消融对比 |
| `run_benchmark_suite.py` | 端到端基准套件（内部模型+外部基线+评估+打包） | 完整对比实验 |
| `sample_conditional.py` | 条件采样脚本（与 evaluate_all 配合） | 采样 |
| `metrics.py` | 指标计算函数库 | 被其他脚本调用 |
| `structured_eval.py` | 结构化语义评测（车辆/事故类型一致性） | 辅助 |

---

## 标准评测命令

### v2 域内评测（2024）

```bash
python pipeline/evaluate_all.py \
  --real_test data/nyc_crash_2024_v2/test.csv \
  --file_glob "*_v2_full.csv" \
  --primary_metrics_profile no_rule \
  --output_tag v2_in_domain_2024
```

### v2 迁移评测（2024→2025）

```bash
python pipeline/run_transfer_eval.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --output_tag v2_transfer_2025 \
  --primary_metrics_profile no_rule
```

### 批量消融对比（训练+采样）

```bash
python pipeline/run_all_experiments.py \
  --model macro_soft_2024_v2 --tier full --dataname nyc_crash_2024_v2

python pipeline/run_all_experiments.py \
  --model ablation_no_causal --tier full --dataname nyc_crash_2024_v2

python pipeline/run_all_experiments.py \
  --model ablation_no_hierarchy --tier full --dataname nyc_crash_2024_v2
```

### 外部基线生成（CTGAN/TVAE/SMOTE）

```bash
python pipeline/run_external_baselines.py \
  --tier full --dataname nyc_crash_2024_v2
```

### 端到端基准套件（含迁移评估）

```bash
python pipeline/run_benchmark_suite.py \
  --tier full \
  --transfer_test data/nyc_crash_2025_v2/test.csv
```

---

## 当前最新主线结果摘要

> 来源：2026-05-08 日志

| 模型 | 评估集 | tstr_avg_score ↑ | wass_num ↓ | mean_js_cat ↓ |
|------|--------|:---:|:---:|:---:|
| macro_soft_2024 | 2024 域内 | ~0.41 | ~0.33 | ~0.010 |
| macro_soft_2024 | 2025 迁移 | ~0.38 | ~0.45 | ~0.020 |
| our_model_no_h3 | 2024 域内 | 参见 05-08 日志 | — | — |
| semantic_heads_vc_2024 | 2024 域内 | 0.396 | 0.436 | 0.014 |

> 精确数值请查阅 `06_logs_and_reports/Experiment_Logs/2026-05-08.md`

---

## 不再作为主结论依据的旧评测方式

1. **`evaluate_postcovid_transfer.py`**：迁移口径已改为 2024→2025，不再使用 2017→2025。
2. **单纯 R² 排名**：R² 在跨年份迁移场景本身不稳定，不作为主比较指标。
3. **采样后硬修复（`--semantic_repair`）再评测**：已验证显著恶化 mean_js_cat，不在主线使用。

---

## 注意事项

1. `evaluate_all.py` 中的 `_inverse_transform_continuous` 对物理空间 CSV 安全（NYC 经纬度范围检测会短路），无需 `--no_inverse_real` 参数。
2. 对于 vocab 离散化连续目标（如伤亡人数 bin 化），推荐在评测时用 softmax 期望值 `E[v]=Σ p_i·v_i` 解码，而非 argmax，可降低 RMSE。
