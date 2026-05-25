# v2 Schema 2025 迁移评估执行指南

> 生成日期：2026-05-23  
> 适用场景：`macro_soft_2024_v2` 及所有消融/基线模型的 2024→2025 迁移学习效果补全

---

## 一、当前状态速览

| 模型 | 2024 域内合成数据 | 2024 域内评估 | 2025 迁移评估 |
|------|------------------|--------------|--------------|
| `macro_soft_2024_v2` | ✅ | ✅ | ❌ **待补** |
| `ablation_no_causal_v2` | ✅ | ✅ | ❌ **待补** |
| `ablation_no_hierarchy_v2` | ✅ | ✅ | ❌ **待补** |
| `baseline_tabddpm_v2` | ✅ | ✅ | ❌ **待补** |
| `baseline_ctgan` | ✅ | ✅ | ❌ **待补** |
| `baseline_tvae` | ✅ | ✅ | ❌ **待补** |
| `baseline_smote` | ✅ | ✅ | ❌ **待补** |

**迁移测试集已就绪**：`data/nyc_crash_2025_v2/test.csv`（16,540 行）

---

## 二、新增/完善脚本清单

| 脚本 | 路径 | 用途 |
|------|------|------|
| `run_external_baselines.py` | `03_evaluation_pipeline/` | 生成 CTGAN/TVAE/SMOTE 合成数据 |
| `run_transfer_eval.py` | `03_evaluation_pipeline/` | **一键批量迁移评估**：自动聚合所有 v2 模型+基线，统一在 2025 测试集上评估 |
| `merge_transfer_reports.py` | `03_evaluation_pipeline/` | 合并域内报告与迁移报告，生成对比表格 |
| `run_benchmark_suite.py` | `03_evaluation_pipeline/` | 更新：新增 `--transfer_test` 参数，支持端到端基准套件含迁移评估 |

---

## 三、一键执行命令

### 3.1 快速方案：仅补迁移评估（推荐）

如果你已经拥有所有 v2 合成数据，直接运行：

```bash
# 1. 批量迁移评估（自动收集所有模型，统一在 2025 测试集上跑）
python pipeline/run_transfer_eval.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --output_tag v2_transfer_2025 \
  --primary_metrics_profile no_rule

# 2. 域内评估（如之前未跑）
python pipeline/evaluate_all.py \
  --real_test data/nyc_crash_2024_v2/test.csv \
  --file_glob "*_v2_full.csv" \
  --primary_metrics_profile no_rule \
  --output_tag v2_in_domain_2024

# 3. 合并对比报告
python pipeline/merge_transfer_reports.py \
  --in_domain_json results/eval_report_v2_in_domain_2024.json \
  --transfer_json results/eval_report_v2_transfer_2025.json \
  --output results/transfer_comparison_v2.md
```

### 3.2 完整方案：重新生成所有基线 + 评估

如果你需要重新生成外部基线（CTGAN/TVAE/SMOTE）：

```bash
# 1. 外部基线生成
python pipeline/run_external_baselines.py \
  --tier full --dataname nyc_crash_2024_v2

# 2. 批量迁移评估
python pipeline/run_transfer_eval.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --output_tag v2_transfer_2025 \
  --primary_metrics_profile no_rule
```

### 3.3 端到端方案：run_benchmark_suite

```bash
python pipeline/run_benchmark_suite.py \
  --tier full \
  --transfer_test data/nyc_crash_2025_v2/test.csv \
  --skip_internal_train
```

> `--skip_internal_train`：跳过内部模型训练（假设已有 checkpoint），只跑外部基线+评估+打包。

---

## 四、dry_run 预检

在正式跑之前，先检查会评估哪些文件：

```bash
python pipeline/run_transfer_eval.py --dry_run
```

预期输出示例：

```
[transfer_eval] 发现 7 个合成文件:
  - macro_soft_2024_v2_full.csv
  - ablation_no_causal_v2_full.csv
  - ablation_no_hierarchy_v2_full.csv
  - baseline_tabddpm_v2_full.csv
  - baseline_ctgan_full.csv
  - baseline_tvae_full.csv
  - baseline_smote_full.csv
```

---

## 五、预期产出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 迁移评估报告 | `results/eval_report_v2_transfer_2025.md` | Markdown 格式完整评估结果 |
| 迁移评估 JSON | `results/eval_report_v2_transfer_2025.json` | JSON 格式，供后续脚本读取 |
| 域内评估报告 | `results/eval_report_v2_in_domain_2024.md` | 域内对比基准 |
| 合并对比表 | `results/transfer_comparison_v2.md` | 域内 vs 迁移关键指标对比 |

---

## 六、论文用结果汇总表模板

迁移评估完成后，可将以下指标填入论文表格：

### Table X：v2 模型 2024 域内 vs 2025 迁移效果对比

| 模型 | 评估集 | W-num ↓ | JS-cat ↓ | TSTR avg ↑ | R2 ↑ | 备注 |
|------|--------|:-------:|:--------:|:----------:|:----:|------|
| `macro_soft_2024_v2` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | 当前主模型 |
| `macro_soft_2024_v2` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `ablation_no_causal_v2` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | 去除因果约束 |
| `ablation_no_causal_v2` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `ablation_no_hierarchy_v2` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | 去除分层结构 |
| `ablation_no_hierarchy_v2` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `baseline_tabddpm_v2` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | TabDDPM 基线 |
| `baseline_tabddpm_v2` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `baseline_ctgan` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | CTGAN 基线 |
| `baseline_ctgan` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `baseline_tvae` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | TVAE 基线 |
| `baseline_tvae` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |
| `baseline_smote` | 2024 域内 | 待填 | 待填 | 待填 | 待填 | SMOTE 基线 |
| `baseline_smote` | 2025 迁移 | 待填 | 待填 | 待填 | 待填 | |

> 数据来源：`results/transfer_comparison_v2.md`

---

## 七、注意事项

1. **Windows 路径**：如果直接在 PowerShell 中运行，注意路径分隔符用 `/` 或 `\` 均可。
2. **评估耗时**：单个模型评估约 2-5 分钟（取决于 TSTR 下游模型训练时间），7 个模型总耗时约 15-30 分钟。
3. **info.json**：v2 评估建议显式传入 `--info_json data/nyc_crash_2024_v2/info.json`，确保 schema 一致。
4. **外部基线文件名**：CTGAN/TVAE/SMOTE 的合成文件名不含 `_v2` 后缀，`run_transfer_eval.py` 已通过 `DEFAULT_BASELINE_PATTERNS` 自动识别。

---

## 八、常见问题排查

### Q1: `run_transfer_eval.py` 找不到合成文件？

检查 `results/synthetic/` 目录下是否存在以下文件：

```bash
ls results/synthetic/*_v2_full.csv
ls results/synthetic/baseline_*_full.csv
```

如果不存在，需先运行 `run_all_experiments.py`（内部模型）或 `run_external_baselines.py`（外部基线）生成。

### Q2: 迁移评估报告为空或只有部分模型？

可能是 `file_glob` 不匹配。使用 `--dry_run` 检查文件列表，或用 `--include_patterns` 手动指定：

```bash
python pipeline/run_transfer_eval.py \
  --include_patterns "*_v2_full.csv" "baseline_ctgan_full.csv" \
  --dry_run
```

### Q3: 如何只评估特定模型？

```bash
python pipeline/evaluate_all.py \
  --real_test data/nyc_crash_2025_v2/test.csv \
  --file_glob "macro_soft_2024_v2_full.csv" \
  --primary_metrics_profile no_rule \
  --output_tag v2_transfer_macro_soft_only
```
