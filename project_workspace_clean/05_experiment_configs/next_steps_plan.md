# 后续实验计划：消融优化 → 迁移学习

> 前提：exp_c_macro_full_v1（full tier + Macro Consistency Loss）训练已完成，且结果符合预期（no_proxy TSTR > 0.40, proxy_gap < 0.05）。

---

## Phase 1：消融实验优化（域内）

### 背景
当前已有 4 组 v2 域内评估结果（macro_soft_2024_v2 / ablation_no_causal_v2 / ablation_no_hierarchy_v2 / baseline_tabddpm_v2），但均基于"包含 proxy 列"的 standard TSTR，指标虚高且模型间差异被抹平。

本阶段目标：**在统一且公平的评估标准下（no_proxy TSTR + macro_relation），重新评估所有模型**，验证主模型改进的真实增益。

### 评估标准统一

| 指标 | 说明 | 优先级 |
|------|------|--------|
| **no_proxy TSTR** | 排除 8 个 proxy 列后的 TSTR | P0（主指标） |
| **proxy_gap** | standard - no_proxy | P0（越小越好，<0.05） |
| **macro_relation MAE** | group-wise injury mean error | P1（<0.08） |
| **CMI error** | 条件互信息误差 | P1（<0.11） |
| **standard TSTR** | 保留，仅作参考 | P2 |

### 需重新评估的模型清单

```yaml
internal_models:
  - name: "ours_full_model"
    ckpt: "ckpt/nyc_crash_2024_v2/stage3_full_full_exp_c_macro_full_v1"
    desc: "主模型（Macro Consistency + Causal Guidance + Proxy 后置）"
  
  - name: "macro_sparse_anneal_v2"
    ckpt: "ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2"
    desc: "v2 基线主模型（无 Macro Loss，无 Proxy 后置）"
  
  - name: "ablation_no_causal_v2"
    ckpt: "ckpt/nyc_crash_2024_v2/stage3_full_full_ablation_no_causal_v2"
    desc: "消融：无 causal mask"
  
  - name: "ablation_no_hierarchy_v2"
    ckpt: "ckpt/nyc_crash_2024_v2/stage3_full_full_ablation_no_hierarchy_v2"
    desc: "消融：无 hierarchy"
  
  - name: "baseline_tabddpm_v2"
    ckpt: "ckpt/nyc_crash_2024_v2/stage3_full_full_baseline_tabddpm_v2"
    desc: "TabDDPM 基线"

external_baselines:
  - name: "CTGAN"
    csv: "results/synthetic/baseline_ctgan_full.csv"
  - name: "TVAE"
    csv: "results/synthetic/baseline_tvae_full.csv"
  - name: "SMOTE"
    csv: "results/synthetic/baseline_smote_full.csv"
```

### 执行步骤

#### Step 1.1：统一采样（internal models）

对所有 internal models 使用**相同的评估采样配置**：
- `condition_train_indices`: 5,000 条（基于训练集前 5000 行）
- `impute_condition`: x_0
- `impute_resample_rounds`: 1
- `do_postprocess`: True

```bash
# 采样脚本模板
IDX_FILE="results/_cache/eval_uniform_5k.txt"
seq 0 4999 > "$IDX_FILE"

for model in ours_full_model macro_sparse_anneal_v2 ablation_no_causal_v2 ablation_no_hierarchy_v2 baseline_tabddpm_v2; do
  python src/sample_conditional.py \
    --ckpt_dir "ckpt/nyc_crash_2024_v2/stage3_full_full_${model}" \
    --condition_train_indices "$IDX_FILE" \
    --num_samples 5000 --batch_size 500 --device cuda:0 \
    --output_csv "results/synthetic/eval_${model}_uniform5k.csv" \
    --do_postprocess
done
```

#### Step 1.2：统一评估

对所有采样结果（internal + external）运行 `evaluate_all.py`，强制 `--exclude_proxy_outcomes`：

```bash
for model in ours_full_model macro_sparse_anneal_v2 ablation_no_causal_v2 ablation_no_hierarchy_v2 baseline_tabddpm_v2 CTGAN TVAE SMOTE; do
  python pipeline/evaluate_all.py \
    --real_test "data/nyc_crash_2024_v2/test.csv" \
    --synthetic_dir "results/synthetic" \
    --file_list "eval_${model}_uniform5k_physical_raw_aligned.csv" \
    --info_json "data/nyc_crash_2024_v2/info.json" \
    --exclude_proxy_outcomes \
    --causal_eval_suite \
    --output_tag "v2_ablation_${model}"
done
```

#### Step 1.3：合并报告

使用 `merge_transfer_reports.py` 的变体（或新建 `merge_ablation_reports.py`）合并所有模型的 eval_report：

```python
# 伪代码
reports = []
for model in all_models:
    with open(f"results/eval_report_v2_ablation_{model}.json") as f:
        data = json.load(f)
    row = data["rows"][0]
    reports.append({
        "model": model,
        "no_proxy_tstr": row["no_proxy_tstr_avg_score"],
        "proxy_gap": row.get("proxy_leakage_avg_score_gap", 0),
        "macro_mae": row.get("macro_relation_mae_mean", float("nan")),
        "cmi_err": row.get("cmi_error_mean", float("nan")),
        "standard_tstr": row["tstr_avg_score"],
    })
# 输出 Markdown 表格
```

### 预期叙事

> "在排除 proxy 泄漏后， ours_full_model 的 no_proxy TSTR 为 X.XXX，显著优于 ablation_no_causal (X.XXX) 和 baseline_tabddpm (X.XXX)，证明因果结构和宏观一致性约束对真实预测力的提升。"

---

## Phase 2：迁移学习评估

### 背景
域内优化完成后，需验证模型在 2025 数据上的迁移能力。2025 与 2024 的差异主要在于分布漂移（COVID 后交通模式变化）。

### 评估方案

#### 方案 A：直接迁移（Zero-Shot）

用 2024 训练的最优模型，直接在 2025 test 上采样并评估。

```bash
# 采样：2024 模型 → 2025 条件
python src/sample_conditional.py \
  --ckpt_dir "ckpt/nyc_crash_2024_v2/stage3_full_full_exp_c_macro_full_v1" \
  --data_dir "data/nyc_crash_2025_v2" \
  --condition_train_indices "results/_cache/eval_2025_5k.txt" \
  --num_samples 5000 --device cuda:0 \
  --output_csv "results/synthetic/transfer_2025_ours.csv"

# 评估
python pipeline/evaluate_all.py \
  --real_test "data/nyc_crash_2025_v2/test.csv" \
  --synthetic_dir "results/synthetic" \
  --file_list "transfer_2025_ours_physical_raw_aligned.csv" \
  --info_json "data/nyc_crash_2025_v2/info.json" \
  --exclude_proxy_outcomes \
  --causal_eval_suite \
  --output_tag "transfer_2025_ours"
```

#### 方案 B：微调迁移（Fine-Tune）

用 2024 模型作为预训练权重，在 2025 训练数据上微调少量 epochs（如 200-500）：

```bash
python src/train_hierarchical.py \
  --stage 3 --tier full \
  --dataname nyc_crash_2025_v2 \
  --macro_relation_weight 0.01 \
  --lambda_causal 1.0 \
  --device cuda:0 \
  --experiment_id exp_c_macro_full_v1_finetune_2025
  # TODO: 需添加 --pretrained_ckpt 参数支持
```

> 注：当前 `train_hierarchical.py` 不支持加载预训练权重微调。如需实现，需修改 `load_model` 逻辑。

#### 方案 C：逐年链式迁移（Year-by-Year Chain）

如果 2024→2025 漂移大，可尝试 2024→2025 的链式分层采样：
- Stage1（空间）：2024 模型直接用于 2025（空间分布变化小）
- Stage2（上下文）：2024 模型直接用于 2025
- Stage3（结果）：用 2025 数据微调或直接迁移

### 迁移评估指标

| 指标 | 说明 |
|------|------|
| **迁移 TSTR** | 2025 test 上的 no_proxy TSTR |
| **退化率** | (域内 TSTR - 迁移 TSTR) / 域内 TSTR |
| **W-num / JS-cat** | 连续/分类分布距离 |
| **macro_relation MAE (2025)** | 2025 数据上的宏观关系匹配度 |

### 基线对比

所有迁移评估需与以下基线同步进行：
- CTGAN / TVAE / SMOTE（在 2024 数据上训练，生成 2025 分布）
- ablation_no_causal_v2（2024 模型直接迁移）
- baseline_tabddpm_v2（2024 模型直接迁移）

### 预期叙事

> "尽管 2025 数据存在显著分布漂移（W-num 从 0.21 升至 0.88），ours_full_model 的迁移 TSTR 仅退化 X%，优于 TabDDPM 的 Y% 退化，证明因果结构先验增强了跨年度泛化能力。"

---

## 关键风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| exp_c_macro_full_v1 训练失败或 loss 不收敛 | 已保存 balanced tier 成功 checkpoint 作为 fallback |
| no_proxy TSTR 所有模型仍接近（~0.39） | 检查 injury bins 后置规则是否仍泄漏；考虑进一步收紧 proxy 定义 |
| 2025 迁移时 macro_relation MAE 恶化 | 这是预期现象（分布漂移），重点看相对退化率而非绝对值 |
| evaluate_all.py macro_MAE/CMI 输出 nan | 修复 `--causal_eval_suite` 与 `--enable_macro_relations` 参数冲突 |

---

## 执行顺序检查清单

- [ ] Phase 1.1：对所有 internal models 统一采样 5000 条
- [ ] Phase 1.2：统一运行 evaluate_all.py（--exclude_proxy_outcomes）
- [ ] Phase 1.3：合并报告，生成 ablation 对比表
- [ ] Phase 2.0：确认域内最优模型（ours_full_model vs 其他）
- [ ] Phase 2.1：最优模型 zero-shot 迁移到 2025
- [ ] Phase 2.2：所有基线模型同步迁移到 2025
- [ ] Phase 2.3：合并迁移报告，计算退化率
- [ ] Phase 2.4（可选）：2025 微调实验
- [ ] 最终：整理所有代码、报告、日志到 workspace_clean，准备上传
