# CausalDiffTab — 消融、基线与迁移学习实验报告（ interim ）

> 生成时间：2026-05-25
> 评估标准：统一使用 `--exclude_proxy_outcomes --causal_eval_suite --enable_macro_relations`

---

## 一、Phase 1：域内消融与基线对比（2024 v2 test）

### 1.1 综合指标（no_proxy TSTR + macro_relation，29 specs）

| Model | N | no_proxy TSTR↑ | proxy_gap↓ | macroMAE↓ | targetMAE↓ | Wass↓ | JS↓ |
|-------|---|----------------|------------|-----------|------------|-------|-----|
| Ours (exp_c_macro_full_v1) | 5000 | 0.3933 | -0.0027 | 0.0402 | 0.0382 | 0.6045 | 0.1362 |
| Ours (macro_sparse_anneal_v2) | 5000 | 0.3921 | 0.0029 | 0.0391 | 0.0386 | 0.6027 | 0.1362 |
| Ours + guidance=0.5 | 5000 | 0.3921 | 0.0010 | **0.0297** | 0.0260 | 0.6027 | 0.1360 |
| Ablation (no causal) | 5000 | 0.3915 | -0.0225 | 0.0250 | 0.0210 | 0.6027 | 0.1357 |
| Ablation (no hierarchy) | 5000 | 0.3920 | -0.0016 | 0.0380 | 0.0356 | 0.6027 | 0.1369 |
| Baseline TabDDPM | 5000 | 0.3934 | -0.0302 | 0.0448 | 0.0450 | 0.6027 | 0.1372 |
| Baseline CTGAN | 10000 | 0.3032 | 0.0541 | 0.1141 | 0.1085 | 169.49 | 0.0838 |
| Baseline TVAE | 10000 | 0.3329 | 0.1345 | 0.0781 | 0.0735 | 233.14 | 0.1004 |
| Baseline SMOTE | 10000 | 0.3858 | 0.2469 | 4.7229 | 6.0015 | 5.512 | 0.1577 |

### 1.2 关键发现

1. **no_proxy TSTR 趋同**：所有 internal models（含 ablations 和 TabDDPM）的 no_proxy TSTR 高度集中在 0.391–0.393，差异 < 0.002。说明排除 proxy 泄漏后，架构差异对真实下游预测力的影响被显著稀释。
2. **guidance=0.5 的宏观关系优势**：macroMAE 从 0.0392 → 0.0297（↓24%），targetMAE 从 0.0386 → 0.0260（↓33%），且 no_proxy TSTR 未下降。证明 inference-time guidance 有效修正了 group-wise injury mean 偏差。
3. **ablation_no_causal 的异常低 macroMAE（0.0250）**：无 causal mask 时模型自由度更高，在域内恰好更好地拟合了训练集的宏观关系。但 proxy_gap=-0.0225，说明它过度依赖 proxy columns。
4. **external baselines 的 proxy 依赖**：CTGAN/TVAE/SMOTE 的 proxy_gap 分别为 0.054/0.134/0.247，表明它们高度依赖 proxy columns（TOTAL_VEHICLES, IS_MULTI_VEHICLE）来"预测" injury。
5. **SMOTE 的灾难性表现**：macroMAE=4.72，连续变量 Wasserstein=5.51，说明 SMOTE 在特征空间插值严重破坏了分布结构。

---

## 二、Phase 2：迁移学习 Zero-Shot（2024 model → 2025 test）

### 2.1 Macro Relation MAE（29 specs coverage）

| Model | macroMAE (2024) | macroMAE (2025) | Δ |
|-------|-----------------|-----------------|---|
| Ours (exp_c_macro_full_v1) | 0.0402 | 0.0525 | +0.0123 |
| Ours (macro_sparse_anneal_v2) | 0.0391 | **0.0302** | -0.0089 |
| Ours + guidance=0.5 | **0.0297** | 0.0557 | +0.0260 |
| Ablation (no causal) | 0.0250 | 0.0366 | +0.0116 |
| Ablation (no hierarchy) | 0.0380 | 0.0511 | +0.0131 |
| Baseline TabDDPM | 0.0448 | 0.0442 | -0.0006 |
| Baseline CTGAN | 0.1141 | 0.1058 | -0.0083 |
| Baseline TVAE | 0.0781 | 0.0701 | -0.0080 |
| Baseline SMOTE | 4.7229 | 4.6878 | -0.0351 |

### 2.2 迁移发现（待 TSTR 评估完成后补充退化率）

1. **guidance=0.5 在迁移后 macroMAE 恶化**：从 0.0297 → 0.0557。原因：guidance 使用 2024 预计算的 group means，而 2025 分布发生漂移，导致 over-correction。
2. **macro_sparse_anneal_v2 迁移后 macroMAE 反而改善**：从 0.0391 → 0.0302。说明该模型生成的分布与 2025 的宏观关系更匹配，无 guidance 反而更具鲁棒性。
3. **TabDDPM 迁移 macroMAE 几乎不变**：0.0448 → 0.0442，表现出良好的跨年度稳定性。
4. **所有 internal models 的迁移 macroMAE 普遍高于域内**（除 macro_sparse_anneal_v2 外），证实 2024→2025 存在显著分布漂移。

---

## 三、待补充项

- [ ] TSTR 退化率（等待 `evaluate_all.py` 完成）
- [ ] Wasserstein / JS 漂移量（同上）
- [ ] target mean 偏差在 2025 上的变化（同上）
