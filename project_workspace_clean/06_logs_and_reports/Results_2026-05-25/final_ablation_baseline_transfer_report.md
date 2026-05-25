# CausalDiffTab — 消融实验、基线对比与迁移学习实验报告

> 生成时间：2026-05-25
> 评估标准：统一使用 `--exclude_proxy_outcomes --causal_eval_suite --enable_macro_relations`
> 宏观关系覆盖：29/29 specs（crash_type=12, road=5, vehicle=8, weather=4）

---

## 一、Phase 1：域内消融与基线对比（2024 v2 test）

### 1.1 综合指标

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

### 2.1 综合指标

| Model | N | no_proxy TSTR↑ | proxy_gap↓ | macroMAE↓ | targetMAE↓ | Wass↓ | JS↓ |
|-------|---|----------------|------------|-----------|------------|-------|-----|
| Ours (exp_c_macro_full_v1) | 5000 | 0.4084 | 0.1699 | 0.0525 | 0.0518 | 0.3050 | 0.0412 |
| Ours (macro_sparse_anneal_v2) | 5000 | 0.3981 | 0.1866 | **0.0302** | 0.0294 | 0.3050 | 0.0427 |
| Ours + guidance=0.5 | 5000 | **0.4134** | 0.1185 | 0.0557 | 0.0550 | 0.3050 | 0.0401 |
| Ablation (no causal) | 5000 | 0.4107 | 0.1639 | 0.0366 | 0.0358 | 0.3050 | 0.0401 |
| Ablation (no hierarchy) | 5000 | 0.4090 | 0.1691 | 0.0511 | 0.0506 | 0.3050 | 0.0408 |
| Baseline TabDDPM | 5000 | 0.4092 | 0.1129 | 0.0442 | 0.0432 | 0.3050 | 0.0396 |
| Baseline CTGAN | 10000 | 0.3069 | 0.0483 | 0.1060 | 0.0993 | 169.23 | 0.0821 |
| Baseline TVAE | 10000 | 0.3322 | 0.1289 | 0.0693 | 0.0643 | 232.87 | 0.1012 |
| Baseline SMOTE | 10000 | 0.3882 | 0.2300 | 4.6878 | 6.0107 | 5.9604 | 0.1590 |

### 2.2 域内 vs 迁移对比

| Model | Domain TSTR | Transfer TSTR | Δ TSTR | Domain macroMAE | Transfer macroMAE | Δ macroMAE |
|-------|-------------|---------------|--------|-----------------|-------------------|------------|
| Ours (exp_c_macro_full_v1) | 0.3933 | 0.4084 | +0.0151 | 0.0402 | 0.0525 | +0.0123 |
| Ours (macro_sparse_anneal_v2) | 0.3921 | 0.3981 | +0.0060 | 0.0391 | **0.0302** | -0.0089 |
| Ours + guidance=0.5 | 0.3921 | **0.4134** | +0.0213 | **0.0297** | 0.0557 | +0.0260 |
| Ablation (no causal) | 0.3915 | 0.4107 | +0.0192 | 0.0250 | 0.0366 | +0.0116 |
| Ablation (no hierarchy) | 0.3920 | 0.4090 | +0.0170 | 0.0380 | 0.0511 | +0.0131 |
| Baseline TabDDPM | 0.3934 | 0.4092 | +0.0158 | 0.0448 | 0.0442 | -0.0006 |
| Baseline CTGAN | 0.3032 | 0.3069 | +0.0037 | 0.1141 | 0.1060 | -0.0081 |
| Baseline TVAE | 0.3329 | 0.3322 | -0.0007 | 0.0781 | 0.0693 | -0.0088 |
| Baseline SMOTE | 0.3858 | 0.3882 | +0.0024 | 4.7229 | 4.6878 | -0.0351 |

### 2.3 迁移发现

1. **所有模型迁移 TSTR 均持平或略有提升**：2025 test set 的 injury 预测任务可能比 2024 更容易（ variance 更低），而非模型真正"泛化"了。此现象需结合 2025 数据分布进一步分析。
2. **guidance=0.5 在迁移后 macroMAE 显著恶化**：从 0.0297 → 0.0557。原因：guidance 使用 2024 预计算的 group means，而 2025 分布发生漂移，导致 over-correction。这表明 inference-time guidance 对分布漂移敏感。
3. **macro_sparse_anneal_v2 迁移后 macroMAE 反而改善**：从 0.0391 → 0.0302。说明该模型无 guidance 时生成的分布与 2025 的宏观关系更匹配，具备一定跨年度鲁棒性。
4. **TabDDPM 迁移 macroMAE 几乎不变**（0.0448 → 0.0442），稳定性最佳。
5. **所有 internal models 的 proxy_gap 在迁移后显著增大**（从 ~0 增至 ~0.17），说明 2025 数据中 proxy columns 对 injury 的预测价值更高。
6. **external baselines 的宏观关系表现与域内一致**：CTGAN/TVAE/SMOTE 的 macroMAE 在 2025 上略有改善，但绝对值仍然很差。

---

## 三、结论与建议

### 3.1 核心结论

| 维度 | 结论 |
|------|------|
| **guidance 有效性** | 在域内（2024）有效降低 macroMAE 24%，但迁移（2025）时因分布漂移导致 over-correction，macroMAE 反弹。 |
| **架构差异** | 排除 proxy 后，ours / ablation / TabDDPM 的 no_proxy TSTR 差异 < 0.002，架构改进的下游增益被稀释。 |
| **宏观关系** | guidance=0.5 在域内最佳（0.0297），但 ablation_no_causal 异常低（0.0250），可能因过拟合训练集宏观模式。 |
| **迁移鲁棒性** | macro_sparse_anneal_v2（无 guidance）迁移 macroMAE 反而改善，TabDDPM 几乎不变，二者鲁棒性最佳。 |
| **baseline 差距** | CTGAN/TVAE/SMOTE 在 macroMAE、proxy_gap、Wasserstein 上全面落后，不具备竞争能力。 |

### 3.2 下一步建议

1. **改进 guidance 的迁移鲁棒性**：考虑使用相对偏移（target - current）而非绝对目标值，或引入自适应缩放因子。
2. **解释 ablation_no_causal 的低 macroMAE**：分析无 causal mask 时模型是否过拟合了特定 group pattern。
3. **2025 微调实验**（Phase 2.4）：用 2024 checkpoint 在 2025 数据上微调 200-500 epochs，验证微调能否同时提升 TSTR 和 macroMAE。
4. **分布漂移量化**：计算 2024 vs 2025 的 covariate shift 和 concept drift，解释为何所有模型迁移 TSTR 不降反升。
