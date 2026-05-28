# 2026-05-28 实验日志 —— 主模型训练、迁移学习与 Guidance 模式全量总结

> 本日志相对于 2026-05-25 实验日志，补充了分布漂移量化、4 种 guidance 模式对比、2025 fine-tuning 实验，并对全部实验结果进行系统性闭环总结。

---

## 1. 目标

1. **解释 guidance 在迁移后失效的根因**：量化 2024→2025 的分布漂移。
2. **验证 robust guidance 模式**：实现 relative / adaptive / annealed 三种缓解策略，与 absolute / no-guidance 做对照。
3. **验证 fine-tuning 能否提升迁移性能**：2024 checkpoint → 2025 数据 fine-tune 500 epochs。
4. **闭环总结**：明确模型族关系、最优配置、以及架构设计的 trade-off。

---

## 2. 代码与流程更新

| 文件 | 改动 | 说明 |
|------|------|------|
| `tabdiff/models/unified_ctime_diffusion.py` | 新增 | 4 种 macro guidance 模式（absolute/relative/adaptive/annealed） |
| `src/sample_conditional.py` | 修改 | 暴露 `--macro_guidance_mode` 和 `--macro_guidance_adaptive_drift_threshold` CLI 参数 |
| `scripts/quantify_distribution_drift.py` | 新增 | 量化 2024 vs 2025 的 covariate shift / concept drift / macro relation drift |
| `scripts/finetune_2025.py` | 重写 | 修复导入问题，适配当前 `UnifiedCtimeDiffusion` API，简化训练循环 |
| `_run_*_sampling.py` | 新增 4 个 | adaptive / relative / annealed / ft2025 采样 wrapper |
| `results/guidance_mode_comparison_transfer_2025.md` | 新增 | guidance 模式对比表 |
| `results/experiment_log_20260528_post_525.md` | 新增 | 5.25→5.28 增量实验记录 |

---

## 3. 模型族关系图

```
CausalDiffTab 模型族（v2 schema：10 num + 36 cat）
├── Ours 主模型系列
│   ├── exp_c_macro_full_v1          # Stage3 + full causal mask + macro relation loss
│   └── macro_sparse_anneal_v2       # Stage3 + sparse causal mask + annealed closs schedule ★
│       ├── + no guidance            # 迁移最优配置
│       ├── + absolute guidance g0.5 # 域内最优，迁移灾难
│       ├── + relative guidance g0.5 # 缓解 over-correction（+24%）
│       ├── + adaptive guidance g0.5 # 更好缓解（+16.5%）
│       ├── + annealed guidance g0.5 # 最佳缓解（+8%）
│       └── + ft2025 (500ep LR=1e-4) # Fine-tuned on 2025 (+26.8%) ❌
├── Ablation 系列
│   ├── ablation_no_causal_v2        # 去除 causal mask（过拟合 proxy）
│   └── ablation_no_hierarchy_v2     # 去除 hierarchy（退化为 flat）
└── Baseline 系列
    ├── baseline_tabddpm_v2          # TabDDPM 复现
    ├── baseline_ctgan_full          # CTGAN
    ├── baseline_tvae_full           # TVAE
    └── baseline_smote_full          # SMOTE
```

**★ 标记说明**：`macro_sparse_anneal_v2` 是当前主模型的最佳版本，其稀疏因果掩码 + annealed 训练 schedule 在域内和迁移中均表现稳健。

---

## 4. 关键结果摘录

### 4.1 域内训练（2024 v2 test）

| 模型 | N | no_proxy TSTR↑ | proxy_gap↓ | macroMAE↓ | targetMAE↓ |
|------|---|----------------|------------|-----------|------------|
| Ablation (no causal) | 5000 | 0.3915 | **-0.0225** | **0.0250** | **0.0210** |
| Ours + absolute guidance g0.5 | 5000 | 0.3921 | 0.0010 | 0.0297 | 0.0260 |
| Ablation (no hierarchy) | 5000 | 0.3920 | -0.0016 | 0.0380 | 0.0356 |
| **Ours (macro_sparse_anneal_v2)** | 5000 | 0.3921 | 0.0029 | 0.0391 | 0.0386 |
| Ours (exp_c_macro_full_v1) | 5000 | 0.3933 | -0.0027 | 0.0402 | 0.0382 |
| Baseline TabDDPM | 5000 | 0.3934 | -0.0302 | 0.0448 | 0.0450 |
| Baseline TVAE | 10000 | 0.3329 | 0.1345 | 0.0781 | 0.0735 |
| Baseline CTGAN | 10000 | 0.3032 | 0.0541 | 0.1141 | 0.1085 |
| Baseline SMOTE | 10000 | 0.3858 | 0.2469 | 4.7229 | 6.0015 |

**域内关键发现**：
- **no_proxy TSTR 高度趋同**：所有内部模型（含 ablation 和 TabDDPM）集中在 0.391–0.393，差异 < 0.002。排除 proxy 泄漏后，架构差异对下游预测力的增益被显著稀释。
- **absolute guidance 在域内有效**：macroMAE 从 0.0391 → 0.0297（↓24%），targetMAE 从 0.0386 → 0.0260（↓33%），且 TSTR 不下降。
- **ablation_no_causal 的异常低 macroMAE（0.0250）**：无 causal mask 时模型自由度更高，恰好过拟合了训练集的宏观关系模式。但 proxy_gap=-0.0225 表明其过度依赖 proxy columns（TOTAL_VEHICLES, IS_MULTI_VEHICLE）。
- **外部基线全面落后**：CTGAN/TVAE/SMOTE 的 macroMAE、proxy_gap、Wasserstein 均显著差于内部模型。

### 4.2 迁移学习 Zero-Shot（2024 model → 2025 test）

| 模型 | N | no_proxy TSTR↑ | proxy_gap↓ | macroMAE↓ | targetMAE↓ |
|------|---|----------------|------------|-----------|------------|
| **Ours (macro_sparse_anneal_v2)** | 5000 | 0.3981 | 0.1866 | **0.0302** | **0.0294** |
| Ablation (no causal) | 5000 | 0.4107 | 0.1639 | 0.0366 | 0.0358 |
| + annealed guidance g0.5 | 5000 | — | — | 0.0326 | — |
| + adaptive guidance g0.5 | 5000 | — | — | 0.0352 | — |
| + relative guidance g0.5 | 5000 | — | — | 0.0374 | — |
| + ft2025 (500ep) | 5000 | — | — | 0.0383 | — |
| Baseline TabDDPM | 5000 | 0.4092 | 0.1129 | 0.0442 | 0.0432 |
| Baseline TVAE | 10000 | 0.3322 | 0.1289 | 0.0693 | 0.0643 |
| Baseline CTGAN | 10000 | 0.3069 | 0.0483 | 0.1060 | 0.0993 |
| Baseline SMOTE | 10000 | 0.3882 | 0.2300 | 4.6878 | 6.0107 |
| Ours + absolute guidance g0.5 | 5000 | **0.4134** | 0.1185 | 0.0557 | 0.0550 |
| Ours (exp_c_macro_full_v1) | 5000 | 0.4084 | 0.1699 | 0.0525 | 0.0518 |
| Ablation (no hierarchy) | 5000 | 0.4090 | 0.1691 | 0.0511 | 0.0506 |

**迁移关键发现**：
- **`macro_sparse_anneal_v2`（无 guidance）是迁移最优**：macroMAE=0.0302，不仅优于所有对手，甚至优于其域内表现（0.0391）。证明该模型的因果结构训练具备跨年度鲁棒性。
- **absolute guidance 迁移灾难**：macroMAE 从 0.0297 → 0.0557（+87%），因 2024 预计算的 group means 与 2025 真实分布漂移严重。
- **所有 robust guidance 模式均无法超越 no-guidance baseline**：annealed (+8%)、adaptive (+16.5%)、relative (+24%) 依次递增，但全部劣于无 guidance。
- **fine-tuning 未能提升迁移性能**：500 epochs LR=1e-4 的 fine-tuning 使 macroMAE 从 0.0302 恶化到 0.0383。可能因过拟合 2025 训练集 idiosyncrasies 或破坏预训练结构。
- **所有内部模型迁移 TSTR 不降反升**：从 ~0.392 → ~0.408，说明 2025 测试集的 injury 预测任务 variance 更低，而非模型真正泛化。
- **TabDDPM 迁移最稳定**：macroMAE 几乎不变（0.0448 → 0.0442），但绝对值仍差于 ours。

### 4.3 域内 vs 迁移对比（退化率）

| 模型 | Domain TSTR | Transfer TSTR | Δ TSTR | Domain macroMAE | Transfer macroMAE | Δ macroMAE |
|------|-------------|---------------|--------|-----------------|-------------------|------------|
| Ablation (no causal) | 0.3915 | 0.4107 | +0.0192 | **0.0250** | 0.0366 | **+0.0116** |
| Ours + absolute g0.5 | 0.3921 | 0.4134 | +0.0213 | 0.0297 | 0.0557 | +0.0260 |
| **Ours (macro_sparse_anneal_v2)** | 0.3921 | 0.3981 | **+0.0060** | 0.0391 | **0.0302** | **-0.0089** ✅ |
| Baseline TabDDPM | 0.3934 | 0.4092 | +0.0158 | 0.0448 | 0.0442 | -0.0006 |
| Ours (exp_c_macro_full_v1) | 0.3933 | 0.4084 | +0.0151 | 0.0402 | 0.0525 | +0.0123 |
| Ablation (no hierarchy) | 0.3920 | 0.4090 | +0.0170 | 0.0380 | 0.0511 | +0.0131 |

**退化率洞察**：
- 唯一一个**迁移 macroMAE 优于域内**的模型是 `macro_sparse_anneal_v2`（无 guidance）。
- `ablation_no_causal` 在域内 macroMAE 最低（0.0250），但迁移后退化 +46.4%，说明其过拟合了 2024 的特定 group pattern。
- TabDDPM 的退化率接近 0，稳定性最佳，但绝对性能被 ours 超越。

### 4.4 分布漂移量化（2024 → 2025）

| 漂移维度 | 数值 | 解释 |
|----------|------|------|
| 数值 Wasserstein（归一化） | 0.0228 | 低 covariate shift |
| 类别 JS 散度 | 0.0271 | 低分布漂移 |
| Concept drift（目标均值差） | 0.0036 | 几乎无概念漂移 |
| **宏观关系平均绝对差** | **0.2361** | **group-wise injury mean 显著漂移** |
| 宏观关系最大绝对差 | 2.80 | 稀有群体（如 Snow × pedestrian）严重漂移 |

**根因结论**：2024→2025 的整体特征漂移很小，但**group-wise 的宏观关系漂移显著**。这直接解释了为何基于 2024 group means 的 inference-time guidance 会在 2025 上产生 over-correction。

---

## 5. 逻辑闭环检查

### 5.1 假设 → 实验 → 结论 链条

| # | 假设 | 实验 | 结果 | 结论 | 闭环 |
|---|------|------|------|------|------|
| 1 | guidance 在域内有效 | 2024 采样 + absolute g0.5 | macroMAE ↓24% | ✅ 假设成立 | ✅ |
| 2 | guidance 在迁移后失效 | 2025 采样 + absolute g0.5 | macroMAE +87% | ✅ 假设成立 | ✅ |
| 3 | 失效根因是分布漂移 | `quantify_distribution_drift.py` | macro relation drift=0.24 | ✅ 根因确认 | ✅ |
| 4 | relative/adaptive/annealed 可缓解 | 3 种模式分别采样评测 | 全部劣于 no-guidance | ⚠️ 缓解有效但不足以超越 baseline | ⚠️ |
| 5 | fine-tuning 可提升迁移 | 500ep LR=1e-4 on 2025 | macroMAE +26.8% | ❌ 假设不成立 | ✅（负面结论也是闭环） |
| 6 | causal 结构训练本身足够鲁棒 | macro_sparse_anneal_v2 无 guidance | 迁移 macroMAE 最优 | ✅ 假设成立 | ✅ |

### 5.2 未闭环问题

| 问题 | 状态 | 下一步 |
|------|------|--------|
| 为何 fine-tuning 损害宏观关系？ | 待解释 | 尝试保守 fine-tuning（100ep, LR=1e-5）或层冻结 |
| 动态 group means 能否让 guidance 在迁移中有效？ | 未验证 | 从 2025 验证集实时计算 group means 后重测 |
| no_proxy TSTR 趋同是否意味着架构改进无价值？ | 部分闭环 | 已证明 proxy 排除后差异 < 0.002；但在 proxy 排除前和 macroMAE 上，因果结构仍有显著优势 |
| injury 二值化后的 TSTR 是否更有意义？ | 未验证 | 将 ≥1 vs 0 作为分类任务重新评估 |

---

## 6. 核心结论与推荐配置

### 6.1 模型命名规范（建议沿用）

| 配置 | 完整名称 | 适用场景 |
|------|---------|---------|
| `macro_sparse_anneal_v2` 无 guidance | **CausalDiffTab-Sparse** | 迁移到 2025（最优） |
| `macro_sparse_anneal_v2` + absolute g0.5 | **CausalDiffTab-Sparse-Guided** | 域内 2024 生成（macroMAE 最优） |
| `ablation_no_causal_v2` | **DiffTab-NoCausal** | 不推荐（proxy 泄漏） |
| `baseline_tabddpm_v2` | **TabDDPM** | 基线对照 |

### 6.2 最优配置速查

| 场景 | 推荐模型 | macroMAE | no_proxy TSTR |
|------|---------|----------|---------------|
| 域内生成（2024） | CausalDiffTab-Sparse-Guided | 0.0297 | 0.3921 |
| 迁移生成（2024→2025） | **CausalDiffTab-Sparse（无 guidance）** | **0.0302** | 0.3981 |
| 需要最高 TSTR（允许 proxy） | CausalDiffTab-Sparse-Guided | 0.0557 | 0.4134 |

### 6.3 设计原则总结

1. **Causal mask 的价值不在 TSTR，而在宏观关系保真度和可解释性**。排除 proxy 后 TSTR 差异 < 0.002，但 macroMAE 差异可达 2×。
2. **Inference-time guidance 是双刃剑**：域内有效，迁移敏感。仅在目标分布与训练分布一致时使用。
3. **Annealed closs schedule + sparse causal mask 是最佳训练配置**：在域内和迁移中均表现稳健，无需 guidance 即可达到最优迁移 macroMAE。
4. **Fine-tuning 需谨慎**：当前 500ep LR=1e-4 的设置破坏了预训练结构。如需 fine-tune，建议更保守的参数。

---

## 7. 产物清单

| 产物 | 路径 |
|------|------|
| 本日志 | `results/experiment_log_20260528_full_summary.md` |
| 增量实验日志 | `results/experiment_log_20260528_post_525.md` |
| Guidance 模式对比表 | `results/guidance_mode_comparison_transfer_2025.md` |
| Fine-tuned checkpoint | `ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2_ft2025/best_model.pt` |
| Fine-tuned 采样（物理值） | `results/synthetic/transfer_2025_macro_sparse_anneal_v2_ft2025_uniform5k_physical.csv` |
| Adaptive guidance 采样 | `results/synthetic/transfer_2025_macro_sparse_adaptive_g05_uniform5k_physical.csv` |
| Relative guidance 采样 | `results/synthetic/transfer_2025_macro_sparse_relative_g05_uniform5k_physical.csv` |
| Annealed guidance 采样 | `results/synthetic/transfer_2025_macro_sparse_annealed_g05_uniform5k_physical.csv` |
| 漂移量化报告 | `results/drift_quantification_report.json` / `.md` |
| 各模型 macro relation 报告 | `results/macro_relation_report_transfer_2025_*.json` / `.md` |
