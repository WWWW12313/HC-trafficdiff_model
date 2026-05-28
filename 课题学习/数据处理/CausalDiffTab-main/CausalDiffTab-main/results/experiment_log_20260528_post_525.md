# CausalDiffTab — 实验日志（2026-05-28）
> 相对于 2026-05-25 实验日志的增量更新

---

## 一、新增实验 1：分布漂移量化（2024 → 2025）

为解释 guidance 在迁移后失效的原因，运行 `scripts/quantify_distribution_drift.py`：

| 指标 | 数值 | 说明 |
|------|------|------|
| 数值特征 Wasserstein（归一化） | 0.0228 | 低 covariate shift |
| 类别特征 JS 散度 | 0.0271 | 低分布漂移 |
| Concept drift（目标均值差） | 0.0036 | 几乎无概念漂移 |
| **宏观关系平均绝对差** | **0.2361** | **显著漂移** |
| 宏观关系最大绝对差 | 2.80 | 稀有群体严重漂移 |

**关键发现**：2024→2025 的整体 covariate shift 和 concept drift 都很小，但**group-wise injury mean 的宏观关系漂移显著**（mean=0.24, max=2.8）。这直接解释了为何使用 2024 预计算 group means 的 guidance 会在 2025 上产生 over-correction。

---

## 二、新增实验 2：Robust Guidance 模式对比

为解决 absolute guidance 的 over-correction 问题，在 `unified_ctime_diffusion.py` 和 `sample_conditional.py` 中实现了 4 种 guidance 模式：

### 2.1 实现改动

- **新增参数**：`macro_guidance_mode`（absolute/relative/adaptive/annealed）、`macro_guidance_adaptive_drift_threshold`
- **absolute**（原有）：`offset = (target - current) * scale`
- **relative**：`offset = tanh((target - current) / |current|) * scale`
- **adaptive**：`scale = base_scale * exp(-drift / threshold)`，drift 越大惩罚越大
- **annealed**：`scale = base_scale * (1 - progress)`，线性衰减到 0

### 2.2 迁移（2025）评测结果

| Mode | Crash | Road | Vehicle | Weather | **macroMAE** | vs Baseline |
|------|-------|------|---------|---------|-------------|-------------|
| **no_guidance** | 0.0301 | 0.0310 | 0.0300 | 0.0296 | **0.0302** | — ✅ |
| annealed_g0.5 | 0.0316 | 0.0327 | 0.0319 | 0.0342 | 0.0326 | +8.0% |
| adaptive_g0.5 | 0.0349 | 0.0367 | 0.0345 | 0.0346 | 0.0352 | +16.5% |
| relative_g0.5 | 0.0366 | 0.0383 | 0.0375 | 0.0373 | 0.0374 | +24.0% |
| absolute_g0.5 | 0.0548 | 0.0565 | 0.0570 | 0.0546 | 0.0557 | +84.7% ❌ |

### 2.3 结论

1. **annealed 是最佳 guidance 模式**，但仅比 no-guideline 差 8%，无法超越 baseline。
2. **adaptive 的指数衰减有效**，将 absolute 的 +84.7% 降至 +16.5%。
3. **relative 的 tanh 饱和不够**，仍产生 +24.0% 的退化。
4. **核心原因**：任何基于 2024 group means 的 guidance 都会因 2025 的宏观关系漂移而引入误差。因果结构训练本身已足够鲁棒，无需 inference-time guidance。

---

## 三、新增实验 3：2025 Fine-Tuning

### 3.1 实验设置

- **Source**：`macro_sparse_anneal_v2`（2024 训练）
- **Target**：`data/nyc_crash_2025_v2` 训练集
- **配置**：500 epochs，LR=1e-4，batch=4096，CosineAnnealingLR，EMA decay=0.999
- **输出**：`ckpt/nyc_crash_2024_v2/stage3_full_full_macro_sparse_anneal_v2_ft2025`

### 3.2 Fine-Tuning 过程

| Epoch | Loss | LR |
|-------|------|-----|
| 1 | 2.0740 | 9.98e-05 |
| 50 | 2.0572 | 6.58e-05 |
| 100 | 2.0446 | 1.05e-05 |
| 200 | 2.0224 | 6.58e-05 |
| 300 | 2.0212 | 6.58e-05 |
| 400 | 2.0179 | 1.05e-05 |
| 500 | 2.0298 | 1.00e-04 |
| **Best (epoch 473)** | **2.0049** | — |

Loss 从 2.074 → 2.005，下降约 3.3%。

### 3.3 迁移评测结果

| Model | macroMAE (2025) | vs Zero-Shot |
|-------|----------------|--------------|
| Zero-shot (no guidance) | **0.0302** | — |
| **Fine-tuned (500ep)** | **0.0383** | **+26.8%** ❌ |

Fine-tuned 模型在 2025 上的 macroMAE 反而从 0.0302 恶化到 0.0383。

### 3.4 分析

Fine-tuning 未能提升迁移性能，反而损害了宏观关系保真度。可能原因：

1. **过拟合 2025 训练集 idiosyncrasies**：2025 训练数据本身可能与测试集存在差异，模型过度适应训练分布。
2. **破坏了 2024 学到的良好结构**：低 LR 的 fine-tuning 本应保留大部分预训练知识，但 500 epochs 可能仍足以扭曲因果结构。
3. **2025 数据量或多样性不足**：如果 2025 训练集较小或分布不均，fine-tuning 无法获得足够的泛化信号。

**建议**：尝试更保守的 fine-tuning 设置（如 50-100 epochs、更低 LR 1e-5、冻结部分层），或改用 2025 数据从头训练并与迁移对比。

---

## 四、代码改动汇总

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `tabdiff/models/unified_ctime_diffusion.py` | 新增 | 4 种 guidance 模式（absolute/relative/adaptive/annealed） |
| `src/sample_conditional.py` | 修改 | 暴露 `--macro_guidance_mode` 和 `--macro_guidance_adaptive_drift_threshold` |
| `scripts/quantify_distribution_drift.py` | 新增 | 量化 2024 vs 2025 covariate/concept/macro drift |
| `scripts/finetune_2025.py` | 重写 | 修复导入问题，适配当前 `UnifiedCtimeDiffusion` API，简化训练循环 |
| `_run_adaptive_sampling.py` | 新增 | adaptive guidance 采样 wrapper |
| `_run_relative_sampling.py` | 新增 | relative guidance 采样 wrapper |
| `_run_annealed_sampling.py` | 新增 | annealed guidance 采样 wrapper |
| `_run_finetuned_sampling.py` | 新增 | fine-tuned 模型采样 wrapper |

---

## 五、下一步建议

1. **尝试保守 fine-tuning**：100 epochs，LR=1e-5，或冻结 backbone 只 fine-tune schedule/head。
2. **动态 group means**：在 sampling 时从 2025 验证集实时计算 group means，彻底消除 drift 影响。
3. **injury 二值化 TSTR**：将 ≥1 vs 0 作为二分类重新评估 TSTR，减少类别不平衡对分数的扭曲。
4. **放弃 guidance for transfer**：当前证据强烈表明，对于 2024→2025 迁移，`macro_sparse_anneal_v2` 无 guidance 是最佳配置。
