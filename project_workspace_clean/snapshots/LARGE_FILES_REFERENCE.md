# 大文件路径参照

> 本文件不含实际数据，仅记录原始项目中大文件的路径和规格，供后续管理参考。  
> 这些文件不进入 GitHub，需要在工作站本地保留。

---

## 训练数据（.npy）

**位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\data\`

| 数据集目录 | 内容 | 估计大小 |
|-----------|------|---------|
| `data/nyc_crash/` | 2024 训练集（当前主线）X_num/X_cat/y × train/val/test | ~200MB |
| `data/nyc_crash_2025/` | 2025 迁移测试集 | ~50MB |
| `data/nyc_stage1_2024/` | Stage1 单独数据集 | ~30MB |
| `data/nyc_stage2_2024/` | Stage2 单独数据集 | ~30MB |

**位置（旧版）**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\tab-ddpm-main\data\`

| 数据集目录 | 内容 | 估计大小 |
|-----------|------|---------|
| `data/nyc_crash_v3/` | 旧版 2017 训练集 | ~55MB |
| `data/nyc_crash_c4/` | 旧版大规模版本 | ~360MB+ |
| `data/nyc_crash_v7/` | v7 版本 | ~100MB |
| `data/nyc_crash_v8/` | v8 版本 | ~100MB |

---

## Checkpoint 文件

**位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\ckpt\nyc_crash\`

| Checkpoint 目录 | 对应实验 | 状态 |
|----------------|---------|------|
| `stage1_full_full_macro_soft_2024/` | Stage1 主线 | ✅ 当前主线 |
| `stage3_full_full_macro_soft_2024/` | Stage3 主线 | ✅ 当前主线 |
| `stage1_full_full_ours_stage2_causal/` | 历史 Stage1 | 参考 |
| `stage3_full_full_ours_stage2_causal/` | 历史 Stage3 | 参考 |
| `stage3_full_full_semantic_heads_vc_2024/` | 消融（语义 CE）| 已弃用 |

---

## 合成结果文件

**位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\results\`

| 文件名模式 | 内容 |
|-----------|------|
| `synth_macro_soft_2024_n10000.csv` | 主线采样结果（10000条） |
| `results/synthetic/macro_soft_2024_*_compare_n10000.csv` | 主线评测输入 |
| `synth_macro_soft_2024_n10000_semantic_repair_*.csv` | 语义修复后版本（已弃用） |

---

## 原始大型 CSV

**位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\`

| 文件 | 大小估计 | 说明 |
|------|---------|------|
| `nyc_accidents_2017.csv` | ~200MB | 原始 NYC 事故数据 2017 |
| `nyc_accidents_2018.csv` | ~200MB | 原始 NYC 事故数据 2018 |
| `nyc_2017_final_v8.csv` | ~100MB | 富化后 2017 |
| `nyc_2017_pristine_v9.csv` | ~100MB | v9 版本 |
| `CausalDiffTab-main/.../raw_data/` | — | 原始下载目录 |

---

## 因果矩阵 .npy 文件

**位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\configs\`

| 文件 | 说明 | 是否在 GitHub |
|------|------|-------------|
| `causal_matrix_macro_soft.npy` | 当前主线因果矩阵 | ❌（.npy 排除）|
| `causal_matrix_macro_soft.csv` | 同上，CSV 版 | ✅ |
| `causal_matrix_v2_constrained.npy` | 旧版 222 边 | ❌ |
| `causal_matrix_v2_constrained.csv` | 同上，CSV 版 | ✅ |

> `.npy` 因果矩阵文件在本工作站有完整版本，`.csv` 版本已同步到 GitHub，可从 CSV 重新加载。
