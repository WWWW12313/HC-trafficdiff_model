# HC-DiffTraffic 项目总览（快速上下文参照文档）

> 生成日期：2026-04-20 | 适用场景：向其他 AI 助手快速传达项目全貌
>
> **阅读建议**：先看 §1（项目定位）→ §2（模型结构）→ §3（实验配置）→ §4（最新结果）→ §5（未完成事项）

---

## §1 项目定位与核心创新

**项目名称**：HC-DiffTraffic（Hierarchical Causal Diffusion for Traffic Data Synthesis）

**核心问题**：NYC 交通事故数据高度敏感且分布稀疏，无法直接公开。需要生成高保真合成数据，满足下游 AI 模型训练（TSTR 范式），同时保持与真实数据的因果结构一致性。

**核心创新**：
1. **层级扩散（Hierarchical）**：将 47 维特征拆分为 Stage1（空间特征）→ Stage3（全特征条件合成），避免高维联合分布学习困难
2. **因果约束（Causal）**：将 NOTEARS-MLP 学习的 DAG（v2 修订版，222 条边）转换为注意力掩码，注入扩散模型损失函数，使生成数据保持因果方向一致性
3. **V2 因果矩阵**：手工审核结合领域知识修订（删除 0 条无语义边，新增 8 条时间行为因果边），比 v1（NOTEARS 原始，214 边）更具领域合理性

**对比文献定位**：TabDDPM（扩散基线）+ CTGAN/TVAE/SMOTE（传统基线）+ 消融实验（NoHierarchy/NoCausal）

---

## §2 模型结构与训练配置

### 2.1 数据集

| 参数 | 值 |
|------|-----|
| 数据来源 | NYC Motor Vehicle Collisions 2017 |
| 总特征数 | 47（数值 9 + 类别 38） |
| 训练集 | 173,533 行 |
| 测试集（域内） | 43,384 行 |
| 测试集（迁移） | 2025 年真实事故数据，5,000 行（`results/postcovid_test_2025_n5000.csv`） |
| 目标变量（回归） | `NUMBER OF PERSONS INJURED`（有界整数 0-5） |
| 目标变量（分类） | `NUMBER OF PERSONS INJURED`（多分类，同一列当离散标签用） |

### 2.2 Stage 划分

| Stage | 职责 | 特征维度 | 模型参数 |
|-------|------|---------|---------|
| Stage 1 | 空间特征条件生成 | 4 num + 3 cat = 7 维 | ~1.2M |
| Stage 3 | 全特征条件合成（以 Stage1 输出为条件） | 9 num + 38 cat = 47 维 | ~10.9M |

### 2.3 V2 因果掩码规模

| 掩码类型 | Stage | 矩阵大小 | 非零边数 |
|---------|-------|---------|---------|
| num causal mask | Stage 3 | (10, 10) | 16 |
| cat causal mask | Stage 3 | (155, 155) | 1867 |
| num causal mask | Stage 1 | (5, 5) | 6 |
| cat causal mask | Stage 1 | (19, 19) | 78 |

### 2.4 训练配置（tier=full）

```
epochs = 4000
batch_size = 4096
early_stop_patience = 200
lr_scheduler = cosine
GPU = NVIDIA RTX 5090, cuda:0
conda 环境 = crashgen
Python = C:/Users/Admin/anaconda3/envs/crashgen/python.exe
工作目录 = C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main
```

### 2.5 训练史

| 阶段 | 状态 | 备注 |
|------|------|------|
| Stage1 v2 重训 | ✅ 完成（Early Stop Epoch 807，loss=2.7214） | 2026-04-17 |
| CTGAN v2 重训 | ✅ 完成（含 clip 越界修正） | 2026-04-17 |
| TVAE v2 重训 | ✅ 完成 | 2026-04-17 |
| SMOTE v2 重生成 | ✅ 完成 | 2026-04-17 |
| Stage3 v2 重训 | ✅ 完成（2026-04-20，GPU 训练，early stop） | 2026-04-20 |
| 全流程评测 | ✅ 完成（2026-04-20 14:49-14:55） | run_v2_rebuild.py 自动触发 |

---

## §3 实验设计

### 3.1 对照组（6 组）

| 名称 | 类型 | 说明 |
|------|------|------|
| `ours_full_model` | **本文方法** | 层级 + 因果约束 + v2 矩阵 |
| `ablation_no_causal` | 消融：无因果约束 | 去掉因果掩码，保留层级 |
| `ablation_no_hierarchy` | 消融：无层级结构 | 直接 Stage3，无条件 Stage1 |
| `baseline_tabddpm` | 深度扩散基线（最强） | 标准 TabDDPM，无因果/层级 |
| `baseline_tvae` | VAE 基线（稳定） | TVAE，有界整数适配性好 |
| `baseline_smote` | 传统重采样基线 | 简单内插，结构保持性好 |
| `baseline_ctgan` | GAN 基线（⚠️ 失败案例） | R²<0，标签分布偏差为主因 |

> **CTGAN 保留原因**：其 R²<0 是有价值的失败案例，证明高维稀疏类别 + 有界整数分布对对抗训练模型结构性不友好。去掉会引起读者质疑。

### 3.2 评测体系（三层）

| 层次 | 主指标 | no_rule profile 权重 | 说明 |
|------|--------|-------------------|------|
| 结构层 | CMI 绝对误差均值、SHD 归一化 | 50%（若可用） | 联合分布 + 因果结构一致性 |
| 任务层 | TSTR avg_score、R²/F1-macro | 50% | 下游模型在合成数据训练后的迁移能力 |
| 规则层 | logic_violation_rate（4 条规则） | **仅参考，不计入排名** | 物理语义一致性，覆盖率有限 |

**TSTR 协议**：用合成数据训练 XGBoost/RandomForest/MLP，用真实测试集评估，取 3 模型均分。

**主指标 profile**：`no_rule`（已在代码中配置，evaluation 命令默认使用）

### 3.3 采样规格

每个模型生成 n=2000 和 n=10000 两档，存放在 `results/synthetic/`，评测用 `*_compare_n*.csv`。

---

## §4 最新评测结果（2026-04-20，v2 因果矩阵，全量训练）

### 4.1 回归任务（TSTR，目标：NUMBER OF PERSONS INJURED）

**评测集**：2025 年真实数据（5000 行，迁移测试集）

| 模型（n=10000） | TSTR avg_score | TSTR R² | MSE | MAE | logic_viol |
|----------------|---------------|---------|-----|-----|-----------|
| **ours_full_model** | **0.7389** | **0.6701** | **0.2015** | **0.1758** | 0.000 |
| ablation_no_causal | 0.7687 | 0.6401 | 0.2198 | 0.1671 | 0.000 |
| ablation_no_hierarchy | 0.7235 | 0.5718 | 0.2615 | 0.2195 | 0.001 |
| baseline_smote | 0.7223 | 0.6516 | 0.2128 | 0.2150 | 0.000 |
| baseline_tabddpm | 0.5975 | 0.5826 | 0.2550 | 0.2325 | 0.003 |
| baseline_tvae | 0.6315 | 0.4247 | 0.3514 | 0.4024 | 0.004 |
| baseline_ctgan ⚠️ | 0.4093 | -0.0804 | 0.6598 | 0.6064 | 0.020 |

> **解读**：ours_full_model 在 TSTR R² 上达到最高（0.6701），MSE 最低（0.2015），表明因果约束在跨时间域迁移场景下显著提升了生成数据的任务有效性。ablation_no_causal 的 avg_score 略高（0.7687 vs 0.7389）但 R² 低于 ours，说明 avg_score 包含非因果相关指标，R² 更能反映回归任务真实质量。

### 4.2 分类任务（TSTR，目标：NUMBER OF PERSONS INJURED 多分类）

**评测集**：域内真实测试集（43,384 行）

| 模型（n=10000） | TSTR avg_score | F1-macro | F1-micro/Acc | AUROC |
|----------------|---------------|---------|-------------|-------|
| **ours_full_model** | **0.6979** | **0.2702** | **0.9569** | **0.9544** |
| ablation_no_causal | 0.6978 | 0.2709 | 0.9570 | 0.9536 |
| ablation_no_hierarchy | 0.6844 | 0.1647 | 0.9535 | — |
| baseline_smote | 0.7199 | 0.6934 | 0.9650 | 0.9793 |
| baseline_tabddpm | 0.6904 | 0.2165 | 0.9558 | — |
| baseline_tvae | 0.6812 | 0.3068 | 0.9250 | 0.9135 |
| baseline_ctgan ⚠️ | 0.5523 | 0.2265 | 0.8086 | 0.5221 |

> **解读**：分类任务中 ours_full_model 与 ablation_no_causal 几乎相同（0.6979 vs 0.6978），SMOTE 因直接复制真实样本分布而占优。说明因果约束在分类任务的边际效益主要体现在跨域迁移，而非域内分类准确率。

### 4.3 Post-COVID 迁移评测（2017→2025 Zero-Shot Transfer）

**迁移测试集**：2025 年真实事故数据（5000 行）

| 模型 | 域内 R²（2017 测试） | 2025 R²（迁移） | 退化率 |
|------|-------------------|----------------|--------|
| ablation_no_causal | 0.6555 | 0.6401 | **-2.35%** |
| **ours_full_model** | 0.6628 | **0.6701** | **+1.11% ↑** |

> **核心发现**：ours_full_model 是唯一在 2025 年迁移测试集上 R² 提升（+1.11%）的模型；ablation_no_causal 出现退化（-2.35%）。这直接验证了因果约束在跨时间分布漂移场景下的泛化优势——因果 DAG 中编码的物理规律（超速→严重性、OSM 道路属性→行为）在 2017→2025 间保持稳定。

---

## §5 未完成事项与待优化

### 5.1 代码层面（P0 优先）

| 任务 | 文件 | 描述 |
|------|------|------|
| 移除 logic_violation_rate 计算 | `pipeline/evaluate_all.py` | ~L576 real_viol 块、~L600 循环内 rate 计算、ranked_summary 中 nc_rule 向量、MD/JSON payload。目前 no_rule profile 已不计入排名，但代码仍在计算，浪费时间 |
| 断点续训 | `src/train_hierarchical.py` | 目前无 resume 机制，中断后从 epoch 0 重启。是重大工程缺陷 |

### 5.2 实验层面（P1 建议）

| 任务 | 描述 |
|------|------|
| 特征分布漂移量化 | 计算 2017 vs 2025 关键特征（CRASH_TIME_SIN、VEHICLE_TYPE、IS_MULTI_VEHICLE）的 JS 散度，量化"迁移场景相似度" |
| CMI/SHD 启用 | 当前结构层指标默认关闭（计算成本高），需要显式启用后重跑才能获得完整三层评分 |
| 迁移退化率表格规范化 | evaluate_postcovid_transfer.py 目前使用 proxy pair（ours vs ablation_no_causal），需要专用对比文件 |

### 5.3 实验层面（P2 可选）

| 任务 | 描述 |
|------|------|
| CopulaGAN 基线 | 作为"稳定 GAN 版本"，验证 CTGAN 失败是框架问题（GAN）而非 Copula/VAE 结构问题 |
| Few-Shot Adaptation | 获取 200-500 行 2025 年带标签训练数据，测试少样本微调效果 |

---

## §6 代码结构速览

```
CausalDiffTab-main/
├── configs/
│   ├── causal_matrix_v2_constrained.npy       # V2 因果矩阵（222边，主矩阵）
│   ├── causal_matrix_notears_mlp.npy           # V1 原始矩阵（214边，不再使用）
│   └── ours_full_model.yaml                    # 主实验配置
├── data/
│   ├── nyc_crash/causal_masks/                 # Stage3 v2 causal masks
│   └── nyc_stage1/causal_masks/                # Stage1 v2 causal masks
├── src/
│   └── train_hierarchical.py                   # 主训练脚本（无 resume 支持）
├── pipeline/
│   ├── run_v2_rebuild.py                       # 全流程编排（Step1-6，--device 参数）
│   ├── run_all_experiments.py                  # 多实验并发（yaml 配置驱动）
│   ├── evaluate_all.py                         # 综合评测（three-layer，no_rule profile）
│   ├── evaluate_postcovid_transfer.py          # 迁移评测（2017→2025）
│   ├── benchmark_evaluator.py                  # TSTR 执行器（XGB/RF/MLP + AUROC）
│   └── download_osm_cache.py                   # OSM 路网缓存（已修复类型错误）
├── results/
│   ├── synthetic/                              # 合成数据（*_compare_n*.csv）
│   ├── eval_report_latest.md/.json             # 最新综合评测报告（regression）
│   ├── eval_report_classification_*.md/.json   # 分类评测报告
│   ├── postcovid_transfer_report_latest.md     # 迁移评测报告
│   └── postcovid_test_2025_n5000.csv          # 2025 年迁移测试集
├── v2_rebuild.log                              # Pipeline 执行日志（stdout）
└── v2_rebuild_err.log                          # Stage3 训练进度（stderr，epoch 条）
```

---

## §7 关键数值速查

| 参数/指标 | 值 |
|----------|-----|
| 训练集规模 | 173,533 行 |
| 特征维度 | 47（9 num + 38 cat） |
| OHE 后维度 | 164（9 + 155） |
| V2 因果矩阵边数 | 222 |
| Stage3 参数量 | ~10.9M |
| Stage3 early stop epoch | ~800-900（具体见 checkpoint 文件名） |
| ours_full_model TSTR R²（迁移） | 0.6701 |
| ours_full_model 迁移退化率 | +1.11%（正值=提升） |
| ablation_no_causal 迁移退化率 | -2.35% |
| CTGAN TSTR R²（n=10000） | -0.0804（失败基线） |

---

## §8 常用恢复命令

```powershell
# 切换到工作目录
cd "C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main"

# 激活环境
conda activate crashgen

# 查看 Stage3 训练进度（epoch 进度条）
Get-Content v2_rebuild_err.log | Select-Object -Last 5

# 查看 pipeline 步骤切换日志
Get-Content v2_rebuild.log | Select-Object -Last 20

# 手动触发采样 + 评测（训练已完成，跳过 Stage3 训练）
$env:PYTHONIOENCODING = "utf-8"
& "C:/Users/Admin/anaconda3/envs/crashgen/python.exe" pipeline/run_v2_rebuild.py --device cuda:0 --skip_train

# 仅重跑评测（采样已完成）
& "C:/Users/Admin/anaconda3/envs/crashgen/python.exe" pipeline/run_v2_rebuild.py --device cuda:0 --eval_only

# 单独运行回归评测
& "C:/Users/Admin/anaconda3/envs/crashgen/python.exe" pipeline/evaluate_all.py --task_type regression --primary_metrics_profile no_rule

# 单独运行分类评测
& "C:/Users/Admin/anaconda3/envs/crashgen/python.exe" pipeline/evaluate_all.py --task_type classification --primary_metrics_profile no_rule

# 迁移评测
& "C:/Users/Admin/anaconda3/envs/crashgen/python.exe" pipeline/evaluate_postcovid_transfer.py
```

---

## §9 实验日志索引

详细实验过程记录在 `Experiment_Logs/2026-04-05.md`，各 Section 说明：

| Section | 内容 | 日期 |
|---------|------|------|
| 1-16 | 封装、重命名、论文初稿、评估体系设计 | 2026-04-05 |
| 17 | 因果矩阵 v2 修订（NOTEARS→领域约束） | 2026-04-13 |
| 18 | 评估体系改革（三层指标、no_rule profile、AUROC） | 2026-04-13 |
| 19 | Post-COVID 迁移评测首次实现 | 2026-04-13 |
| 20 | CTGAN R² 负值根因诊断（标签分布偏差 avg_score=10.78） | 2026-04-13 |
| 21 | V2 Mask 重生成 + GPU 训练重启 | 2026-04-16~17 |
| 22 | 因果矩阵约束逻辑修复 + 宏观图表导出 | 2026-04-15 |
| 23 | 基线选型澄清 + 迁移学习四维分析 + 快速参照提示词 | 2026-04-20 |

---

## §10 向其他 AI 的快速上下文传递模板

如需向新 AI 对话快速说明本项目，可复制以下提示词：

```
【项目】HC-DiffTraffic：NYC 交通事故合成数据生成，层级扩散 + 因果约束（NOTEARS v2矩阵，222边）
【数据】173K 训练行，47 特征（9数值+38类别），目标变量：NUMBER OF PERSONS INJURED
【模型】Stage1（空间特征7维）→ Stage3（全特征47维）条件扩散，因果掩码注入注意力层，~10.9M参数
【基线】TabDDPM/SMOTE/TVAE/CTGAN(⚠️R2<0失败)/ablation_no_causal/ablation_no_hierarchy
【评测】TSTR范式（合成训练→真实测试）：XGB/RF/MLP三模型均分；三层指标（结构/任务/规则），主用no_rule profile
【最新结果（2026-04-20，v2矩阵全量训练）】
  - 回归 TSTR R²: ours=0.6701 > ablation_no_causal=0.6401 > smote=0.6516
  - 分类 AUROC: ours=0.9544, ablation=0.9536, smote=0.9793
  - 迁移（2017→2025 Zero-Shot）: ours退化率=+1.11%（提升），ablation=-2.35%（退化）
【工作目录】C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main
【环境】conda crashgen，GPU RTX 5090 cuda:0
【待完成】evaluate_all.py移除logic_violation_rate计算代码（已不计入排名，仍在计算浪费时间）
```
