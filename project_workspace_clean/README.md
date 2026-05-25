# HC-DiffTraffic — 项目整理工作区

> **整理日期**：2026-05-16  
> **原始项目位置**：`C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\`  
> **本目录性质**：只读结构镜像，不替代原始项目，不含大文件（.npy / checkpoint / 大结果）  
> **当前正式主线**：2024 → 2025 迁移学习 · 无 H3-cell · 简化 DAG（macro soft mask）  

---

## 目录结构说明

```
project_workspace_clean/
  README.md                  ← 本文件：项目总览与导航
  00_project_overview/       ← 项目定位、正式主线声明、字段规范
  01_data_pipeline/          ← 数据链路：清洗 / 特征构造 / 2024-2025 准备
  02_training_pipeline/      ← 训练链路：分层扩散 / Stage 划分 / checkpoint 管理
  03_evaluation_pipeline/    ← 评测链路：域内 / 迁移 / 结构化指标
  04_causal_graph/           ← 因果图链路：NOTEARS 发现 / 人工修正 / soft mask
  05_experiment_configs/     ← 实验配置：所有 YAML / 消融开关
  06_logs_and_reports/       ← 实验日志体系（完整保留）+ 日志索引
  07_sync_to_github/         ← GitHub 同步方案 / .gitignore 建议
  scripts/                   ← 一次性诊断 / 消融辅助脚本
  snapshots/                 ← 大文件路径参照（不含实际数据）
```

---

## 快速导航

| 我想了解… | 去哪里看 |
|-----------|----------|
| 项目整体定位和创新点 | `00_project_overview/PROJECT_OVERVIEW.md` |
| 当前正式主线（字段、DAG、mask 策略） | `00_project_overview/MAINLINE.md` |
| 数据怎么从原始 CSV 到训练集 | `01_data_pipeline/README.md` |
| 模型怎么训练、Stage 怎么划分 | `02_training_pipeline/README.md` |
| 评测指标体系和怎么跑评测 | `03_evaluation_pipeline/README.md` |
| NOTEARS 怎么用 / soft mask 怎么生成 | `04_causal_graph/README.md` |
| 哪个 YAML 对应当前主线 | `05_experiment_configs/README.md` |
| 哪些日志属于当前主线 | `06_logs_and_reports/LOG_INDEX.md` |
| 如何同步到 GitHub | `07_sync_to_github/GITHUB_SYNC_GUIDE.md` |

---

## 当前正式主线一句话摘要

> 用 2024 年 NYC 事故数据训练分层扩散模型（Stage1 时空锚点 → Stage3 事故微观变量），  
> 以宏观软因果 DAG（7组约束，λ=0.3）作为结构正则，  
> 在 2025 年数据上做迁移测试，  
> 空间使用经纬度，不使用 H3-cell，  
> 评价重心在 **跨年份迁移稳健性** 和 **结构真实性**，不以 R² 为主指标。

---

## 原始项目位置（只读参考）

| 子项目 | 路径 |
|--------|------|
| CausalDiffTab 主模型 | `课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\` |
| TabDDPM 基线 / 实验日志 | `课题学习\数据处理\tab-ddpm-main\` |
| 原始数据集 | `课题学习\数据处理\原始数据集\` |
| .npy 训练数据 | `课题学习\数据处理\tab-ddpm-main\data\` |
| Checkpoint | `课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\ckpt\` |
| 采样结果 | `课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\results\` |
