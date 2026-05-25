# 实验配置说明

> 配置目录：`experiments/`（对应原项目 `configs/experiments/`）

---

## 配置文件总览

| 配置文件 | 实验 ID | 状态 | 说明 |
|---------|---------|------|------|
| `experiments/macro_soft_2024.yaml` | macro_soft_2024 | ✅ **当前主线** | 宏观软 DAG + 分层，λ=0.3，2024→2025 |
| `experiments/our_model_no_h3.yaml` | ours_stage2_causal | 🔵 消融基准 | 无 H3，历史版本，用作消融对照 |
| `experiments/ablation_no_causal.yaml` | — | 🔵 消融 | 去除因果约束（λ=0）|
| `experiments/ablation_no_hierarchy.yaml` | — | 🔵 消融 | 去除分层（Stage1+Stage3 合并为单级）|
| `experiments/baseline_tabddpm.yaml` | — | 🔵 基线 | 原版 TabDDPM，无因果约束、无分层 |
| `experiments/ours_full_model.yaml` | — | ⚠️ 旧版 | 2017 域内全年版本，已归档 |
| `experiments/our_model.yaml` | — | ⚠️ 旧版 | 早期版本，已归档 |
| `experiments/ours_stage2_causal.yaml` | — | ⚠️ 旧版 | Stage2 实验版，已被 macro_soft_2024 取代 |

---

## 当前主线配置详解

```yaml
# macro_soft_2024.yaml
model_name: macro_soft_2024
experiment_id: macro_soft_2024
lambda_causal: 0.3          # 弱正则强度（容忍分布漂移）
use_causal_masks: true       # 启用因果 mask
hierarchical: true           # 启用分层生成（Stage1+Stage3）
description: 宏观软因果骨架 + 分层生成，lambda=0.3，2024训练/2025迁移
sampling:
  mode: unconditional        # 无条件采样（非条件修复）
  num_samples: 10000
```

---

## 消融实验配置

```yaml
# ablation_no_causal.yaml
lambda_causal: 0.0
use_causal_masks: false
hierarchical: true

# ablation_no_hierarchy.yaml
lambda_causal: 0.3
use_causal_masks: true
hierarchical: false

# baseline_tabddpm.yaml
lambda_causal: 0.0
use_causal_masks: false
hierarchical: false
```

---

## 添加新实验配置

在 `experiments/` 目录下新建 YAML 文件，字段参照 `macro_soft_2024.yaml`。  
`experiment_id` 字段决定 checkpoint 目录名和结果文件名，必须全局唯一。

---

## CausalDiffTab 框架级配置

`CausalDiffTab.yaml`：框架全局默认配置（数据路径、模型超参等），实验配置中的字段会覆盖此默认值。
