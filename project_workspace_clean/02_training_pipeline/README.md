# 训练链路说明

> 主入口脚本：`train_hierarchical.py`  
> 当前主线实验 ID：`macro_soft_2024`  
> 训练环境：conda 环境 `crashgen`，GPU RTX 5090，cuda:0

---

## 分层生成架构

```
Stage 1：时空锚点生成
  输入：无条件
  输出：LATITUDE / LONGITUDE / 时间宏观变量（SEASON / IS_WEEKEND / IS_AM_PEAK / IS_PM_PEAK）
  模型参数量：~1.2M

      ↓ Stage1 输出作为条件

Stage 3：事故微观变量生成
  输入：Stage1 输出（时空锚点）
  输出：车辆类型 / 事故类型 / 天气 / 道路特征 / 伤亡结果
  模型参数量：~10.9M
  因果约束：macro soft mask，λ=0.3
```

> Stage 2（天气/OSM 上下文）目前作为辅助阶段，不在主线训练中强制启用。

---

## 关键训练命令

### 主线训练（Stage 1）

```bash
python src/train_hierarchical.py \
  --stage 1 \
  --tier full \
  --device cuda:0 \
  --no_wandb \
  --lambda_causal 0.3 \
  --experiment_id macro_soft_2024
```

### 主线训练（Stage 3）

```bash
python src/train_hierarchical.py \
  --stage 3 \
  --tier full \
  --device cuda:0 \
  --no_wandb \
  --lambda_causal 0.3 \
  --experiment_id macro_soft_2024
```

### 训练参数说明

| 参数 | 当前主线值 | 说明 |
|------|-----------|------|
| `--stage` | 1 或 3 | Stage 2 暂不在主线中 |
| `--tier` | full | 使用完整训练集 |
| `--lambda_causal` | 0.3 | 软因果正则强度（弱正则，容忍分布漂移） |
| `--experiment_id` | macro_soft_2024 | 与 configs/experiments/ 中的 YAML 对应 |
| `--no_wandb` | 是 | 不使用 WandB 日志（使用本地日志） |
| `--semantic_heads` | **不启用** | 已验证损害 TSTR，不在主线（见 2026-04-29 日志） |

---

## Checkpoint 位置

```
CausalDiffTab-main/ckpt/nyc_crash/
  stage1_full_full_macro_soft_2024/   ← Stage1 主线 checkpoint
  stage3_full_full_macro_soft_2024/   ← Stage3 主线 checkpoint
  stage1_full_full_ours_stage2_causal/  ← 历史 checkpoint（消融参考）
  stage3_full_full_ours_stage2_causal/
```

> ⚠️ Checkpoint 不进入 GitHub，路径参照见 `snapshots/LARGE_FILES_REFERENCE.md`

---

## 各训练相关脚本职责

| 脚本 | 职责 |
|------|------|
| `train_hierarchical.py` | **主训练入口**；Stage1/Stage3 分层扩散模型 |
| `stage2_offline_context.py` | Stage2 离线上下文补全（天气/OSM），非扩散 |
| `context_lookup.py` | Stage2 时空上下文查找辅助 |
| `data.py` | 数据加载器（.npy → DataLoader） |
| `util.py` | 通用工具（归一化、seed、etc.） |
| `env.py` | 环境路径常量 |
| `utils_train.py` | 训练循环辅助函数 |
| `main.py` | 旧版入口（兼容性保留，当前主线不直接调用） |

---

## 训练历史摘要

| 阶段 | 实验 ID | 时间 | 状态 | 备注 |
|------|---------|------|------|------|
| Stage1 v2 重训 | ours_stage2_causal | 2026-04-17 | ✅ | Early Stop Epoch 807, loss=2.7214 |
| Stage3 v2 重训 | ours_stage2_causal | 2026-04-20 | ✅ | GPU 训练，early stop |
| Stage3 macro_soft | macro_soft_2024 | 2026-04-26+ | ✅ | λ=0.3，当前主线 |
| Stage3 semantic_vc | semantic_heads_vc_2024 | 2026-04-29 | 🗄️ 存档 | TSTR 不如主线，已弃用 |

---

## 注意事项

1. **schema 变更后必须重训**：`is_other_vehicle` 引入（2026-04-29）改变了 cat schema，旧 checkpoint 不可复用。
2. **WandB 替代**：使用 `swanlab`（本地日志）替代 WandB，无需外网。
3. **Early Stopping**：patience=200，模型训练通常在 800-1500 epoch 内收敛。
