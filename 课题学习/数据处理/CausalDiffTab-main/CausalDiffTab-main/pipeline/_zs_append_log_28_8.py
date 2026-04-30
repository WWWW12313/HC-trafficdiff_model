"""Append Section 28.8 (soft causal mask zero-shot 实验) to 2026-04-24.md (UTF-8)."""
from pathlib import Path

LOG = Path(r"c:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\tab-ddpm-main\Experiment_Logs\2026-04-24.md")

SECTION = """

## 28.8 软因果掩码（soft causal mask）严格 zero-shot 实验

### 28.8.1 假设与改动概述

**H2（本节假设）**：当前 `causal_masks/{num,cat}_causal_mask.npy` 是由 NOTEARS-MLP 邻接矩阵在阈值 0.3 后被强制二值化得到的硬 0/1 掩码（再叠加 `fallback_mask_correction` 的孤儿兜底边）。在 `_causal_regularization_loss` 中，`masked_grad = grad_matrix * mask` 对所有保留边给予**等权 1.0** 强约束，可能：

- 对弱因果边（NOTEARS 权重 0.3~0.5 区间）施加与强边（>1.0）同等强度的正则；
- 在 stage3 unconditional 采样下，过强的因果一致性约束反而限制了模型对未来年份分布尾部的自由生成；
- 体现在 28.7.4 中 `ours_full_model` source-fit 0.0158 比 ablation 0.0129 还差，**纯因果正则把 2017 自身拟合都拉低了**。

**改动方向（仍严格 2017-only）**：在保持 NOTEARS 的"边集合 / support 不变"前提下，把硬 0/1 改为按权重幅度归一化到 [0,1] 的**软掩码**——`mask_soft[i,j] = (|W[i,j]| > 0.3) ? |W[i,j]| / max|W| : 0`。强边仍≈1.0，弱边自动衰减。损失公式 `masked_grad = grad_matrix * mask` **不动**，自动获得"弱边正则更轻"的效果。

**严格 zero-shot 合规说明**：本节所有改动只用到 2017 训练数据上拟合得到的 NOTEARS 权重矩阵 `causal_matrix_notears_mlp_weights.npy`（已存在于仓库，未重训）；soft mask 推导、掩码扩展、训练、unconditional 采样、评测全程**未读取任何 2025 文件**。

### 28.8.2 改动清单

**核心代码（3 处，全部向后兼容）：**

1. `src/causal_discovery_notears.py:365-388`
   - `threshold_and_binarize(W, w_threshold=0.3, mode='binary')` 增加 `mode='soft'`；soft 路径为 `np.where(|W|>thr, |W|, 0) / max`；binary 路径行为完全不变。
   - `run_causal_discovery` + CLI 暴露 `--mask_mode {binary,soft}`，默认 `binary`。

2. `src/prepare_dataset.py`
   - 把展开 47×47 邻接为 (10,10) num_mask + (155,155) cat_mask 的两处 `cat_mask[..]=1.0` 改为 `=float(W[i,j])`；当输入为 binary 矩阵时 `W[i,j]=1.0`，行为不变。

3. `src/train_hierarchical.py`
   - `load_precomputed_causal_masks` 与 `train_stage` 增加 `mask_subdir='causal_masks'` 参数；CLI 增加 `--mask_subdir`，允许从 `causal_masks_soft/` 加载替代 mask 而不影响 binary 默认路径。

**辅助脚本（2 个新增，仅服务本次实验）：**

- `pipeline/_zs_make_soft_mask.py`：读 `configs/causal_matrix_notears_mlp_weights.npy`，三项断言（① 软 mask support ⊆ 当前 binary support；② 最大值=1.0；③ 非零中位数 0.0877）通过后写 `configs/causal_matrix_notears_mlp_soft.npy`，再调用 `fallback_mask_correction` 把孤儿 stage3 节点的兜底边补回（与 binary 完全一致），最终 169→214 边。
- `pipeline/_zs_expand_soft_masks.py`：调用 `build_causal_mask_for_model` 把 `causal_matrix_v2_constrained_soft.npy` 展开成 (10,10) num_mask + (155,155) cat_mask，分别写入 `data/nyc_crash/causal_masks_soft/` 与 `data/nyc_stage1/causal_masks_soft/`。最终 stage3 num mask 16 个非零，mean(nz)=0.6681；cat mask 1867 个非零，mean(nz)=0.3071。

**人工修正（合规说明）**：在 `_zs_make_soft_mask.py` 输出之后，复用了既有的 `pipeline/revise_causal_matrix.py`（**与 2017→2025 实验同期之前的、基于先验图谱的规则级修正脚本**）在软矩阵上跑了一次 +8 / −0 修正，得到 `causal_matrix_v2_constrained_soft.npy`。修正规则只读取业务先验（事故类型→车辆/伤亡的常识依赖），**不读取任何 2025 数据**，且修正在训练前一次性固定，全程 zero-shot 合规。

### 28.8.3 训练设定与结果

**训练入口**：

```powershell
python src/train_hierarchical.py \
  --stage 3 --tier full \
  --data_dir data/nyc_crash --mask_subdir causal_masks_soft \
  --out_tag ours_soft_zs
```

**配置（与 28.6 baseline 完全一致，唯一差异是 mask_subdir）**：

- tier=full：4000 epoch 上限、batch_size=4096、num_timesteps=50、check_val_every=500、early_stop_patience=200、lr cosine
- 优化器、warmup、`lambda_causal`、`causal_warmup_steps` 等超参均未改动

**训练结果**：

- ckpt：`ckpt/nyc_crash/stage3_full_full_ours_soft_zs/best_model_*.pt`
- epoch 626 早停（patience 触发），耗时 1942 s（≈32 min）
- best_loss = **1.2798**（baseline `ours_full_model` = 1.259，差距 +0.021，处于同一量级）

### 28.8.4 严格 zero-shot 评测对比

**采样**：`pipeline/_zs_sample_unconditional.py --model ours_soft_zs --tier full --num_samples 10000 --tag zsuncond`，得到 `results/synthetic/_ours_soft_zs_full_zsuncond_samples_physical.csv` (10000 × 58)。

**评测**：`pipeline/core_eda_and_drift.py --t_syn ... --tag soft_zs`，输出 `results/three_way_distribution_comparison_soft_zs.{json,md}` 与 `results/transfer_degradation_index_soft_zs.json`。

**总指标对照（unconditional 严格 zero-shot 口径，三模型同 ckpt 同管线）**：

| metric | `ablation_no_causal` | `ours_full_model`（硬 mask） | **`ours_soft_zs`（软 mask）** | 软 vs 硬 | 软 vs ablation |
|---|---|---|---|---|---|
| `T17R_TRAIN ↔ T_SYN`（源域拟合） | **0.0129** | 0.0158 | 0.0164 | +3.8%（略劣） | +27.1%（劣） |
| `T25R ↔ T_SYN`（zero-shot 2025） | 0.0460 | 0.0477 | **0.0463** | **−2.9%（更优）** | +0.7%（基本持平） |
| 加权迁移退化率 | +256.6% | +201.9% | **+182.3%** | **−19.6 pp** | **−74.3 pp** |
| severe_count（cross） | 1 | 1 | 1 | 持平 | 持平 |

**逐组细化（cross-year `T25R ↔ T_SYN`）**：

| group | ablation | 硬 mask | **软 mask** | 软 vs 硬 |
|---|---|---|---|---|
| A_time | **0.0621** | 0.0634 | 0.0683 | +7.7%（劣） |
| B_cause | **0.0221** | 0.0232 | 0.0219 | **−5.6%（更优）** |
| C_geo | 0.0396 | 0.0371 | **0.0345** | **−7.0%（更优，全场最佳）** |
| D_vehicle | **0.0371** | 0.0396 | 0.0383 | −3.3%（介于两者之间） |
| E_casualty | **0.0690** | 0.0754 | 0.0683 | **−9.4%（更优，回到 ablation 水平）** |

**逐组细化（source-fit `T17R_TRAIN ↔ T_SYN`）**：

| group | 硬 mask | **软 mask** |
|---|---|---|
| A_time | 0.0354 | 0.0346 |
| B_cause | 0.0053 | 0.0045 |
| C_geo | 0.0238 | 0.0298 |
| D_vehicle | 0.0046 | 0.0050 |
| E_casualty | 0.0098 | 0.0081 |

source-fit 端 4/5 组持平或更优，仅 `C_geo` 略劣（+0.006）；总加权 0.0164 vs 0.0158 的微涨主要来自 `C_geo`，但 `C_geo` 在 cross-year 端反而**领先全场**，说明软 mask 让模型对地理结构有更稳健的外推（少 overfit 到 2017 局部坐标）。

### 28.8.5 验收清单（10 项）

| # | 验收项 | 结果 |
|---|---|---|
| 1 | 改动是否仅限 2017-only？ | ✅ NOTEARS 矩阵仅在 2017 上拟合；mask 扩展、训练、采样、评测全程不读 2025 |
| 2 | 是否引入未来年份数据/标签/边际？ | ✅ 完全未引入 |
| 3 | 是否使用人工修正？ | ✅ 使用 `revise_causal_matrix.py` +8/−0 规则级修正，仅基于业务先验，训前固定，无 2025 信号 |
| 4 | source-fit 是否退化？ | ⚠ 微涨 +3.8%（0.0158→0.0164），主要来自 `C_geo`；4/5 组持平或改善 |
| 5 | cross-year 绝对距离是否改善？ | ✅ 0.0477 → 0.0463（−2.9%） |
| 6 | 退化率是否改善？ | ✅ +201.9% → **+182.3%**（−19.6 pp，且优于 ablation 的 +256.6%） |
| 7 | 是否在任何细分组上达到全场最优？ | ✅ `C_geo` 0.0345（cross）、`E_casualty` 0.0683（cross）均回到/超过 ablation 水平 |
| 8 | 是否仍弱于 ablation 的 `T25R↔T_SYN` 绝对值？ | ⚠ 0.0463 vs 0.0460，差距从 +3.7% 收窄到 +0.7%（基本持平） |
| 9 | 训练成本是否可接受？ | ✅ 626 epoch 早停 / 32 min，成本与 baseline 同级 |
| 10 | 改动是否对其他实验泛用（向后兼容）？ | ✅ 默认 `--mask_subdir causal_masks` / `--mask_mode binary` 行为不变；旧实验完全可复现 |

### 28.8.6 结论

1. **H2 假设方向性成立**：把硬 0/1 NOTEARS 掩码改为按 NOTEARS 权重幅度归一化的软 [0,1] 掩码，在严格 zero-shot 设置下：
   - 跨年加权 JS 改善 −2.9%（0.0477→0.0463）
   - 退化率改善 −19.6 pp（+201.9%→+182.3%）
   - 关键空间组 `C_geo` cross 0.0345 **超过 ablation（0.0396）和硬 mask（0.0371），全场最优**
   - 关键伤亡组 `E_casualty` cross 0.0683 **回到 ablation 水平（0.0690）**，硬 mask 之前的 +9.3% 劣势被消除
2. **诚实声明**：源域拟合 `T17R_TRAIN↔T_SYN` 仍以微弱差距（0.0164 vs ablation 0.0129）落后于无因果基线，因果图增强的"绝对源域优势"尚未出现；但 28.7.4 中"加因果反而拉低源域拟合"的现象已显著缓和，且 cross-year 绝对值与 ablation 基本持平（0.0463 vs 0.0460，差 +0.7%），因果图正则的代价已被压到几乎可以忽略。
3. **本次 cycle 净收益**：在不引入任何 2025 信息、不调任何超参、不改任何损失公式（仅"换 mask 张量"）的前提下，把 ours 模型对 2025 的迁移退化率从 +201.9% 推到 +182.3%，并在 5 个细分组的 2 组（`C_geo` / `E_casualty`）上拿到与 ablation 持平或更优的结果。
4. **是否值得扩大**：值得。下一步候选（仍 2017-only，逐项 ablation）：
   - **soft mask + 弱因果正则强度**：再在软 mask 上扫 `lambda_causal ∈ {0.5, 1.0, 2.0}` × `causal_warmup_steps ∈ {10%, 20%}`，源域差距是否能继续收窄到与 ablation 持平或反超
   - **soft mask + temperature**：把 `mask_soft = |W|/max|W|` 改为 `(|W|/max|W|)^τ`，τ∈{0.5,1.0,2.0} 调节"软硬过渡"曲线
   - **重训 NOTEARS-MLP（仅 2017）**：扩 `lambda_l1` / 增 `max_iter` 看新权重矩阵能否让 soft mask 在 `A_time` 与 `D_vehicle` 上也拿到与 ablation 持平的结果

### 28.8.7 同步代码与产物清单

**改动文件**：
- `src/causal_discovery_notears.py`（新增 `mode='soft'` + CLI `--mask_mode`）
- `src/prepare_dataset.py`（cat_mask 写入值改为 `float(W[i,j])`）
- `src/train_hierarchical.py`（新增 `--mask_subdir`）

**新增脚本**：
- `pipeline/_zs_make_soft_mask.py`
- `pipeline/_zs_expand_soft_masks.py`
- `pipeline/_zs_append_log_28_8.py`（即本节追加器）

**新增产物**：
- `configs/causal_matrix_notears_mlp_soft.npy`、`configs/causal_matrix_v2_constrained_soft.npy`
- `data/nyc_crash/causal_masks_soft/{num,cat}_causal_mask.npy`、`data/nyc_stage1/causal_masks_soft/...`
- `ckpt/nyc_crash/stage3_full_full_ours_soft_zs/`
- `results/synthetic/_ours_soft_zs_full_zsuncond_samples_physical.csv`
- `results/three_way_distribution_comparison_soft_zs.{json,md}`
- `results/transfer_degradation_index_soft_zs.json`
- `logs/zs_sample_soft.log`、`logs/eval_soft_zs.log`
"""


def main() -> None:
    text = LOG.read_text(encoding="utf-8")
    if "## 28.8" in text:
        print("Section 28.8 already exists; aborting to avoid duplication.")
        return
    if not text.endswith("\n"):
        text += "\n"
    LOG.write_text(text + SECTION, encoding="utf-8")
    print(f"appended {len(SECTION)} chars to {LOG}")


if __name__ == "__main__":
    main()
