# M4 CausalDDPM 实验日志

> 自动生成于 2026-03-09，最后更新于 2026-03-10。记录项目中所有训练运行及其状态。

---

## 一、训练运行记录 (`runs/`)

### Phase 1: 早期探索 (2026-03-04)

| 运行名 | Epochs | Best Loss | 状态 | 说明 |
|--------|--------|-----------|------|------|
| `M4_CausalDDPM_v3_e50_cuda_20260304_172334` | 50 | — | 仅best.pt | 最初测试运行，无 summary |
| `M4_CausalDDPM_v3_e50_cuda_20260304_172448` | 50 | — | 仅best.pt | 同上 |
| `M4_CausalDDPM_v3_e50_cuda_20260304_172550` | 50 | — | 仅best.pt | 同上 |
| `M4_CausalDDPM_v3_e50_cuda_20260304_172655` | 50/50 | 1.1193 | ✅ 完成 | 首个完整运行 |

### Phase 2: 调参迭代 (2026-03-05)

| 运行名 | Epochs | Best Loss | 状态 | 说明 |
|--------|--------|-----------|------|------|
| `M4_CausalDDPM_v3_e50_cuda_20260305_102816` | 50/50 | 1.1208 | ✅ 完成 | 参数微调 |
| `M4_CausalDDPM_v3_e50_cuda_20260305_103725` | 50/50 | 1.1345 | ✅ 完成 | |
| `M4_CausalDDPM_v3_e50_cuda_20260305_111724` | 50/50 | 1.1273 | ✅ 完成 | |
| `M4_CausalDDPM_v3_e50_cuda_20260305_160209` | 50 | — | 仅best.pt | |
| `M4_CausalDDPM_v3_e50_cuda_20260305_160558` | 50/50 | 1.1232 | ✅ 完成 | |
| `M4_CausalDDPM_v3_e50_cuda_20260305_173130` | 50/50 | 1.2831 | ✅ 完成 | 修改特征后 loss 上升 |

### Phase 3: 长训练 & 架构修改 (2026-03-06)

| 运行名 | Epochs | Best Loss | 状态 | 说明 |
|--------|--------|-----------|------|------|
| `M4_CausalDDPM_v3_e50_cuda_20260306_101058` | 50/50 | **1.0613** | ✅ 完成 | 历史最低 loss |
| `M4_CausalDDPM_v3_e150_cuda_20260306_112428` | 150/150 | 0.8653 | ✅ 完成 | 长训练，低 loss 但过拟合 |
| `M4_CausalDDPM_v3_e150_cpu_20260306_112517` | — | — | 仅best.pt | CPU 回退测试 |
| `M4_CausalDDPM_v3_e150_cuda_20260306_112614` | 150/150 | 0.8691 | ✅ 完成 | 长训练复现 |
| `causal_ddpm_experiment` | — | — | ❌ 无ckpt | 早期实验目录（仅含日志/图表） |

### Phase 4: AMP/MaskedLinear 修复 (2026-03-07 上午)

> 遇到 AMP BFloat16 dtype 不匹配问题，以及 USE_MASKED_LINEAR NameError。
> 以下运行均在启动时立即崩溃，目录为空。

| 运行名 | 状态 | 说明 |
|--------|------|------|
| `M4_CausalDDPM_v3_e5_cuda_20260307_102209` | ❌ 空目录 | AMP dtype 崩溃 |
| `M4_CausalDDPM_v3_e20_cuda_20260307_104000` | ❌ 空目录 | AMP 修复迭代 |
| `M4_CausalDDPM_v3_e20_cuda_20260307_104115` | ❌ 空目录 | 同上 |
| `M4_CausalDDPM_v3_e20_cuda_20260307_104545` | ❌ 空目录 | 同上 |
| `M4_CausalDDPM_v3_e20_cuda_20260307_104846` | ❌ 空目录 | 同上 |
| `M4_CausalDDPM_v3_e8_cuda_20260307_105939` | ❌ 空目录 | MaskedLinear NameError |
| `M4_CausalDDPM_v3_e8_cuda_20260307_110016` | ❌ 空目录 | 同上 |
| `M4_CausalDDPM_v3_e8_cuda_20260307_110254` | ❌ 空目录 | 同上 |
| `M4_CausalDDPM_v3_e8_cuda_20260307_110344` | ❌ 空目录 | 同上 |

### Phase 5: 修复后快速训练 (2026-03-07 下午)

> AMP 改为 opt-in，MaskedLinear 改为 opt-in，特征白名单修复。

| 运行名 | Epochs | Best Loss | 状态 | 说明 |
|--------|--------|-----------|------|------|
| `M4_CausalDDPM_v3_e3_cuda_20260307_113234` | 3/3 | 1.6956 | ✅ 完成 | AMP/Masked 修复后首次运行 |
| `M4_CausalDDPM_v3_e3_cuda_20260307_154125` | 3/3 | 1.6414 | ✅ 完成 | 特征白名单+高基数修复 |
| `M4_CausalDDPM_v3_e3_cuda_20260307_160225` | 3/3 | 1.5677 | ✅ 完成 | 移除地理文本列后 |

### Phase 6: v4 训练循环对齐原版 TabDDPM (2026-03-09)

> 发现 v3 训练循环与原版 TabDDPM 存在 9 处关键偏差，全面重写：
> - epoch制 → **steps制**（固定步数迭代）
> - 线性LR退火（lr × (1 - step/total_steps)）
> - EMA 每步更新 (rate=0.999)
> - AdamW betas=(0.9, 0.999)
> - 无梯度裁剪/无梯度累积
> - 无 AMP

| 运行名 | Steps | Best Loss | 状态 | 说明 |
|--------|-------|-----------|------|------|
| `M4_CausalDDPM_v4_s1000_cuda_20260309_111958` | 1000 | 1.23 | ✅ 完成 | v4 quick 首次验证 |
| `M4_CausalDDPM_v4_s3000_cuda_20260309_133006` | 3000 | 0.83 | ✅ 完成 | v4 balanced，loss 显著下降 |

### Phase 7: v5 采样修复 + full 训练 (2026-03-10)

> **关键发现**: 诊断发现反向扩散过程中 z_norm 从 N(0,1) 指数级发散（std=949→25075），
> 根因是 cosine 调度下 alphas_cumprod[999]≈2e-9，使 pred_x0 被放大20000倍。
> 修复方案（3处代码改动）：
> 1. `gaussian_p_mean_variance`: 当 `clip_denoised=True` 时，裁剪 pred_xstart 至 [-6, 6]
> 2. `sample()`: 启用 `clip_denoised=True`（之前为 False）
> 3. `model_variance`: 改用后验方差（更稳定），而非 β_t
>
> 修复后验证：z_norm 最终 std = 0.88~4.88，完全正常。

| 运行名 | Steps | Best Loss | 状态 | 说明 |
|--------|-------|-----------|------|------|
| `M4_CausalDDPM_v4_s5000_cuda_20260309_175604` | 5000 | **0.7880** | ✅ 完成 | **full profile**, d_layers=[768,768,768,768], 历史最佳 |

---

## 二、核心问题发现

### 2.1 生成质量持续坍缩
- **现象**: 生成数据 y 均值≈7.4（真实≈0.26），PRIMARY_CAUSE 99.9% 单一值
- **根因分析 (2026-03-09)**:
  1. **时间特征完全丢失**: `_add_time_features` 生成 14 个时间特征，全部被 `info.json` 白名单过滤
  2. **22 个二元 IS_XXX 标签被当作连续变量**: 高斯扩散无法学习 0/1 分布
  3. **高基数文本变量** (VEHICLE TYPE 40+类, OSM_TYPE 23类) 增加噪声
  4. **冗余特征**: coco/WEATHER_CONDITION/OSM_SPEED_TAG 等重复信息

### 2.2 解决方案 (v2 重构, 2026-03-09)

| 改动 | 之前 | 之后 |
|------|------|------|
| 日期/时间 | 完全丢失（0 个特征参与训练） | 3 个分类特征: IS_WEEKEND, CRASH_SEASON, CRASH_TIME_PERIOD |
| IS_XXX 标签 | float64 → 高斯扩散 | str → **multinomial 扩散** |
| CAUSE_XXX | 仅 5 列 categorical | 同 → multinomial 扩散 |
| VEHICLE TYPE 1/2 | 40+/53 类别 | 合并至 8/9 类别 |
| VEHICLE TYPE 3-5 | 极稀疏，部分保留 | **删除** |
| REAL_WEATHER | 9 类别 | 合并至 5 类别 |
| OSM_TYPE | 23 类别 | 合并至 6 类别 |
| TOTAL_VEHICLES | 连续 | 分类 (1-4+) |
| INFERRED_LANES | 连续 | 分类 (1-4+) |
| 冗余列 | coco, OSM_SPEED_TAG, WEATHER_CONDITION 等 | **删除** |
| 连续特征 | ~44 个 | **7 个** (lat, lon, temp, prcp, wind, dist_signal, speed_limit) |
| 分类特征 | ~8 个 | **40 个** (含 27 个二元) |
| **总 one-hot 维度** | ~80 | ~106 |

---

## 三、基线对照结果

### 3.1 旧特征集 (Phase 5, ~44 连续特征)

| 方法 | Test R² | 备注 |
|------|---------|------|
| SMOTE | 0.1103 | 传统过采样基线 |
| DDPM_MLP (原版) | -170.20 | 原版 TabDDPM pipeline |
| M4 Raw (e3 quick) | -125.40 | 因果扩散，未后处理 |
| M4 + Hurdle | -0.0018 | Hurdle 后处理修正零膨胀 |

### 3.2 新特征集 v2 (7 连续 + 40 分类, 200K 样本)

> 评估方式：Train-on-Synthetic, Test-on-Real (CatBoost)
> 评估日期：2026-03-10

| 方法 | Test RMSE | Test R² | Val RMSE | Val R² | 备注 |
|------|-----------|---------|----------|--------|------|
| **SMOTE** | **0.5993** | **0.1103** | 0.6590 | 0.0939 | 传统过采样基线 |
| DDPM_MLP (原版) | 8.3135 | -170.20 | 8.2975 | -142.67 | 原版 TabDDPM（严重坍缩） |
| M4 v4 s3000 (无fix) Raw | 2.43 | -13.58 | — | — | 采样未裁剪，数值发散 |
| M4 v4 s3000 (无fix) +Hurdle | 0.63 | 0.006 | — | — | Hurdle 修正部分问题 |
| **M4 v5 s5000 Raw** | **0.6013** | **0.1045** | 0.6610 | 0.0882 | **采样修复后，接近 SMOTE** |
| M4 v5 s5000 +Hurdle | 0.6224 | 0.0405 | 0.6806 | 0.0334 | Hurdle 反而降低（模型已学到零膨胀） |

**关键结论**：
- M4 v5 (RMSE=0.6013, R²=0.1045) **几乎追平 SMOTE** (RMSE=0.5993, R²=0.1103)
- 采样修复后 Hurdle 后处理不再需要——模型本身已正确学到目标分布
- 原版 DDPM_MLP 仍然严重坍缩，证明因果结构和修复的必要性

---

## 四、待清理目录

以下目录将被删除（训练时崩溃，无任何有效文件）：

```
runs/M4_CausalDDPM_v3_e5_cuda_20260307_102209    (空)
runs/M4_CausalDDPM_v3_e20_cuda_20260307_104000   (空)
runs/M4_CausalDDPM_v3_e20_cuda_20260307_104115   (空)
runs/M4_CausalDDPM_v3_e20_cuda_20260307_104545   (空)
runs/M4_CausalDDPM_v3_e20_cuda_20260307_104846   (空)
runs/M4_CausalDDPM_v3_e8_cuda_20260307_105939    (空)
runs/M4_CausalDDPM_v3_e8_cuda_20260307_110016    (空)
runs/M4_CausalDDPM_v3_e8_cuda_20260307_110254    (空)
runs/M4_CausalDDPM_v3_e8_cuda_20260307_110344    (空)
runs/causal_ddpm_experiment                       (仅日志，无 checkpoint)
```

以下旧 exp 输出也将清理（基于旧特征集，已失效）：

```
exp/nyc_crash_c4/causal_m4/                       (旧合成数据, 将被新训练覆盖)
exp/nyc_crash_c4/causal_m4_hurdle/                (旧 Hurdle 结果)
exp/nyc_crash_c4/causal_m4_hurdle_grid_g1~g6/     (旧网格搜索)
exp/nyc_crash_c4/causal_m4_mean_shift/            (旧均值偏移)
exp/nyc_crash_c4/causal_m4_quantile_map/          (旧分位映射)
exp/nyc_crash_c4/causal_m4_resampled159990/       (旧重采样)
exp/nyc_crash_c4/causal_m4_teacher/               (旧教师标签)
exp/nyc_crash_c4/ddpm_mlp_mean_shift/             (旧DDPM均值偏移)
exp/nyc_crash_c4/ddpm_mlp_quantile_map/           (旧DDPM分位映射)
exp/nyc_crash_c4/ddpm_mlp_resampled159990/        (旧DDPM重采样)
```
