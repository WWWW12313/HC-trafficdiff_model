# Phase 1 消融实验同步日志

> 方案：C（同步进行）
> - 当前主模型 (exp_c_macro_full_v1) 扩展至 10k 评估样本
> - 同时启动所有消融/基线模型的 5k 统一评估
> - 统一标准：no_proxy TSTR + macro_relation (v2 schema) + proxy_gap

## 模型清单

| 模型 | Checkpoint 目录 | 采样量 | 状态 |
|------|----------------|--------|------|
| ours_full_model (exp_c_macro_full_v1) | stage3_full_full_exp_c_macro_full_v1 | 10k | **RUNNING** |
| macro_sparse_anneal_v2 (v2 基线) | stage3_full_full_macro_sparse_anneal_v2 | 5k | **RUNNING** |
| ablation_no_causal_v2 | stage3_full_full_ablation_no_causal_v2 | 5k | **RUNNING** |
| ablation_no_hierarchy_v2 | stage3_full_full_ablation_no_hierarchy_v2 | 5k | **RUNNING** |
| baseline_tabddpm_v2 | stage3_full_full_baseline_tabddpm_v2 | 5k | **RUNNING** |

## 操作记录

### 2026-05-23 21:58
- 启动统一采样脚本（task_id: `bash-hj9uycvh`）
- 串行执行 5 个模型的采样（避免 GPU 冲突）
- 预计总耗时：10k(3min) + 4*5k(1.5min*4=6min) ≈ **9-10 分钟**
- 命令详情见 `/tmp/run_all_sampling.sh`

### 当前已知结果（exp_c_macro_full_v1, 5k）
| 指标 | 值 |
|------|-----|
| standard TSTR | 0.3933 |
| no_proxy TSTR | 0.3933 |
| proxy_gap | -0.0027 |
| macro_relation MAE (29 specs) | **0.0399** |
| macro_relation CMI err | 0.3453 |

## 待执行

- [ ] 5 个模型采样全部完成
- [ ] 对每个模型运行 evaluate_all.py（--exclude_proxy_outcomes --causal_eval_suite）
- [ ] 对每个模型的 physical.csv 运行 evaluate_macro_relations.py（v2 schema）
- [ ] 汇总对比表格
