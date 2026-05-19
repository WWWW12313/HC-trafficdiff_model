# Synthetic evaluation report

- real_test: `C:\Users\Admin\Desktop\hujunzhe\课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `macro_sparse_anneal_v2_full.csv`
- primary_metrics_profile: `auto`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_sparse_anneal_v2_full.csv | 10000 | 0.210084 | 0.008924 | 0.719881 | random_forest | 0.74401 | 0.613785 | 0.284447 | 0.201776 |

## TSTR Benchmark Details

### macro_sparse_anneal_v2_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.7198807020443455, "best_model": "random_forest", "best_model_avg_score": 0.7440095112960671, "mean_r2": 0.5917854785888831, "mean_mse": 0.30064926043567025, "mean_mae": 0.2569740632905913}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7414775622438317 | 0.6137851476669312 | 0.2844465672969818 | 0.20177558064460754 |  |
| random_forest | ok | 0.7440095112960671 | 0.6170157432465078 | 0.2820672242467691 | 0.19757253519724396 |  |
| mlp | ok | 0.6741550325931377 | 0.5445555448532104 | 0.3354339897632599 | 0.3715740740299225 |  |

