# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `macro_sparse_anneal_v2_full_relation_calibrated_s015.csv`
- primary_metrics_profile: `auto`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_sparse_anneal_v2_full_relation_calibrated_s015.csv | 10000 | 0.210084 | 0.008924 | 0.73392 | random_forest | 0.742703 | 0.615105 | 0.283474 | 0.208047 |

## TSTR Benchmark Details

### macro_sparse_anneal_v2_full_relation_calibrated_s015.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.7339201110206798, "best_model": "random_forest", "best_model_avg_score": 0.7427034256889931, "mean_r2": 0.6086072298876973, "mean_mse": 0.2882600636611257, "mean_mae": 0.2245958119915115}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7406741841150302 | 0.6151052713394165 | 0.28347426652908325 | 0.20804743468761444 |  |
| random_forest | ok | 0.7427034256889931 | 0.61981528192536 | 0.28000536898668826 | 0.2091192663568816 |  |
| mlp | ok | 0.7183827232580161 | 0.5909011363983154 | 0.3013005554676056 | 0.25662073493003845 |  |

