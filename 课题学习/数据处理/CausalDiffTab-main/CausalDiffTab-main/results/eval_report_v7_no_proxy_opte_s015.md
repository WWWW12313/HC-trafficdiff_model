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

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_excluded_proxy_cols_count | tstr_feature_cols_used | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_sparse_anneal_v2_full_relation_calibrated_s015.csv | 10000 | 0.210084 | 0.008924 | 8 | 37 | 0.402321 | xgboost | 0.414704 | 0.028947 | 0.715178 | 0.581944 |

## TSTR Benchmark Details

### macro_sparse_anneal_v2_full_relation_calibrated_s015.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.40232142455991987, "best_model": "xgboost", "best_model_avg_score": 0.41470359364436654, "mean_r2": -0.035154253084165345, "mean_mse": 0.7623892382273203, "mean_mae": 0.6122224567401898}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.41470359364436654 | 0.02894735336303711 | 0.7151784896850586 | 0.5819437503814697 |  |
| random_forest | ok | 0.41210037683160405 | 0.025729924921087943 | 0.7175481782821198 | 0.5914800876583135 |  |
| mlp | ok | 0.38016030320378896 | -0.1601400375366211 | 0.8544410467147827 | 0.6632435321807861 |  |

