# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `macro_sparse_anneal_v2_full.csv`
- primary_metrics_profile: `auto`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_excluded_proxy_cols_count | tstr_feature_cols_used | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_sparse_anneal_v2_full.csv | 10000 | 0.210084 | 0.008924 | 8 | 37 | 0.402093 | xgboost | 0.413749 | 0.02542 | 0.717777 | 0.578081 |

## TSTR Benchmark Details

### macro_sparse_anneal_v2_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.4020930137507585, "best_model": "xgboost", "best_model_avg_score": 0.41374943863896424, "mean_r2": -0.03720851698823132, "mean_mse": 0.76390220428637, "mean_mae": 0.6081858233382659}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.41374943863896424 | 0.02541959285736084 | 0.7177767157554626 | 0.5780813694000244 |  |
| random_forest | ok | 0.4117502959513641 | 0.024084211608548234 | 0.7187602432111303 | 0.5889360176832548 |  |
| mlp | ok | 0.3807793066619471 | -0.16112935543060303 | 0.8551696538925171 | 0.6575400829315186 |  |

