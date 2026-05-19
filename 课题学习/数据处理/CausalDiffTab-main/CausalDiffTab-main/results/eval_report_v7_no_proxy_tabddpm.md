# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `baseline_tabddpm_full.csv`
- primary_metrics_profile: `auto`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_excluded_proxy_cols_count | tstr_feature_cols_used | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_tabddpm_full.csv | 10000 | 0.784472 | 0.014067 | 8 | 24 | 0.393443 | xgboost | 0.396222 | -0.008783 | 0.742967 | 0.626198 |

## TSTR Benchmark Details

### baseline_tabddpm_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.39344313709177503, "best_model": "xgboost", "best_model_avg_score": 0.3962218886018256, "mean_r2": -0.02979913471191183, "mean_mse": 0.7584452017121609, "mean_mae": 0.6351366621840523}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.3962218886018256 | -0.008782625198364258 | 0.7429665923118591 | 0.6261981725692749 |  |
| random_forest | ok | 0.39462063617593257 | -0.01544222587493227 | 0.7478713941493916 | 0.6346882529820579 |  |
| mlp | ok | 0.38948688649756696 | -0.06517255306243896 | 0.7844976186752319 | 0.644523561000824 |  |

