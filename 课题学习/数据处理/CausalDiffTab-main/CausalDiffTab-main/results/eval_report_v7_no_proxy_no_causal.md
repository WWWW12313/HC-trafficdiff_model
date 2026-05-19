# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `ablation_no_causal_full.csv`
- primary_metrics_profile: `auto`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | tstr_excluded_proxy_cols_count | tstr_feature_cols_used | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_no_causal_full.csv | 10000 | 0.867831 | 0.01585 | 8 | 24 | 0.402598 | random_forest | 0.406332 | 0.013686 | 0.726418 | 0.601102 |

## TSTR Benchmark Details

### ablation_no_causal_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.4025976386227625, "best_model": "random_forest", "best_model_avg_score": 0.40633219211325783, "mean_r2": 0.00549367003017535, "mean_mse": 0.7324521257247153, "mean_mae": 0.6128658945324723}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.40583009761691063 | 0.013686418533325195 | 0.726418137550354 | 0.6011017560958862 |  |
| random_forest | ok | 0.40633219211325783 | 0.017852155338939135 | 0.7233501415746281 | 0.6106190368421983 |  |
| mlp | ok | 0.39563062613811883 | -0.015057563781738281 | 0.7475880980491638 | 0.6268768906593323 |  |

