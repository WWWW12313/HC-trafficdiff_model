# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `macro_sparse_forbidden_v2_balanced.csv`
- primary_metrics_profile: `full`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | cmi_abs_error_mean | cmi_rel_error_mean | shd | shd_normalized | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_sparse_forbidden_v2_balanced.csv | 10000 | 0.210084 | 0.010758 | 0.037197 | 0.138777 | nan | nan | 0.682873 | random_forest | 0.694227 | 0.514117 | 0.357852 | 0.258388 |

## TSTR Benchmark Details

### macro_sparse_forbidden_v2_balanced.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.6828728242523382, "best_model": "random_forest", "best_model_avg_score": 0.6942271867971517, "mean_r2": 0.520927243536533, "mean_mse": 0.3528362168978096, "mean_mae": 0.2689907817788715}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.6817472989934501 | 0.5141173005104065 | 0.35785171389579773 | 0.258388489484787 |  |
| random_forest | ok | 0.6942271867971517 | 0.5312502774769694 | 0.3452333373160483 | 0.23752295039528448 |  |
| mlp | ok | 0.672643986966413 | 0.5174141526222229 | 0.35542359948158264 | 0.31106090545654297 |  |


## 综合排名（profile=full）

综合分 = 各层归一化分数的平均值（1.0=最优，0.0=最差）。排名依据由 `--primary_metrics_profile` 决定。

| rank | file | composite_score | cmi_abs_error_mean | shd_normalized | tstr_avg_score | tstr_r2_or_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | macro_sparse_forbidden_v2_balanced.csv | 0.0 | 0.037197 | nan | 0.682873 | 0.514117 |
