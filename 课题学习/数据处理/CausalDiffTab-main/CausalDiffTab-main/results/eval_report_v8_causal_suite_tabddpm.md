# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `baseline_tabddpm_full.csv`
- primary_metrics_profile: `causal_traffic`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | target_real_mean | target_syn_mean | target_mean_abs_error | target_real_zero_rate | target_syn_zero_rate | target_zero_rate_abs_error | target_real_ge1_rate | target_syn_ge1_rate | target_ge1_rate_abs_error | target_real_ge2_rate | target_syn_ge2_rate | target_ge2_rate_abs_error | target_wasserstein | tstr_standard_feature_cols_available | tstr_proxy_cols_count | tstr_no_proxy_feature_cols_available | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae | standard_tstr_avg_score | standard_tstr_best_model | standard_tstr_best_model_score | standard_tstr_r2 | standard_tstr_mse | standard_tstr_mae | no_proxy_tstr_avg_score | no_proxy_tstr_best_model | no_proxy_tstr_best_model_score | no_proxy_tstr_r2 | no_proxy_tstr_mse | no_proxy_tstr_mae | proxy_only_tstr_avg_score | proxy_only_tstr_best_model | proxy_only_tstr_best_model_score | proxy_only_tstr_r2 | proxy_only_tstr_mse | proxy_only_tstr_mae | proxy_leakage_r2_gap | proxy_leakage_avg_score_gap | macro_relation_n_specs | macro_relation_group_mae_mean | macro_relation_cmi_abs_error_mean | macro_relation_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_tabddpm_full.csv | 10000 | 0.784472 | 0.014067 | 0.589641 | 0.6036 | 0.013959 | 0.560846 | 0.5463 | 0.014546 | 0.439154 | 0.4537 | 0.014546 | 0.096242 | 0.0906 | 0.005642 | 0.025745 | 32 | 8 | 24 | 0.707796 | random_forest | 0.730714 | 0.602562 | 0.292712 | 0.232711 | 0.707796 | random_forest | 0.730714 | 0.602562 | 0.292712 | 0.232711 | 0.393443 | xgboost | 0.396222 | -0.008783 | 0.742967 | 0.626198 | 0.73993 | random_forest | 0.741612 | 0.628894 | 0.273319 | 0.233928 | 0.611345 | 0.314353 | 16 | 0.019162 | 0.312846 | [{'relation': 'road_to_injury', 'n_specs': 5, 'mean_group_mae': 0.02442, 'max_group_mae': 0.033902, 'mean_cmi_abs_error': 0.243829, 'max_cmi_abs_error': 0.264409}, {'relation': 'vehicle_to_injury', 'n_specs': 7, 'mean_group_mae': 0.014425, 'max_group_mae': 0.015702, 'mean_cmi_abs_error': 0.006724, 'max_cmi_abs_error': 0.009064}, {'relation': 'weather_to_injury', 'n_specs': 4, 'mean_group_mae': 0.01864, 'max_group_mae': 0.027108, 'mean_cmi_abs_error': 0.687986, 'max_cmi_abs_error': 1.092133}] |

## TSTR Benchmark Details

### baseline_tabddpm_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.70779638836133, "best_model": "random_forest", "best_model_avg_score": 0.7307139549804357, "mean_r2": 0.5692125942459733, "mean_mse": 0.31727413865712717, "mean_mae": 0.26003445835486416}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7291166781966316 | 0.602562427520752 | 0.2927120625972748 | 0.23271101713180542 |  |
| random_forest | ok | 0.7307139549804357 | 0.603644724533269 | 0.29191495575867404 | 0.22781886612103472 |  |
| mlp | ok | 0.6635585319069226 | 0.5014306306838989 | 0.36719539761543274 | 0.3195734918117523 |  |

### baseline_tabddpm_full.csv::standard_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.70779638836133, "best_model": "random_forest", "best_model_avg_score": 0.7307139549804357, "mean_r2": 0.5692125942459733, "mean_mse": 0.31727413865712717, "mean_mae": 0.26003445835486416}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7291166781966316 | 0.602562427520752 | 0.2927120625972748 | 0.23271101713180542 |  |
| random_forest | ok | 0.7307139549804357 | 0.603644724533269 | 0.29191495575867404 | 0.22781886612103472 |  |
| mlp | ok | 0.6635585319069226 | 0.5014306306838989 | 0.36719539761543274 | 0.3195734918117523 |  |

### baseline_tabddpm_full.csv::no_proxy_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.39344313709177503, "best_model": "xgboost", "best_model_avg_score": 0.3962218886018256, "mean_r2": -0.02979913471191183, "mean_mse": 0.7584452017121609, "mean_mae": 0.6351366621840523}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.3962218886018256 | -0.008782625198364258 | 0.7429665923118591 | 0.6261981725692749 |  |
| random_forest | ok | 0.39462063617593257 | -0.01544222587493227 | 0.7478713941493916 | 0.6346882529820579 |  |
| mlp | ok | 0.38948688649756696 | -0.06517255306243896 | 0.7844976186752319 | 0.644523561000824 |  |

### baseline_tabddpm_full.csv::proxy_only_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.7399297026609215, "best_model": "random_forest", "best_model_avg_score": 0.7416124479825532, "mean_r2": 0.628727894807818, "mean_mse": 0.2734412286168207, "mean_mae": 0.24110454085496855}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7415542709026487 | 0.6288935542106628 | 0.27331921458244324 | 0.23392772674560547 |  |
| random_forest | ok | 0.7416124479825532 | 0.629017596796043 | 0.27322788044389157 | 0.23393663057767725 |  |
| mlp | ok | 0.7366223890975627 | 0.628272533416748 | 0.2737765908241272 | 0.2554492652416229 |  |

### baseline_tabddpm_full.csv::macro_relations

- summary: {"macro_relation_n_specs": 16, "macro_relation_group_mae_mean": 0.019162, "macro_relation_cmi_abs_error_mean": 0.312846}

| relation | parent_col | kind | status | n_groups_real | n_groups_syn | weighted_mean_abs_error | max_abs_error | group_details | real_cmi | syn_cmi | cmi_abs_error | cmi_rel_error | cond_cols |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weather_to_injury | WEATHER_CONDITION | categorical | ok | 4 | 4 | 0.013997 | 0.058329 | [{'group': 'Clear', 'real_mean_injury': 0.603471, 'syn_mean_injury': 0.606507, 'real_weight': 0.450923, 'abs_error': 0.003036}, {'group': 'Cloudy', 'real_mean_injury': 0.581876, 'syn_mean_injury': 0.59447, 'real_weight': 0.3962, 'abs_error': 0.012594}, {'group': 'Rain', 'real_mean_injury': 0.576271, 'syn_mean_injury': 0.625532, 'real_weight': 0.140988, 'abs_error': 0.049261}, {'group': 'Snow', 'real_mean_injury': 0.482412, 'syn_mean_injury': 0.540741, 'real_weight': 0.011888, 'abs_error': 0.058329}] | 4.278674 | 3.750995 | 0.527679 | 0.123328 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | TEMP_C | numeric_bin | ok | 5 | 5 | 0.027108 | 0.056007 | [{'group': '(-inf, 4.0]', 'real_mean_injury': 0.548083, 'syn_mean_injury': 0.533613, 'real_weight': 0.202521, 'abs_error': 0.014469}, {'group': '(10.1, 17.1]', 'real_mean_injury': 0.598863, 'syn_mean_injury': 0.613918, 'real_weight': 0.199713, 'abs_error': 0.015054}, {'group': '(17.1, 23.4]', 'real_mean_injury': 0.587848, 'syn_mean_injury': 0.643855, 'real_weight': 0.199594, 'abs_error': 0.056007}, {'group': '(23.4, inf]', 'real_mean_injury': 0.660617, 'syn_mean_injury': 0.644256, 'real_weight': 0.197503, 'abs_error': 0.016361}, {'group': '(4.0, 10.1]', 'real_mean_injury': 0.554332, 'syn_mean_injury': 0.588025, 'real_weight': 0.200669, 'abs_error': 0.033693}] | 1.135817 | 0.043685 | 1.092133 | 0.961539 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | prcp | numeric_bin | ok | 1 | 1 | 0.013959 | 0.013959 | [{'group': '(-inf, inf]', 'real_mean_injury': 0.589641, 'syn_mean_injury': 0.6036, 'real_weight': 1.0, 'abs_error': 0.013959}] | 3.981275 | 3.829327 | 0.151949 | 0.038166 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | WIND_SPEED_KMH | numeric_bin | ok | 5 | 5 | 0.019497 | 0.0599 | [{'group': '(-inf, 7.2]', 'real_mean_injury': 0.566303, 'syn_mean_injury': 0.570683, 'real_weight': 0.208137, 'abs_error': 0.00438}, {'group': '(10.1, 13.5]', 'real_mean_injury': 0.601703, 'syn_mean_injury': 0.586968, 'real_weight': 0.196487, 'abs_error': 0.014734}, {'group': '(13.5, 18.0]', 'real_mean_injury': 0.599461, 'syn_mean_injury': 0.600404, 'real_weight': 0.199415, 'abs_error': 0.000943}, {'group': '(18.0, inf]', 'real_mean_injury': 0.60307, 'syn_mean_injury': 0.621583, 'real_weight': 0.198518, 'abs_error': 0.018513}, {'group': '(7.2, 10.1]', 'real_mean_injury': 0.57882, 'syn_mean_injury': 0.63872, 'real_weight': 0.197443, 'abs_error': 0.0599}] | 1.003119 | 0.022936 | 0.980183 | 0.977135 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| road_to_injury | HAS_TRAFFIC_SIGNAL | categorical | ok | 2 | 2 | 0.016902 | 0.020865 | [{'group': '0.0', 'real_mean_injury': 0.536526, 'syn_mean_injury': 0.548964, 'real_weight': 0.470219, 'abs_error': 0.012438}, {'group': '1.0', 'real_mean_injury': 0.636784, 'syn_mean_injury': 0.657649, 'real_weight': 0.529781, 'abs_error': 0.020865}] | 0.288408 | 0.023999 | 0.264409 | 0.91679 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | OSM_ONEWAY | categorical | ok | 2 | 2 | 0.025912 | 0.042521 | [{'group': '0.0', 'real_mean_injury': 0.613838, 'syn_mean_injury': 0.602544, 'real_weight': 0.531872, 'abs_error': 0.011294}, {'group': '1.0', 'real_mean_injury': 0.562149, 'syn_mean_injury': 0.60467, 'real_weight': 0.468128, 'abs_error': 0.042521}] | 0.257255 | 0.020588 | 0.236666 | 0.91997 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | OSM_TYPE | categorical | ok | 15 | 14 | 0.033902 | 1.666667 | [{'group': 'living_street', 'real_mean_injury': 0.461538, 'syn_mean_injury': 0.4, 'real_weight': 0.000777, 'abs_error': 0.061538}, {'group': 'motorway', 'real_mean_injury': 0.811606, 'syn_mean_injury': 0.924933, 'real_weight': 0.071032, 'abs_error': 0.113327}, {'group': 'motorway_link', 'real_mean_injury': 0.815427, 'syn_mean_injury': 0.735294, 'real_weight': 0.021686, 'abs_error': 0.080133}, {'group': 'pedestrian', 'real_mean_injury': 0.666667, 'syn_mean_injury': None, 'real_weight': 0.000179, 'abs_error': None}, {'group': 'primary', 'real_mean_injury': 0.621278, 'syn_mean_injury': 0.633748, 'real_weight': 0.136448, 'abs_error': 0.01247}, {'group': 'primary_link', 'real_mean_injury': 0.555556, 'syn_mean_injury': 0.85, 'real_weight': 0.002688, 'abs_error': 0.294444}, {'group': 'residential', 'real_mean_injury': 0.499852, 'syn_mean_injury': 0.526949, 'real_weight': 0.404086, 'abs_error': 0.027097}, {'group': 'secondary', 'real_mean_injury': 0.637476, 'syn_mean_injury': 0.629418, 'real_weight': 0.214887, 'abs_error': 0.008058}, {'group': 'secondary_link', 'real_mean_injury': 0.764706, 'syn_mean_injury': 0.545455, 'real_weight': 0.003047, 'abs_error': 0.219251}, {'group': 'service', 'real_mean_injury': 0.442991, 'syn_mean_injury': 0.466667, 'real_weight': 0.031961, 'abs_error': 0.023676}, {'group': 'tertiary', 'real_mean_injury': 0.649689, 'syn_mean_injury': 0.616063, 'real_weight': 0.096183, 'abs_error': 0.033627}, {'group': 'tertiary_link', 'real_mean_injury': 0.75, 'syn_mean_injury': 0.666667, 'real_weight': 0.000478, 'abs_error': 0.083333}, {'group': 'trunk', 'real_mean_injury': 0.64, 'syn_mean_injury': 0.971429, 'real_weight': 0.005974, 'abs_error': 0.331429}, {'group': 'trunk_link', 'real_mean_injury': 0.333333, 'syn_mean_injury': 2.0, 'real_weight': 0.000358, 'abs_error': 1.666667}, {'group': 'unclassified', 'real_mean_injury': 0.520468, 'syn_mean_injury': 0.678161, 'real_weight': 0.010216, 'abs_error': 0.157693}] | 0.252569 | 0.002147 | 0.250422 | 0.9915 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | DIST_TO_SIGNAL_M | numeric_bin | ok | 5 | 5 | 0.029491 | 0.068501 | [{'group': '(-inf, 1.194]', 'real_mean_injury': 0.667065, 'syn_mean_injury': 0.661554, 'real_weight': 0.200072, 'abs_error': 0.005511}, {'group': '(1.194, 8.213]', 'real_mean_injury': 0.653031, 'syn_mean_injury': 0.666462, 'real_weight': 0.200072, 'abs_error': 0.013431}, {'group': '(105.443, inf]', 'real_mean_injury': 0.565114, 'syn_mean_injury': 0.542619, 'real_weight': 0.200012, 'abs_error': 0.022495}, {'group': '(48.777, 105.443]', 'real_mean_injury': 0.521505, 'syn_mean_injury': 0.559071, 'real_weight': 0.200012, 'abs_error': 0.037565}, {'group': '(8.213, 48.777]', 'real_mean_injury': 0.541405, 'syn_mean_injury': 0.609906, 'real_weight': 0.199833, 'abs_error': 0.068501}] | 0.232109 | 0.007879 | 0.22423 | 0.966055 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | INFERRED_LANES | numeric_bin | ok | 3 | 3 | 0.015893 | 0.079586 | [{'group': '(-inf, 2.0]', 'real_mean_injury': 0.560577, 'syn_mean_injury': 0.56747, 'real_weight': 0.749507, 'abs_error': 0.006892}, {'group': '(2.0, 3.0]', 'real_mean_injury': 0.733148, 'syn_mean_injury': 0.812734, 'real_weight': 0.107235, 'abs_error': 0.079586}, {'group': '(3.0, inf]', 'real_mean_injury': 0.634279, 'syn_mean_injury': 0.649587, 'real_weight': 0.143258, 'abs_error': 0.015308}] | 0.246052 | 0.002635 | 0.243418 | 0.989292 | ['LATITUDE', 'LONGITUDE'] |
| vehicle_to_injury | is_sedan | categorical | ok | 2 | 2 | 0.013916 | 0.01853 | [{'group': '0', 'real_mean_injury': 0.601333, 'syn_mean_injury': 0.608675, 'real_weight': 0.41239, 'abs_error': 0.007342}, {'group': '1', 'real_mean_injury': 0.581436, 'syn_mean_injury': 0.599966, 'real_weight': 0.58761, 'abs_error': 0.01853}] | 0.951568 | 0.943938 | 0.007629 | 0.008017 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_suv | categorical | ok | 2 | 2 | 0.014089 | 0.020348 | [{'group': '0', 'real_mean_injury': 0.57908, 'syn_mean_injury': 0.599428, 'real_weight': 0.545433, 'abs_error': 0.020348}, {'group': '1', 'real_mean_injury': 0.602313, 'syn_mean_injury': 0.608893, 'real_weight': 0.454567, 'abs_error': 0.00658}] | 0.949744 | 0.944406 | 0.005339 | 0.005621 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_taxi | categorical | ok | 2 | 2 | 0.013558 | 0.0136 | [{'group': '0', 'real_mean_injury': 0.584152, 'syn_mean_injury': 0.597752, 'real_weight': 0.955194, 'abs_error': 0.0136}, {'group': '1', 'real_mean_injury': 0.706667, 'syn_mean_injury': 0.719335, 'real_weight': 0.044806, 'abs_error': 0.012668}] | 0.938913 | 0.947753 | 0.008839 | 0.009415 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_truck | categorical | ok | 2 | 2 | 0.013946 | 0.036264 | [{'group': '0', 'real_mean_injury': 0.594449, 'syn_mean_injury': 0.605909, 'real_weight': 0.899755, 'abs_error': 0.01146}, {'group': '1', 'real_mean_injury': 0.546484, 'syn_mean_injury': 0.582748, 'real_weight': 0.100245, 'abs_error': 0.036264}] | 0.943197 | 0.93462 | 0.008576 | 0.009093 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_bus | categorical | ok | 2 | 2 | 0.014789 | 0.030491 | [{'group': '0', 'real_mean_injury': 0.593918, 'syn_mean_injury': 0.608132, 'real_weight': 0.964633, 'abs_error': 0.014214}, {'group': '1', 'real_mean_injury': 0.472973, 'syn_mean_injury': 0.503464, 'real_weight': 0.035367, 'abs_error': 0.030491}] | 0.93983 | 0.944637 | 0.004806 | 0.005114 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_motorcycle | categorical | ok | 2 | 2 | 0.015702 | 0.028139 | [{'group': '0', 'real_mean_injury': 0.579199, 'syn_mean_injury': 0.594396, 'real_weight': 0.960989, 'abs_error': 0.015197}, {'group': '1', 'real_mean_injury': 0.846861, 'syn_mean_injury': 0.875, 'real_weight': 0.039011, 'abs_error': 0.028139}] | 0.95478 | 0.957595 | 0.002815 | 0.002948 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_bicycle | categorical | ok | 2 | 2 | 0.014973 | 0.021683 | [{'group': '0', 'real_mean_injury': 0.561378, 'syn_mean_injury': 0.575731, 'real_weight': 0.915407, 'abs_error': 0.014353}, {'group': '1', 'real_mean_injury': 0.89548, 'syn_mean_injury': 0.873797, 'real_weight': 0.084593, 'abs_error': 0.021683}] | 0.992318 | 0.983254 | 0.009064 | 0.009134 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_other_vehicle | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_distracted | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_speeding | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_failure_to_yield | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_following_too_closely | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_drunk_driving | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_fatigue | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_view_obstructed | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_vehicle_defect | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_backing_unsafely | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_pedestrian_related | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_inexperience | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |
| crash_type_to_injury | is_pavement_slippery | categorical | missing_column |  |  |  |  |  |  |  |  |  |  |


## 综合排名（profile=causal_traffic）

综合分 = 各层归一化分数的平均值（1.0=最优，0.0=最差）。排名依据由 `--primary_metrics_profile` 决定。

| rank | file | composite_score | cmi_abs_error_mean | shd_normalized | tstr_avg_score | tstr_r2_or_accuracy | no_proxy_tstr_avg_score | no_proxy_tstr_r2_or_accuracy | macro_relation_group_mae_mean | macro_relation_cmi_abs_error_mean | target_mean_abs_error | target_zero_rate_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline_tabddpm_full.csv | 0.0 | None | None | 0.707796 | 0.602562 | 0.393443 | -0.008783 | 0.019162 | 0.312846 | 0.013959 | 0.014546 |
