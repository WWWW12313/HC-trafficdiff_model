# Synthetic evaluation report

- real_test: `data\nyc_crash_2024_v2\test.csv`
- task_type: `regression`
- target_col: `NUMBER OF PERSONS INJURED`
- file_glob: `ablation_no_causal_full.csv`
- primary_metrics_profile: `causal_traffic`

## 三层指标体系说明

| 层级 | 指标 | 优先级 | 说明 |
| --- | --- | --- | --- |
| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |
| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |

| file | n_rows | mean_wasserstein_numeric | mean_js_categorical | target_real_mean | target_syn_mean | target_mean_abs_error | target_real_zero_rate | target_syn_zero_rate | target_zero_rate_abs_error | target_real_ge1_rate | target_syn_ge1_rate | target_ge1_rate_abs_error | target_real_ge2_rate | target_syn_ge2_rate | target_ge2_rate_abs_error | target_wasserstein | tstr_standard_feature_cols_available | tstr_proxy_cols_count | tstr_no_proxy_feature_cols_available | tstr_avg_score | tstr_best_model | tstr_best_model_score | tstr_r2 | tstr_mse | tstr_mae | standard_tstr_avg_score | standard_tstr_best_model | standard_tstr_best_model_score | standard_tstr_r2 | standard_tstr_mse | standard_tstr_mae | no_proxy_tstr_avg_score | no_proxy_tstr_best_model | no_proxy_tstr_best_model_score | no_proxy_tstr_r2 | no_proxy_tstr_mse | no_proxy_tstr_mae | proxy_only_tstr_avg_score | proxy_only_tstr_best_model | proxy_only_tstr_best_model_score | proxy_only_tstr_r2 | proxy_only_tstr_mse | proxy_only_tstr_mae | proxy_leakage_r2_gap | proxy_leakage_avg_score_gap | macro_relation_n_specs | macro_relation_group_mae_mean | macro_relation_cmi_abs_error_mean | macro_relation_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_no_causal_full.csv | 10000 | 0.867831 | 0.01585 | 0.589641 | 0.5082 | 0.081441 | 0.560846 | 0.586 | 0.025154 | 0.439154 | 0.414 | 0.025154 | 0.096242 | 0.0714 | 0.024842 | 0.081441 | 32 | 8 | 24 | 0.723948 | random_forest | 0.745223 | 0.615258 | 0.283362 | 0.205648 | 0.723948 | random_forest | 0.745223 | 0.615258 | 0.283362 | 0.205648 | 0.402598 | random_forest | 0.406332 | 0.013686 | 0.726418 | 0.601102 | 0.73655 | xgboost | 0.738379 | 0.611082 | 0.286438 | 0.209605 | 0.601572 | 0.32135 | 16 | 0.080241 | 1.14284 | [{'relation': 'road_to_injury', 'n_specs': 5, 'mean_group_mae': 0.078573, 'max_group_mae': 0.081753, 'mean_cmi_abs_error': 2.76005, 'max_cmi_abs_error': 2.796046}, {'relation': 'vehicle_to_injury', 'n_specs': 7, 'mean_group_mae': 0.080354, 'max_group_mae': 0.081551, 'mean_cmi_abs_error': 0.008615, 'max_cmi_abs_error': 0.017006}, {'relation': 'weather_to_injury', 'n_specs': 4, 'mean_group_mae': 0.081797, 'max_group_mae': 0.082266, 'mean_cmi_abs_error': 0.659856, 'max_cmi_abs_error': 1.363355}] |

## TSTR Benchmark Details

### ablation_no_causal_full.csv

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.7239477735176436, "best_model": "random_forest", "best_model_avg_score": 0.745222808335693, "mean_r2": 0.5884825296638351, "mean_mse": 0.30308188292514976, "mean_mae": 0.2271027806005772}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7412969678184816 | 0.6152581572532654 | 0.28336167335510254 | 0.20564845204353333 |  |
| random_forest | ok | 0.745222808335693 | 0.6214082561248366 | 0.2788321463206916 | 0.2014945872431684 |  |
| mlp | ok | 0.6853235443987562 | 0.5287811756134033 | 0.34705182909965515 | 0.2741653025150299 |  |

### ablation_no_causal_full.csv::standard_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.7239477735176436, "best_model": "random_forest", "best_model_avg_score": 0.745222808335693, "mean_r2": 0.5884825296638351, "mean_mse": 0.30308188292514976, "mean_mae": 0.2271027806005772}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7412969678184816 | 0.6152581572532654 | 0.28336167335510254 | 0.20564845204353333 |  |
| random_forest | ok | 0.745222808335693 | 0.6214082561248366 | 0.2788321463206916 | 0.2014945872431684 |  |
| mlp | ok | 0.6853235443987562 | 0.5287811756134033 | 0.34705182909965515 | 0.2741653025150299 |  |

### ablation_no_causal_full.csv::no_proxy_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.4025976386227625, "best_model": "random_forest", "best_model_avg_score": 0.40633219211325783, "mean_r2": 0.00549367003017535, "mean_mse": 0.7324521257247153, "mean_mae": 0.6128658945324723}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.40583009761691063 | 0.013686418533325195 | 0.726418137550354 | 0.6011017560958862 |  |
| random_forest | ok | 0.40633219211325783 | 0.017852155338939135 | 0.7233501415746281 | 0.6106190368421983 |  |
| mlp | ok | 0.39563062613811883 | -0.015057563781738281 | 0.7475880980491638 | 0.6268768906593323 |  |

### ablation_no_causal_full.csv::proxy_only_tstr

- summary: {"task_type": "regression", "n_models_ok": 3, "avg_score_mean": 0.736550352011435, "best_model": "xgboost", "best_model_avg_score": 0.7383793358410268, "mean_r2": 0.6057139275954869, "mean_mse": 0.290390965155601, "mean_mae": 0.2063423261440501}

| model | status | avg_score | r2 | mse | mae | error |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | ok | 0.7383793358410268 | 0.611081600189209 | 0.28643766045570374 | 0.20960526168346405 |  |
| random_forest | ok | 0.7382542088795899 | 0.6108339543600997 | 0.2866201007719025 | 0.20963088136813207 |  |
| mlp | ok | 0.7330175113136886 | 0.5952262282371521 | 0.2981151342391968 | 0.1997908353805542 |  |

### ablation_no_causal_full.csv::macro_relations

- summary: {"macro_relation_n_specs": 16, "macro_relation_group_mae_mean": 0.080241, "macro_relation_cmi_abs_error_mean": 1.14284}

| relation | parent_col | kind | status | n_groups_real | n_groups_syn | weighted_mean_abs_error | max_abs_error | group_details | real_cmi | syn_cmi | cmi_abs_error | cmi_rel_error | cond_cols |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weather_to_injury | WEATHER_CONDITION | categorical | ok | 4 | 4 | 0.081743 | 0.110983 | [{'group': 'Clear', 'real_mean_injury': 0.603471, 'syn_mean_injury': 0.526304, 'real_weight': 0.450923, 'abs_error': 0.077167}, {'group': 'Cloudy', 'real_mean_injury': 0.581876, 'syn_mean_injury': 0.502597, 'real_weight': 0.3962, 'abs_error': 0.079278}, {'group': 'Rain', 'real_mean_injury': 0.576271, 'syn_mean_injury': 0.475433, 'real_weight': 0.140988, 'abs_error': 0.100839}, {'group': 'Snow', 'real_mean_injury': 0.482412, 'syn_mean_injury': 0.371429, 'real_weight': 0.011888, 'abs_error': 0.110983}] | 4.278674 | 4.239718 | 0.038956 | 0.009105 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | TEMP_C | numeric_bin | ok | 5 | 5 | 0.082266 | 0.124617 | [{'group': '(-inf, 4.0]', 'real_mean_injury': 0.548083, 'syn_mean_injury': 0.504668, 'real_weight': 0.202521, 'abs_error': 0.043414}, {'group': '(10.1, 17.1]', 'real_mean_injury': 0.598863, 'syn_mean_injury': 0.480759, 'real_weight': 0.199713, 'abs_error': 0.118104}, {'group': '(17.1, 23.4]', 'real_mean_injury': 0.587848, 'syn_mean_injury': 0.522613, 'real_weight': 0.199594, 'abs_error': 0.065235}, {'group': '(23.4, inf]', 'real_mean_injury': 0.660617, 'syn_mean_injury': 0.536, 'real_weight': 0.197503, 'abs_error': 0.124617}, {'group': '(4.0, 10.1]', 'real_mean_injury': 0.554332, 'syn_mean_injury': 0.493267, 'real_weight': 0.200669, 'abs_error': 0.061065}] | 1.135817 | 2.316392 | 1.180575 | 1.039406 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | prcp | numeric_bin | ok | 1 | 1 | 0.081441 | 0.081441 | [{'group': '(-inf, inf]', 'real_mean_injury': 0.589641, 'syn_mean_injury': 0.5082, 'real_weight': 1.0, 'abs_error': 0.081441}] | 3.981275 | 3.924738 | 0.056537 | 0.014201 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| weather_to_injury | WIND_SPEED_KMH | numeric_bin | ok | 5 | 5 | 0.08174 | 0.09407 | [{'group': '(-inf, 7.2]', 'real_mean_injury': 0.566303, 'syn_mean_injury': 0.480612, 'real_weight': 0.208137, 'abs_error': 0.085691}, {'group': '(10.1, 13.5]', 'real_mean_injury': 0.601703, 'syn_mean_injury': 0.525, 'real_weight': 0.196487, 'abs_error': 0.076703}, {'group': '(13.5, 18.0]', 'real_mean_injury': 0.599461, 'syn_mean_injury': 0.511111, 'real_weight': 0.199415, 'abs_error': 0.08835}, {'group': '(18.0, inf]', 'real_mean_injury': 0.60307, 'syn_mean_injury': 0.509, 'real_weight': 0.198518, 'abs_error': 0.09407}, {'group': '(7.2, 10.1]', 'real_mean_injury': 0.57882, 'syn_mean_injury': 0.515306, 'real_weight': 0.197443, 'abs_error': 0.063514}] | 1.003119 | 2.366474 | 1.363355 | 1.359115 | ['SEASON', 'CRASH_TIME_SIN', 'CRASH_TIME_COS'] |
| road_to_injury | HAS_TRAFFIC_SIGNAL | categorical | ok | 2 | 2 | 0.075661 | 0.089433 | [{'group': '0.0', 'real_mean_injury': 0.536526, 'syn_mean_injury': 0.447093, 'real_weight': 0.470219, 'abs_error': 0.089433}, {'group': '1.0', 'real_mean_injury': 0.636784, 'syn_mean_injury': 0.573347, 'real_weight': 0.529781, 'abs_error': 0.063437}] | 0.288408 | 3.009014 | 2.720606 | 9.433184 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | OSM_ONEWAY | categorical | ok | 2 | 2 | 0.078048 | 0.102308 | [{'group': '0.0', 'real_mean_injury': 0.613838, 'syn_mean_injury': 0.557143, 'real_weight': 0.531872, 'abs_error': 0.056695}, {'group': '1.0', 'real_mean_injury': 0.562149, 'syn_mean_injury': 0.459841, 'real_weight': 0.468128, 'abs_error': 0.102308}] | 0.257255 | 3.00924 | 2.751986 | 10.697522 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | OSM_TYPE | categorical | ok | 15 | 13 | 0.081753 | 0.524706 | [{'group': 'living_street', 'real_mean_injury': 0.461538, 'syn_mean_injury': 0.6, 'real_weight': 0.000777, 'abs_error': 0.138462}, {'group': 'motorway', 'real_mean_injury': 0.811606, 'syn_mean_injury': 0.645882, 'real_weight': 0.071032, 'abs_error': 0.165724}, {'group': 'motorway_link', 'real_mean_injury': 0.815427, 'syn_mean_injury': 0.7, 'real_weight': 0.021686, 'abs_error': 0.115427}, {'group': 'pedestrian', 'real_mean_injury': 0.666667, 'syn_mean_injury': None, 'real_weight': 0.000179, 'abs_error': None}, {'group': 'primary', 'real_mean_injury': 0.621278, 'syn_mean_injury': 0.554357, 'real_weight': 0.136448, 'abs_error': 0.066922}, {'group': 'primary_link', 'real_mean_injury': 0.555556, 'syn_mean_injury': 0.733333, 'real_weight': 0.002688, 'abs_error': 0.177778}, {'group': 'residential', 'real_mean_injury': 0.499852, 'syn_mean_injury': 0.442568, 'real_weight': 0.404086, 'abs_error': 0.057285}, {'group': 'secondary', 'real_mean_injury': 0.637476, 'syn_mean_injury': 0.5455, 'real_weight': 0.214887, 'abs_error': 0.091976}, {'group': 'secondary_link', 'real_mean_injury': 0.764706, 'syn_mean_injury': 0.24, 'real_weight': 0.003047, 'abs_error': 0.524706}, {'group': 'service', 'real_mean_injury': 0.442991, 'syn_mean_injury': 0.330769, 'real_weight': 0.031961, 'abs_error': 0.112221}, {'group': 'tertiary', 'real_mean_injury': 0.649689, 'syn_mean_injury': 0.569953, 'real_weight': 0.096183, 'abs_error': 0.079736}, {'group': 'tertiary_link', 'real_mean_injury': 0.75, 'syn_mean_injury': 1.2, 'real_weight': 0.000478, 'abs_error': 0.45}, {'group': 'trunk', 'real_mean_injury': 0.64, 'syn_mean_injury': 0.4, 'real_weight': 0.005974, 'abs_error': 0.24}, {'group': 'trunk_link', 'real_mean_injury': 0.333333, 'syn_mean_injury': None, 'real_weight': 0.000358, 'abs_error': None}, {'group': 'unclassified', 'real_mean_injury': 0.520468, 'syn_mean_injury': 0.490909, 'real_weight': 0.010216, 'abs_error': 0.029559}] | 0.252569 | 3.00108 | 2.748511 | 10.882239 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | DIST_TO_SIGNAL_M | numeric_bin | ok | 5 | 5 | 0.078514 | 0.106523 | [{'group': '(-inf, 1.194]', 'real_mean_injury': 0.667065, 'syn_mean_injury': 0.577401, 'real_weight': 0.200072, 'abs_error': 0.089664}, {'group': '(1.194, 8.213]', 'real_mean_injury': 0.653031, 'syn_mean_injury': 0.571839, 'real_weight': 0.200072, 'abs_error': 0.081192}, {'group': '(105.443, inf]', 'real_mean_injury': 0.565114, 'syn_mean_injury': 0.45859, 'real_weight': 0.200012, 'abs_error': 0.106523}, {'group': '(48.777, 105.443]', 'real_mean_injury': 0.521505, 'syn_mean_injury': 0.416107, 'real_weight': 0.200012, 'abs_error': 0.105398}, {'group': '(8.213, 48.777]', 'real_mean_injury': 0.541405, 'syn_mean_injury': 0.551134, 'real_weight': 0.199833, 'abs_error': 0.009728}] | 0.232109 | 3.015209 | 2.7831 | 11.99047 | ['LATITUDE', 'LONGITUDE'] |
| road_to_injury | INFERRED_LANES | numeric_bin | ok | 3 | 3 | 0.078891 | 0.174622 | [{'group': '(-inf, 2.0]', 'real_mean_injury': 0.560577, 'syn_mean_injury': 0.487176, 'real_weight': 0.749507, 'abs_error': 0.073401}, {'group': '(2.0, 3.0]', 'real_mean_injury': 0.733148, 'syn_mean_injury': 0.558525, 'real_weight': 0.107235, 'abs_error': 0.174622}, {'group': '(3.0, inf]', 'real_mean_injury': 0.634279, 'syn_mean_injury': 0.598326, 'real_weight': 0.143258, 'abs_error': 0.035952}] | 0.246052 | 3.042098 | 2.796046 | 11.363613 | ['LATITUDE', 'LONGITUDE'] |
| vehicle_to_injury | is_sedan | categorical | ok | 2 | 2 | 0.081477 | 0.085678 | [{'group': '0', 'real_mean_injury': 0.601333, 'syn_mean_injury': 0.515655, 'real_weight': 0.41239, 'abs_error': 0.085678}, {'group': '1', 'real_mean_injury': 0.581436, 'syn_mean_injury': 0.502907, 'real_weight': 0.58761, 'abs_error': 0.078529}] | 0.951568 | 0.952377 | 0.000809 | 0.000851 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_suv | categorical | ok | 2 | 2 | 0.081551 | 0.085184 | [{'group': '0', 'real_mean_injury': 0.57908, 'syn_mean_injury': 0.500557, 'real_weight': 0.545433, 'abs_error': 0.078523}, {'group': '1', 'real_mean_injury': 0.602313, 'syn_mean_injury': 0.517129, 'real_weight': 0.454567, 'abs_error': 0.085184}] | 0.949744 | 0.952095 | 0.002351 | 0.002475 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_taxi | categorical | ok | 2 | 2 | 0.080764 | 0.082484 | [{'group': '0', 'real_mean_injury': 0.584152, 'syn_mean_injury': 0.501668, 'real_weight': 0.955194, 'abs_error': 0.082484}, {'group': '1', 'real_mean_injury': 0.706667, 'syn_mean_injury': 0.662562, 'real_weight': 0.044806, 'abs_error': 0.044105}] | 0.938913 | 0.95262 | 0.013707 | 0.014599 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_truck | categorical | ok | 2 | 2 | 0.081443 | 0.08341 | [{'group': '0', 'real_mean_injury': 0.594449, 'syn_mean_injury': 0.513225, 'real_weight': 0.899755, 'abs_error': 0.081224}, {'group': '1', 'real_mean_injury': 0.546484, 'syn_mean_injury': 0.463074, 'real_weight': 0.100245, 'abs_error': 0.08341}] | 0.943197 | 0.960202 | 0.017006 | 0.01803 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_bus | categorical | ok | 2 | 2 | 0.081169 | 0.082016 | [{'group': '0', 'real_mean_injury': 0.593918, 'syn_mean_injury': 0.512781, 'real_weight': 0.964633, 'abs_error': 0.081138}, {'group': '1', 'real_mean_injury': 0.472973, 'syn_mean_injury': 0.390957, 'real_weight': 0.035367, 'abs_error': 0.082016}] | 0.93983 | 0.948385 | 0.008555 | 0.009103 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_motorcycle | categorical | ok | 2 | 2 | 0.080731 | 0.083404 | [{'group': '0', 'real_mean_injury': 0.579199, 'syn_mean_injury': 0.495795, 'real_weight': 0.960989, 'abs_error': 0.083404}, {'group': '1', 'real_mean_injury': 0.846861, 'syn_mean_injury': 0.831978, 'real_weight': 0.039011, 'abs_error': 0.014882}] | 0.95478 | 0.962503 | 0.007722 | 0.008088 | ['TOTAL_VEHICLES'] |
| vehicle_to_injury | is_bicycle | categorical | ok | 2 | 2 | 0.07534 | 0.080175 | [{'group': '0', 'real_mean_injury': 0.561378, 'syn_mean_injury': 0.481203, 'real_weight': 0.915407, 'abs_error': 0.080175}, {'group': '1', 'real_mean_injury': 0.89548, 'syn_mean_injury': 0.872464, 'real_weight': 0.084593, 'abs_error': 0.023016}] | 0.992318 | 0.982162 | 0.010156 | 0.010235 | ['TOTAL_VEHICLES'] |
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
| 1 | ablation_no_causal_full.csv | 0.0 | None | None | 0.723948 | 0.615258 | 0.402598 | 0.013686 | 0.080241 | 1.14284 | 0.081441 | 0.025154 |
