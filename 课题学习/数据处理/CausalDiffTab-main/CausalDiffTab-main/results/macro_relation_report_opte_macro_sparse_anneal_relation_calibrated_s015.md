# Macro Relation Report: macro_sparse_anneal_v2_full_relation_calibrated_s015.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.089481 | 0.091349 | 0.20882 | 0.298422 |
| road_to_injury | 5 | 0.088884 | 0.089288 | 0.148544 | 0.17089 |
| vehicle_to_injury | 8 | 0.088898 | 0.089907 | 0.021304 | 0.038915 |
| weather_to_injury | 4 | 0.089458 | 0.089548 | 0.469965 | 0.546201 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.089474 | 0.365502 | 4.278674 | 3.913171 |
| weather_to_injury | TEMP_C | 0.089286 | 0.546201 | 1.135817 | 0.589616 |
| weather_to_injury | prcp | 0.089525 | 0.453869 | 3.981275 | 3.527407 |
| weather_to_injury | WIND_SPEED_KMH | 0.089548 | 0.514289 | 1.003119 | 0.488831 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.0885 | 0.17089 | 0.288408 | 0.117518 |
| road_to_injury | OSM_ONEWAY | 0.089118 | 0.146553 | 0.257255 | 0.110702 |
| road_to_injury | OSM_TYPE | 0.088816 | 0.153686 | 0.252569 | 0.098883 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.088699 | 0.140603 | 0.232109 | 0.091507 |
| road_to_injury | INFERRED_LANES | 0.089288 | 0.130987 | 0.246052 | 0.115066 |
| vehicle_to_injury | is_sedan | 0.08859 | 0.015619 | 0.951568 | 0.967186 |
| vehicle_to_injury | is_suv | 0.089371 | 0.021786 | 0.949744 | 0.927958 |
| vehicle_to_injury | is_taxi | 0.088679 | 0.021486 | 0.938913 | 0.917427 |
| vehicle_to_injury | is_truck | 0.089759 | 0.017274 | 0.943197 | 0.925923 |
| vehicle_to_injury | is_bus | 0.088694 | 0.012982 | 0.93983 | 0.952812 |
| vehicle_to_injury | is_motorcycle | 0.086622 | 0.015015 | 0.95478 | 0.939765 |
| vehicle_to_injury | is_bicycle | 0.089907 | 0.038915 | 0.992318 | 1.031233 |
| vehicle_to_injury | is_other_vehicle | 0.089563 | 0.027353 | 0.886087 | 0.91344 |
| crash_type_to_injury | is_distracted | 0.08972 | 0.169936 | 3.38029 | 3.210353 |
| crash_type_to_injury | is_speeding | 0.088238 | 0.145698 | 3.443506 | 3.297808 |
| crash_type_to_injury | is_failure_to_yield | 0.086988 | 0.179535 | 3.436855 | 3.25732 |
| crash_type_to_injury | is_following_too_closely | 0.089646 | 0.165594 | 3.399603 | 3.234009 |
| crash_type_to_injury | is_drunk_driving | 0.089501 | 0.177867 | 3.34748 | 3.169613 |
| crash_type_to_injury | is_fatigue | 0.091349 | 0.296304 | 2.747656 | 2.451352 |
| crash_type_to_injury | is_view_obstructed | 0.090188 | 0.298422 | 3.222125 | 2.923703 |
| crash_type_to_injury | is_vehicle_defect | 0.089655 | 0.264879 | 3.157054 | 2.892175 |
| crash_type_to_injury | is_backing_unsafely | 0.088612 | 0.170381 | 3.347894 | 3.177512 |
| crash_type_to_injury | is_pedestrian_related | 0.090786 | 0.137781 | 3.137482 | 2.999701 |
| crash_type_to_injury | is_inexperience | 0.089586 | 0.264843 | 3.428328 | 3.163485 |
| crash_type_to_injury | is_pavement_slippery | 0.089504 | 0.234598 | 2.887161 | 2.652563 |
