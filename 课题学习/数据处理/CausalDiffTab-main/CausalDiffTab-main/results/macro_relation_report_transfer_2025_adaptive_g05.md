# Macro Relation Report: transfer_2025_macro_sparse_adaptive_g05_uniform5k_physical.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.034869 | 0.037532 | 0.471772 | 1.707627 |
| road_to_injury | 5 | 0.036674 | 0.040835 | 0.187603 | 0.208801 |
| vehicle_to_injury | 8 | 0.034496 | 0.036806 | 0.21986 | 0.235017 |
| weather_to_injury | 4 | 0.03461 | 0.035344 | 0.603854 | 0.909422 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.035344 | 0.397363 | 4.263275 | 3.865912 |
| weather_to_injury | TEMP_C | 0.034071 | 0.864495 | 1.063031 | 0.198536 |
| weather_to_injury | prcp | 0.034551 | 0.244136 | 4.000739 | 3.756603 |
| weather_to_injury | WIND_SPEED_KMH | 0.034475 | 0.909422 | 1.087664 | 0.178242 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.035843 | 0.208801 | 0.311209 | 0.102408 |
| road_to_injury | OSM_ONEWAY | 0.034358 | 0.178536 | 0.296363 | 0.117827 |
| road_to_injury | OSM_TYPE | 0.040835 | 0.190358 | 0.273087 | 0.082729 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.036789 | 0.183803 | 0.268872 | 0.08507 |
| road_to_injury | INFERRED_LANES | 0.035544 | 0.176518 | 0.266506 | 0.089988 |
| vehicle_to_injury | is_sedan | 0.034311 | 0.231588 | 0.94257 | 0.710981 |
| vehicle_to_injury | is_suv | 0.034718 | 0.226461 | 0.935906 | 0.709445 |
| vehicle_to_injury | is_taxi | 0.033774 | 0.22506 | 0.931512 | 0.706452 |
| vehicle_to_injury | is_truck | 0.034823 | 0.220046 | 0.932641 | 0.712594 |
| vehicle_to_injury | is_bus | 0.033135 | 0.220349 | 0.931513 | 0.711164 |
| vehicle_to_injury | is_motorcycle | 0.033312 | 0.231905 | 0.937216 | 0.705311 |
| vehicle_to_injury | is_bicycle | 0.036806 | 0.235017 | 0.977089 | 0.742072 |
| vehicle_to_injury | is_other_vehicle | 0.035087 | 0.168451 | 0.880617 | 0.712166 |
| crash_type_to_injury | is_distracted | 0.035941 | 0.339673 | 3.386048 | 3.046375 |
| crash_type_to_injury | is_speeding | 0.034565 | 0.286616 | 3.4324 | 3.145783 |
| crash_type_to_injury | is_failure_to_yield | 0.031528 | 0.321974 | 3.46194 | 3.139966 |
| crash_type_to_injury | is_following_too_closely | 0.034674 | 0.328673 | 3.419678 | 3.091005 |
| crash_type_to_injury | is_drunk_driving | 0.034532 | 0.256814 | 3.376756 | 3.119942 |
| crash_type_to_injury | is_fatigue | 0.034668 | 0.95482 | 2.971126 | 2.016306 |
| crash_type_to_injury | is_view_obstructed | 0.03457 | 0.478918 | 3.14091 | 2.661991 |
| crash_type_to_injury | is_vehicle_defect | 0.034928 | 0.170152 | 3.231173 | 3.061021 |
| crash_type_to_injury | is_backing_unsafely | 0.035688 | 0.185899 | 3.300355 | 3.114456 |
| crash_type_to_injury | is_pedestrian_related | 0.037532 | 0.29898 | 3.112658 | 2.813678 |
| crash_type_to_injury | is_inexperience | 0.035328 | 0.331115 | 3.409072 | 3.077957 |
| crash_type_to_injury | is_pavement_slippery | 0.034476 | 1.707627 | 3.025522 | 1.317895 |
