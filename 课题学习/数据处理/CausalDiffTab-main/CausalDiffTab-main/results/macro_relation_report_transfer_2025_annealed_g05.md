# Macro Relation Report: transfer_2025_macro_sparse_annealed_g05_uniform5k_physical.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.031616 | 0.040877 | 0.468283 | 1.193011 |
| road_to_injury | 5 | 0.032666 | 0.037506 | 0.191197 | 0.213409 |
| vehicle_to_injury | 8 | 0.031918 | 0.035081 | 0.230101 | 0.253522 |
| weather_to_injury | 4 | 0.034168 | 0.044566 | 0.598322 | 0.891778 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.031141 | 0.366349 | 4.263275 | 3.896926 |
| weather_to_injury | TEMP_C | 0.044566 | 0.876599 | 1.063031 | 0.186432 |
| weather_to_injury | prcp | 0.030551 | 0.258564 | 4.000739 | 3.742176 |
| weather_to_injury | WIND_SPEED_KMH | 0.030413 | 0.891778 | 1.087664 | 0.195886 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.031859 | 0.213409 | 0.311209 | 0.0978 |
| road_to_injury | OSM_ONEWAY | 0.030414 | 0.181404 | 0.296363 | 0.11496 |
| road_to_injury | OSM_TYPE | 0.037506 | 0.182178 | 0.273087 | 0.090909 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.03163 | 0.202254 | 0.268872 | 0.066618 |
| road_to_injury | INFERRED_LANES | 0.031921 | 0.176738 | 0.266506 | 0.089768 |
| vehicle_to_injury | is_sedan | 0.029541 | 0.253522 | 0.94257 | 0.689048 |
| vehicle_to_injury | is_suv | 0.030587 | 0.235085 | 0.935906 | 0.700821 |
| vehicle_to_injury | is_taxi | 0.030441 | 0.237461 | 0.931512 | 0.694051 |
| vehicle_to_injury | is_truck | 0.033732 | 0.236884 | 0.932641 | 0.695756 |
| vehicle_to_injury | is_bus | 0.030299 | 0.223828 | 0.931513 | 0.707685 |
| vehicle_to_injury | is_motorcycle | 0.034657 | 0.226975 | 0.937216 | 0.710241 |
| vehicle_to_injury | is_bicycle | 0.035081 | 0.244811 | 0.977089 | 0.732278 |
| vehicle_to_injury | is_other_vehicle | 0.031008 | 0.182245 | 0.880617 | 0.698372 |
| crash_type_to_injury | is_distracted | 0.040877 | 0.377973 | 3.386048 | 3.008075 |
| crash_type_to_injury | is_speeding | 0.029948 | 0.324079 | 3.4324 | 3.108321 |
| crash_type_to_injury | is_failure_to_yield | 0.032088 | 0.400568 | 3.46194 | 3.061371 |
| crash_type_to_injury | is_following_too_closely | 0.030612 | 0.374705 | 3.419678 | 3.044973 |
| crash_type_to_injury | is_drunk_driving | 0.030349 | 0.312705 | 3.376756 | 3.064051 |
| crash_type_to_injury | is_fatigue | 0.030664 | 1.193011 | 2.971126 | 1.778114 |
| crash_type_to_injury | is_view_obstructed | 0.031085 | 0.372567 | 3.14091 | 2.768343 |
| crash_type_to_injury | is_vehicle_defect | 0.030657 | 0.354757 | 3.231173 | 2.876416 |
| crash_type_to_injury | is_backing_unsafely | 0.030154 | 0.384336 | 3.300355 | 2.916019 |
| crash_type_to_injury | is_pedestrian_related | 0.031405 | 0.224991 | 3.112658 | 2.887667 |
| crash_type_to_injury | is_inexperience | 0.030928 | 0.392995 | 3.409072 | 3.016078 |
| crash_type_to_injury | is_pavement_slippery | 0.030625 | 0.906711 | 3.025522 | 2.118811 |
