# Macro Relation Report: transfer_2025_macro_sparse_relative_g05_uniform5k_physical.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.036589 | 0.037547 | 0.45197 | 1.042467 |
| road_to_injury | 5 | 0.03832 | 0.041724 | 0.191462 | 0.207302 |
| vehicle_to_injury | 8 | 0.03749 | 0.040304 | 0.240343 | 0.266199 |
| weather_to_injury | 4 | 0.037275 | 0.040046 | 0.59702 | 0.902754 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.040046 | 0.367111 | 4.263275 | 3.896164 |
| weather_to_injury | TEMP_C | 0.036127 | 0.854683 | 1.063031 | 0.208348 |
| weather_to_injury | prcp | 0.036551 | 0.26353 | 4.000739 | 3.737209 |
| weather_to_injury | WIND_SPEED_KMH | 0.036377 | 0.902754 | 1.087664 | 0.18491 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.037959 | 0.207302 | 0.311209 | 0.103907 |
| road_to_injury | OSM_ONEWAY | 0.036357 | 0.199261 | 0.296363 | 0.097102 |
| road_to_injury | OSM_TYPE | 0.041724 | 0.167596 | 0.273087 | 0.105491 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.037649 | 0.197056 | 0.268872 | 0.071816 |
| road_to_injury | INFERRED_LANES | 0.037911 | 0.186095 | 0.266506 | 0.080411 |
| vehicle_to_injury | is_sedan | 0.036672 | 0.266199 | 0.94257 | 0.676371 |
| vehicle_to_injury | is_suv | 0.036437 | 0.24855 | 0.935906 | 0.687356 |
| vehicle_to_injury | is_taxi | 0.035351 | 0.246389 | 0.931512 | 0.685123 |
| vehicle_to_injury | is_truck | 0.040304 | 0.234753 | 0.932641 | 0.697888 |
| vehicle_to_injury | is_bus | 0.036887 | 0.240753 | 0.931513 | 0.69076 |
| vehicle_to_injury | is_motorcycle | 0.037125 | 0.240593 | 0.937216 | 0.696622 |
| vehicle_to_injury | is_bicycle | 0.040121 | 0.255391 | 0.977089 | 0.721698 |
| vehicle_to_injury | is_other_vehicle | 0.037025 | 0.190119 | 0.880617 | 0.690499 |
| crash_type_to_injury | is_distracted | 0.037038 | 0.328396 | 3.386048 | 3.057652 |
| crash_type_to_injury | is_speeding | 0.036092 | 0.324441 | 3.4324 | 3.107958 |
| crash_type_to_injury | is_failure_to_yield | 0.034735 | 0.305498 | 3.46194 | 3.156442 |
| crash_type_to_injury | is_following_too_closely | 0.036622 | 0.360804 | 3.419678 | 3.058873 |
| crash_type_to_injury | is_drunk_driving | 0.037547 | 0.361478 | 3.376756 | 3.015278 |
| crash_type_to_injury | is_fatigue | 0.036871 | 1.042467 | 2.971126 | 1.928659 |
| crash_type_to_injury | is_view_obstructed | 0.036584 | 0.434261 | 3.14091 | 2.706649 |
| crash_type_to_injury | is_vehicle_defect | 0.036894 | 0.325473 | 3.231173 | 2.9057 |
| crash_type_to_injury | is_backing_unsafely | 0.035813 | 0.292421 | 3.300355 | 3.007934 |
| crash_type_to_injury | is_pedestrian_related | 0.037536 | 0.379165 | 3.112658 | 2.733493 |
| crash_type_to_injury | is_inexperience | 0.03672 | 0.322351 | 3.409072 | 3.086722 |
| crash_type_to_injury | is_pavement_slippery | 0.03662 | 0.946887 | 3.025522 | 2.078635 |
