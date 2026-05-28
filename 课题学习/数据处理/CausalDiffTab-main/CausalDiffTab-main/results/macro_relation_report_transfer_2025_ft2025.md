# Macro Relation Report: transfer_2025_macro_sparse_anneal_v2_ft2025_uniform5k_physical.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.037554 | 0.040319 | 0.35064 | 0.77428 |
| road_to_injury | 5 | 0.039679 | 0.042712 | 0.180758 | 0.204512 |
| vehicle_to_injury | 8 | 0.038091 | 0.039033 | 0.245815 | 0.262232 |
| weather_to_injury | 4 | 0.037858 | 0.038357 | 0.592991 | 0.855005 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.038357 | 0.407765 | 4.263275 | 3.85551 |
| weather_to_injury | TEMP_C | 0.037368 | 0.844149 | 1.063031 | 0.218882 |
| weather_to_injury | prcp | 0.037951 | 0.265045 | 4.000739 | 3.735695 |
| weather_to_injury | WIND_SPEED_KMH | 0.037757 | 0.855005 | 1.087664 | 0.232659 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.039214 | 0.204512 | 0.311209 | 0.106697 |
| road_to_injury | OSM_ONEWAY | 0.037813 | 0.17799 | 0.296363 | 0.118373 |
| road_to_injury | OSM_TYPE | 0.042712 | 0.18782 | 0.273087 | 0.085268 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.039438 | 0.170691 | 0.268872 | 0.098181 |
| road_to_injury | INFERRED_LANES | 0.039216 | 0.162779 | 0.266506 | 0.103727 |
| vehicle_to_injury | is_sedan | 0.037988 | 0.260397 | 0.94257 | 0.682173 |
| vehicle_to_injury | is_suv | 0.037953 | 0.252259 | 0.935906 | 0.683647 |
| vehicle_to_injury | is_taxi | 0.037661 | 0.251464 | 0.931512 | 0.680048 |
| vehicle_to_injury | is_truck | 0.037915 | 0.239534 | 0.932641 | 0.693107 |
| vehicle_to_injury | is_bus | 0.039033 | 0.24572 | 0.931513 | 0.685792 |
| vehicle_to_injury | is_motorcycle | 0.037769 | 0.253705 | 0.937216 | 0.683511 |
| vehicle_to_injury | is_bicycle | 0.0384 | 0.262232 | 0.977089 | 0.714857 |
| vehicle_to_injury | is_other_vehicle | 0.038009 | 0.201207 | 0.880617 | 0.67941 |
| crash_type_to_injury | is_distracted | 0.038216 | 0.356661 | 3.386048 | 3.029387 |
| crash_type_to_injury | is_speeding | 0.035633 | 0.298039 | 3.4324 | 3.134361 |
| crash_type_to_injury | is_failure_to_yield | 0.03525 | 0.36158 | 3.46194 | 3.100359 |
| crash_type_to_injury | is_following_too_closely | 0.037961 | 0.352244 | 3.419678 | 3.067434 |
| crash_type_to_injury | is_drunk_driving | 0.038036 | 0.442356 | 3.376756 | 2.9344 |
| crash_type_to_injury | is_fatigue | 0.037806 | 0.161845 | 2.971126 | 2.80928 |
| crash_type_to_injury | is_view_obstructed | 0.037925 | 0.235384 | 3.14091 | 2.905526 |
| crash_type_to_injury | is_vehicle_defect | 0.040319 | 0.303928 | 3.231173 | 2.927245 |
| crash_type_to_injury | is_backing_unsafely | 0.035359 | 0.268975 | 3.300355 | 3.03138 |
| crash_type_to_injury | is_pedestrian_related | 0.037198 | 0.380719 | 3.112658 | 2.731939 |
| crash_type_to_injury | is_inexperience | 0.037904 | 0.271664 | 3.409072 | 3.137409 |
| crash_type_to_injury | is_pavement_slippery | 0.039037 | 0.77428 | 3.025522 | 2.251242 |
