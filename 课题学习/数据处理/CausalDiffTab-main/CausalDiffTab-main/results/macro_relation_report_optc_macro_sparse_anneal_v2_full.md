# Macro Relation Report: macro_sparse_anneal_v2_full.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| crash_type_to_injury | 12 | 0.104664 | 0.106448 | 0.11803 | 0.216521 |
| road_to_injury | 5 | 0.104099 | 0.104503 | 0.144578 | 0.160872 |
| vehicle_to_injury | 8 | 0.104113 | 0.105123 | 0.024114 | 0.03985 |
| weather_to_injury | 4 | 0.104676 | 0.104765 | 0.240853 | 0.439063 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.104691 | 0.021576 | 4.278674 | 4.257098 |
| weather_to_injury | TEMP_C | 0.104507 | 0.439063 | 1.135817 | 0.696754 |
| weather_to_injury | prcp | 0.104741 | 0.082737 | 3.981275 | 3.898538 |
| weather_to_injury | WIND_SPEED_KMH | 0.104765 | 0.420036 | 1.003119 | 0.583083 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.103715 | 0.160872 | 0.288408 | 0.127536 |
| road_to_injury | OSM_ONEWAY | 0.104335 | 0.139197 | 0.257255 | 0.118057 |
| road_to_injury | OSM_TYPE | 0.104025 | 0.150789 | 0.252569 | 0.101779 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.103918 | 0.135092 | 0.232109 | 0.097017 |
| road_to_injury | INFERRED_LANES | 0.104503 | 0.13694 | 0.246052 | 0.109112 |
| vehicle_to_injury | is_sedan | 0.103802 | 0.026323 | 0.951568 | 0.925245 |
| vehicle_to_injury | is_suv | 0.104587 | 0.03985 | 0.949744 | 0.909894 |
| vehicle_to_injury | is_taxi | 0.103895 | 0.015646 | 0.938913 | 0.923267 |
| vehicle_to_injury | is_truck | 0.104974 | 0.023411 | 0.943197 | 0.919785 |
| vehicle_to_injury | is_bus | 0.103907 | 0.009367 | 0.93983 | 0.930463 |
| vehicle_to_injury | is_motorcycle | 0.101835 | 0.018006 | 0.95478 | 0.936775 |
| vehicle_to_injury | is_bicycle | 0.105123 | 0.028067 | 0.992318 | 0.964251 |
| vehicle_to_injury | is_other_vehicle | 0.104779 | 0.032246 | 0.886087 | 0.918334 |
| crash_type_to_injury | is_distracted | 0.104936 | 0.073777 | 3.38029 | 3.306513 |
| crash_type_to_injury | is_speeding | 0.103459 | 0.079864 | 3.443506 | 3.363642 |
| crash_type_to_injury | is_failure_to_yield | 0.102203 | 0.088129 | 3.436855 | 3.348726 |
| crash_type_to_injury | is_following_too_closely | 0.104862 | 0.07235 | 3.399603 | 3.327252 |
| crash_type_to_injury | is_drunk_driving | 0.104717 | 0.083296 | 3.34748 | 3.264183 |
| crash_type_to_injury | is_fatigue | 0.106448 | 0.216521 | 2.747656 | 2.531136 |
| crash_type_to_injury | is_view_obstructed | 0.10512 | 0.198662 | 3.222125 | 3.023463 |
| crash_type_to_injury | is_vehicle_defect | 0.104871 | 0.161008 | 3.157054 | 2.996046 |
| crash_type_to_injury | is_backing_unsafely | 0.103828 | 0.07009 | 3.347894 | 3.277804 |
| crash_type_to_injury | is_pedestrian_related | 0.106004 | 0.062577 | 3.137482 | 3.074905 |
| crash_type_to_injury | is_inexperience | 0.104802 | 0.165843 | 3.428328 | 3.262485 |
| crash_type_to_injury | is_pavement_slippery | 0.10472 | 0.144247 | 2.887161 | 2.742913 |
