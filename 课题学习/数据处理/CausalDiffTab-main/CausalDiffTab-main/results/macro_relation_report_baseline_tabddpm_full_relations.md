# Macro Relation Report: baseline_tabddpm_full.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| road_to_injury | 5 | 0.02442 | 0.033902 | 0.243829 | 0.264409 |
| vehicle_to_injury | 7 | 0.014425 | 0.015702 | 0.006724 | 0.009064 |
| weather_to_injury | 4 | 0.01864 | 0.027108 | 0.687986 | 1.092133 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.013997 | 0.527679 | 4.278674 | 3.750995 |
| weather_to_injury | TEMP_C | 0.027108 | 1.092133 | 1.135817 | 0.043685 |
| weather_to_injury | prcp | 0.013959 | 0.151949 | 3.981275 | 3.829327 |
| weather_to_injury | WIND_SPEED_KMH | 0.019497 | 0.980183 | 1.003119 | 0.022936 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.016902 | 0.264409 | 0.288408 | 0.023999 |
| road_to_injury | OSM_ONEWAY | 0.025912 | 0.236666 | 0.257255 | 0.020588 |
| road_to_injury | OSM_TYPE | 0.033902 | 0.250422 | 0.252569 | 0.002147 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.029491 | 0.22423 | 0.232109 | 0.007879 |
| road_to_injury | INFERRED_LANES | 0.015893 | 0.243418 | 0.246052 | 0.002635 |
| vehicle_to_injury | is_sedan | 0.013916 | 0.007629 | 0.951568 | 0.943938 |
| vehicle_to_injury | is_suv | 0.014089 | 0.005339 | 0.949744 | 0.944406 |
| vehicle_to_injury | is_taxi | 0.013558 | 0.008839 | 0.938913 | 0.947753 |
| vehicle_to_injury | is_truck | 0.013946 | 0.008576 | 0.943197 | 0.93462 |
| vehicle_to_injury | is_bus | 0.014789 | 0.004806 | 0.93983 | 0.944637 |
| vehicle_to_injury | is_motorcycle | 0.015702 | 0.002815 | 0.95478 | 0.957595 |
| vehicle_to_injury | is_bicycle | 0.014973 | 0.009064 | 0.992318 | 0.983254 |
