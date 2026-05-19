# Macro Relation Report: ablation_no_causal_full.csv

## Summary
| relation | n_specs | mean_group_mae | max_group_mae | mean_cmi_abs_error | max_cmi_abs_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| road_to_injury | 5 | 0.078573 | 0.081753 | 2.76005 | 2.796046 |
| vehicle_to_injury | 7 | 0.080354 | 0.081551 | 0.008615 | 0.017006 |
| weather_to_injury | 4 | 0.081797 | 0.082266 | 0.659856 | 1.363355 |

## Per Feature
| relation | parent_col | group_mae | cmi_abs_error | real_cmi | syn_cmi |
| --- | --- | ---: | ---: | ---: | ---: |
| weather_to_injury | WEATHER_CONDITION | 0.081743 | 0.038956 | 4.278674 | 4.239718 |
| weather_to_injury | TEMP_C | 0.082266 | 1.180575 | 1.135817 | 2.316392 |
| weather_to_injury | prcp | 0.081441 | 0.056537 | 3.981275 | 3.924738 |
| weather_to_injury | WIND_SPEED_KMH | 0.08174 | 1.363355 | 1.003119 | 2.366474 |
| road_to_injury | HAS_TRAFFIC_SIGNAL | 0.075661 | 2.720606 | 0.288408 | 3.009014 |
| road_to_injury | OSM_ONEWAY | 0.078048 | 2.751986 | 0.257255 | 3.00924 |
| road_to_injury | OSM_TYPE | 0.081753 | 2.748511 | 0.252569 | 3.00108 |
| road_to_injury | DIST_TO_SIGNAL_M | 0.078514 | 2.7831 | 0.232109 | 3.015209 |
| road_to_injury | INFERRED_LANES | 0.078891 | 2.796046 | 0.246052 | 3.042098 |
| vehicle_to_injury | is_sedan | 0.081477 | 0.000809 | 0.951568 | 0.952377 |
| vehicle_to_injury | is_suv | 0.081551 | 0.002351 | 0.949744 | 0.952095 |
| vehicle_to_injury | is_taxi | 0.080764 | 0.013707 | 0.938913 | 0.95262 |
| vehicle_to_injury | is_truck | 0.081443 | 0.017006 | 0.943197 | 0.960202 |
| vehicle_to_injury | is_bus | 0.081169 | 0.008555 | 0.93983 | 0.948385 |
| vehicle_to_injury | is_motorcycle | 0.080731 | 0.007722 | 0.95478 | 0.962503 |
| vehicle_to_injury | is_bicycle | 0.07534 | 0.010156 | 0.992318 | 0.982162 |
