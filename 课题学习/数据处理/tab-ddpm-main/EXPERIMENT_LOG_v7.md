# EXPERIMENT LOG - v7 Two-Stage CausalDDPM

## Phase A: Code Fixes
- Fixed static type issues in `scripts/evaluate_v7.py` by explicit numpy conversion before arithmetic.
- Fixed static type issues in `scripts/sample_two_stage_v7.py` for optional pandas series, context encoding ndarray conversion, and diffusion internal API typing.

## Phase B: Main Artifacts
- Model dir: `exp/nyc_crash_v7/causal_m4_v7`
- Synthetic csv: `exp/nyc_crash_v7/causal_m4_v7/synthetic_2017_v7.csv`
- Eval json: `exp/nyc_crash_v7/causal_m4_v7/eval_v7.json`
- Figure dir: `exp/nyc_crash_v7/causal_m4_v7/figures`

## Phase C: Key Metrics
- logic_violation_rate: `0.0`
- numeric_drift_avg: `0.005146594089725813`
- categorical_tv_avg: `0.00422069017575726`

### Sparse TSTR Summary
| target | tstr_macro_f1 | real_macro_f1 | fidelity_ratio | tstr_accuracy |
|---|---:|---:|---:|---:|
| NUMBER OF PEDESTRIANS INJURED | 0.1632 | 0.2526 | 0.6461 | 0.9591 |
| NUMBER OF PEDESTRIANS KILLED | 0.4999 | 0.4999 | 1.0000 | 0.9995 |
| NUMBER OF CYCLIST INJURED | 0.3298 | 0.5919 | 0.5571 | 0.9788 |
| NUMBER OF CYCLIST KILLED | 0.5000 | 0.5000 | 1.0000 | 0.9999 |
| NUMBER OF MOTORIST INJURED | 0.1042 | 0.1069 | 0.9745 | 0.8744 |
| NUMBER OF MOTORIST KILLED | 0.4999 | 0.4999 | 1.0000 | 0.9996 |

## Phase D: Figures
- Loss curve: `exp/nyc_crash_v7/causal_m4_v7/figures/loss_curve_v7.png`
- Metric heatmap: `exp/nyc_crash_v7/causal_m4_v7/figures/sparse_tstr_heatmap_v7.png`
- Confusion matrices: `exp/nyc_crash_v7/causal_m4_v7/figures/confusion_matrix_tstr_v7.png`

## Notes
- Figures are saved separately in `figures/` as required.
- Log follows the same phased style as the v6 experiment log.