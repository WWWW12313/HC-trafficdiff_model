# Transfer (2025) Macro Guidance Mode Comparison

## Overview
All experiments use the same base model: `macro_sparse_anneal_v2` trained on 2024 data.
Sampling is done on 2025 test conditions (uniform 5k) with different guidance modes.

## Results

| Mode | Crash | Road | Vehicle | Weather | **macroMAE** | vs Baseline |
|------|-------|------|---------|---------|-------------|-------------|
| **no_guidance** | 0.0301 | 0.0310 | 0.0300 | 0.0296 | **0.0302** | — |
| annealed_g05 | 0.0316 | 0.0327 | 0.0319 | 0.0342 | 0.0326 | +8.0% |
| adaptive_g05 | 0.0349 | 0.0367 | 0.0345 | 0.0346 | 0.0352 | +16.5% |
| relative_g05 | 0.0366 | 0.0383 | 0.0375 | 0.0373 | 0.0374 | +24.0% |
| absolute_g05 | 0.0548 | 0.0565 | 0.0570 | 0.0546 | 0.0557 | +84.7% |

## Key Findings

1. **No-guidance baseline is best for transfer.** The model trained with causal structure
   (`macro_sparse_anneal_v2`) already encodes macro relations well enough that inference-time
   guidance does not help when transferring to 2025.

2. **All guidance modes hurt transfer, but by different amounts.**
   - `absolute`: Catastrophic (+84.7%) due to distribution drift in rare groups.
   - `relative`: Moderate hurt (+24.0%) — tanh-saturation helps but still over-corrects.
   - `adaptive`: Better (+16.5%) — exponential decay based on drift reduces impact.
   - `annealed`: Best among guidance modes (+8.0%) — linear decay to zero by end of denoising
     causes least disturbance.

3. **Root cause**: 2024 pre-computed group means mismatch 2025 distribution for rare groups
   (macro relation mean abs diff = 0.236, max = 2.8). Any attempt to push synthetic samples
   toward stale 2024 targets introduces error.

## Recommendation

For **transfer to 2025**, use `macro_sparse_anneal_v2` **without macro guidance**.

For **in-domain (2024) generation**, `absolute` guidance with scale=0.5 is validated as
beneficial (macroMAE 0.0297 vs 0.0391 without guidance).

## Next Steps

- Run `finetune_2025.py` to adapt model weights to 2025 distribution, then re-evaluate
  whether guidance becomes useful after fine-tuning.
- Consider computing group means dynamically from 2025 data at sampling time (if real 2025
  reference is available) rather than using 2024 pre-computed values.
