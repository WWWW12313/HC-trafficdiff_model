# tab-ddpm-main 清理记录（2026-03-05）

## 已执行
- 将根目录散落的历史产物移动到 `artifacts_legacy/`：
  - `causal_ddpm_best.pt`
  - `causal_ddpm_final.pt`
  - `causal_ddpm_epoch_*.pt`
  - `training_loss_log.csv`
  - `synthetic_nyc_accidents.csv`
  - `clean_synthetic_accidents.csv`
  - `catboost_info/`
- 删除根目录 `__pycache__/`

## 保留不动（当前主链路）
- `runs/`
- `exp/`
- `data/`
- `scripts/`
- `tab_ddpm/`
- `train_causal_yandex.py`
- `causal_scaler.pkl`

## 目的
- 降低根目录噪声，避免历史 checkpoint 与当前实验产物混淆。
- 不改变现有训练与评估入口，确保可复现性。
