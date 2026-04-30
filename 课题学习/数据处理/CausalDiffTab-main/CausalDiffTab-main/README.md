# CausalDiffTab (CDT) NYC Crash Project README

本 README 基于当前仓库真实状态与实验日志 `tab-ddpm-main/Experiment_Logs/2026-03-23.md`（Round 6-8）整理，重点面向论文对比实验自动化管线。

## 1. 项目位置

- CDT 根目录: `课题学习/数据处理/CausalDiffTab-main/CausalDiffTab-main`
- 实验日志: `课题学习/数据处理/tab-ddpm-main/Experiment_Logs/2026-03-23.md`

## 2. 关键改动概览

- 新增配置生成: `src/generate_experiment_configs.py`
- 新增统一训练采样 runner: `pipeline/run_all_experiments.py`
- 新增统一评估脚本: `pipeline/evaluate_all.py`
- 新增 4 份实验 YAML: `configs/experiments/*.yaml`
- 扩展训练入口: `src/train_hierarchical.py`
	- `--experiment_id`
	- `--lambda_causal`
	- `--no_causal_masks`

## 3. 目录结构（与当前落盘一致）

```text
CausalDiffTab-main/CausalDiffTab-main/
	configs/experiments/
		baseline_tabddpm.yaml
		ablation_no_causal.yaml
		ablation_no_hierarchy.yaml
		ours_full_model.yaml
	pipeline/
		run_all_experiments.py
		evaluate_all.py
	results/
		synthetic/               # 每组实验统一输出 {model}_{tier}.csv
		logs/                    # 长任务日志（如 full tier）
		_cache/                  # impute 索引等临时文件
		eval_report_*.md/json    # 带时间戳评估报告
		eval_report_latest.md/json
```

## 4. 环境与依赖

推荐在 `crashgen` 环境执行。

```bash
conda activate crashgen
python -m pip install pyyaml prdc xgboost scikit-learn scipy
```

说明:
- `pyyaml`: 读写实验 YAML
- `prdc`: 完整 TabMetrics 依赖
- `xgboost` + `scikit-learn`: TSTR
- `scipy`: Wasserstein / JS 距离

## 5. 四组实验配置语义

- `baseline_tabddpm`: 无因果 + 无分层，Stage3 无条件采样
- `ablation_no_causal`: 分层 + 无因果，Stage1+3 + `impute_stage3`
- `ablation_no_hierarchy`: 有因果 + 无分层，Stage3 无条件采样
- `ours_full_model`: 有因果 + 分层，Stage1+3 + `impute_stage3`

具体字段请以 `configs/experiments/*.yaml` 为准。

## 6. 一键管线命令

工作目录切到 CDT 根目录后执行。

```bash
python src/generate_experiment_configs.py

python pipeline/run_all_experiments.py --model baseline_tabddpm --tier balanced
python pipeline/run_all_experiments.py --model ablation_no_causal --tier balanced
python pipeline/run_all_experiments.py --model ablation_no_hierarchy --tier balanced
python pipeline/run_all_experiments.py --model ours_full_model --tier balanced

python pipeline/evaluate_all.py --real_test synthetic/nyc_crash/test.csv --file_glob "*_balanced.csv"
```

常用参数:
- `run_all_experiments.py`: `--skip_train`, `--skip_sample`, `--device`
- `evaluate_all.py`: `--file_glob`（如 `*_quick.csv`, `*_balanced.csv`, `*_full.csv`）

## 7. 评估指标定义

`pipeline/evaluate_all.py` 计算以下指标:
- TSTR Macro F1: Train on Synthetic, Test on Real
- `mean_wasserstein_numeric`: 数值列平均 Wasserstein
- `mean_js_categorical`: 分类列平均 Jensen-Shannon
- `logic_violation_rate`: 逻辑违背率

实现细节:
- 列名清洗 `_sanitize_feature_columns()`，规避 XGBoost 特殊字符列名问题
- TSTR 标签预处理（`to_numeric -> floor -> Int64 -> str`）避免 `"0"`/`"0.0"` 类别分裂
- 训练标签重映射到连续 `0..C-1`，满足 XGBoost 多分类要求
- 测试集仅保留合成训练集覆盖的类别后计算 Macro F1

## 8. 当前结果状态（按仓库实况）

已确认:
- `results/synthetic/` 已存在 4 组 `quick` CSV
- `results/synthetic/` 已存在 4 组 `balanced` CSV
- `results/eval_report_latest.md` 当前为 `*_balanced.csv` 评估结果

待确认:
- `results/logs/full_tier_pipeline_20260326_175729.log` 当前仅见启动头部，尚未看到 `DONE`
- `results/synthetic/` 目前未看到 `*_full.csv`

建议补跑/检查:

```bash
python pipeline/run_all_experiments.py --model ours_full_model --tier full
python pipeline/evaluate_all.py --real_test synthetic/nyc_crash/test.csv --file_glob "*_full.csv"
```

## 9. 与旧入口兼容性

`src/train_hierarchical.py` 维持向后兼容:
- 不传 `--experiment_id` 时，checkpoint 路径命名保持旧行为
- `--lambda_causal` 默认 `1.0`
- 不使用 `--no_causal_masks` 时保持加载 NOTEARS 掩码

## 10. 一句话摘要

当前 CDT 已具备论文对比实验自动化能力（配置生成、统一训练采样、批量评估），`balanced/quick` 已跑通并可复现报告，`full` 仍需在本机继续完成并补齐 `*_full.csv` 评估。
