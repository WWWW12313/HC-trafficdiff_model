"""
自动生成论文对比实验的 YAML 配置，写入 configs/experiments/。

用法:
  python src/generate_experiment_configs.py
"""

from __future__ import annotations

import os
from pathlib import Path

CDT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CDT_ROOT / "configs" / "experiments"

EXPERIMENTS = [
    {
        "filename": "baseline_tabddpm.yaml",
        "model_name": "baseline_tabddpm",
        "experiment_id": "baseline_tabddpm",
        "lambda_causal": 0.0,
        "use_causal_masks": False,
        "hierarchical": False,
        "description": "无因果掩码、无分层：仅 Stage3 全表无条件生成（TabDDPM 式基线）",
        "sampling": {
            "mode": "unconditional",
            "num_samples": None,
        },
    },
    {
        "filename": "ablation_no_causal.yaml",
        "model_name": "ablation_no_causal",
        "experiment_id": "ablation_no_causal",
        "lambda_causal": 0.0,
        "use_causal_masks": False,
        "hierarchical": True,
        "description": "分层 + Inpainting，但关闭因果（无 NOTEARS 掩码、lambda_causal=0）",
        "sampling": {
            "mode": "impute_stage3",
            "num_samples": None,
            "impute_indices_count": 2000,
        },
    },
    {
        "filename": "ablation_no_hierarchy.yaml",
        "model_name": "ablation_no_hierarchy",
        "experiment_id": "ablation_no_hierarchy",
        "lambda_causal": 1.0,
        "use_causal_masks": True,
        "hierarchical": False,
        "description": "因果掩码 + 正则，但无分层：仅 Stage3 全 47 维无条件一步生成",
        "sampling": {
            "mode": "unconditional",
            "num_samples": None,
        },
    },
    {
        "filename": "ours_full_model.yaml",
        "model_name": "ours_full_model",
        "experiment_id": "ours_full_model",
        "lambda_causal": 1.0,
        "use_causal_masks": True,
        "hierarchical": True,
        "description": "完全体：动态因果掩码 + 分层 Inpainting（Stage1+Stage3）",
        "sampling": {
            "mode": "impute_stage3",
            "num_samples": None,
            "impute_indices_count": 2000,
        },
    },
]


def _dump_yaml(data: dict) -> str:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "需要 PyYAML: 在当前环境中执行  python -m pip install pyyaml"
        ) from e
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        body = {k: v for k, v in exp.items() if k != "filename"}
        text = "# CausalDiffTab experiment config (auto-generated)\n"
        text += _dump_yaml(body)
        path = OUT_DIR / exp["filename"]
        path.write_text(text, encoding="utf-8")
        print(f"[write] {path}")
    print(f"[done] {len(EXPERIMENTS)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
