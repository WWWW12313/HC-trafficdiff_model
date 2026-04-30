"""
Zero-shot 实验辅助脚本：仅生成 ours_full_model 的 unconditional 对照样本。

目的（严格 2017-only）：
  默认 sampling.mode=impute_stage3 把 Stage1+2（lat/lon/time/weather/OSM）锁定到 2000 行 2017
  训练样本，导致 T17R_TRAIN ↔ T_SYN 的 A_time/C_geo 拟合度虚低、T25R ↔ T_SYN 的同组
  指标被 2017 强制偏置，从而放大退化率。该脚本复用同一 2017 checkpoint，仅切换到
  unconditional 采样，得到“真正由 2017 训练扩散模型生成”的 T_SYN_uncond，便于诚实评估
  zero-shot 可迁移性。

不接触任何 2025 信息：仅用 ckpt + 2017 数据 schema。

用法：
  python pipeline/_zs_sample_unconditional.py \
      --model ours_full_model --tier full --num_samples 10000 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ours_full_model")
    ap.add_argument("--tier", default="full", choices=["quick", "balanced", "full"])
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="zsuncond",
                    help="output filename suffix; final csv: _{model}_{tier}_{tag}_samples_physical.csv")
    args = ap.parse_args()

    from src.sample_conditional import run_sampling

    ckpt_dir = ROOT / "ckpt" / "nyc_crash" / f"stage3_full_{args.tier}_{args.model}"
    if not (ckpt_dir / "config.pkl").is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_dir}")

    out_dir = ROOT / "results" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / f"_{args.model}_{args.tier}_{args.tag}_samples.csv"

    print(f"[zs-uncond] ckpt={ckpt_dir}")
    print(f"[zs-uncond] output={raw_csv} (+ _physical)")

    run_sampling(
        ckpt_dir=str(ckpt_dir),
        data_dir=str(ROOT / "data" / "nyc_crash"),
        condition_train_indices=None,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_csv=str(raw_csv),
        do_postprocess=True,
        road_graphml=str(ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml")
        if (ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml").is_file() else None,
        road_signals=None,
        snap_max_dist_m=300.0,
        recompute_osm_after_snap=True,
    )
    print("[zs-uncond] done.")


if __name__ == "__main__":
    main()
