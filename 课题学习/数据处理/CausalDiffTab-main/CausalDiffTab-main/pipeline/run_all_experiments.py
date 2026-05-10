"""
统一实验管线：按 YAML 训练 Stage1/3、采样并写入 results/synthetic/{model}_{tier}.csv

示例:
  python pipeline/run_all_experiments.py --model ours_full_model --tier balanced
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CDT_ROOT = Path(__file__).resolve().parent.parent

# 与 src/train_hierarchical.TIER_SETTINGS 对齐（用于默认采样条数）
TIER_SAMPLE_DEFAULT = {
    "quick": 500,
    "balanced": 2000,
    "full": 10000,
}


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("请安装 PyYAML: python -m pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_train_stage(
    stage: int,
    tier: str,
    experiment_id: str,
    lambda_causal: float,
    use_causal_masks: bool,
    device: str,
    dataname: str,
) -> None:
    cmd = [
        sys.executable,
        str(CDT_ROOT / "src" / "train_hierarchical.py"),
        "--stage",
        str(stage),
        "--tier",
        tier,
        "--experiment_id",
        experiment_id,
        "--lambda_causal",
        str(lambda_causal),
        "--device",
        device,
    ]
    cmd.extend(["--dataname", dataname])
    if not use_causal_masks:
        cmd.append("--no_causal_masks")
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(CDT_ROOT))


def _default_num_samples(cfg: dict, tier: str) -> int:
    s = cfg.get("sampling") or {}
    n = s.get("num_samples")
    if n is not None:
        return int(n)
    return int(TIER_SAMPLE_DEFAULT.get(tier, 2000))


def _write_impute_indices_file(count: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(str(i) for i in range(max(1, count)))
    out_path.write_text(lines + "\n", encoding="utf-8")


def _infer_train_rows(data_dir: Path) -> int:
    xnum = data_dir / "X_num_train.npy"
    xcat = data_dir / "X_cat_train.npy"
    for p in [xnum, xcat]:
        if p.is_file():
            try:
                arr = np.load(p, mmap_mode="r")
                return int(arr.shape[0])
            except Exception:
                continue
    return 0


def _infer_year_from_dataname(dataname: str) -> int | None:
    tail = str(dataname).rsplit("_", 1)[-1]
    if tail.isdigit() and len(tail) == 4:
        return int(tail)
    return None


def _write_random_impute_indices_file(count: int, train_rows: int, seed: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if train_rows <= 0:
        _write_impute_indices_file(count, out_path)
        return

    n = max(1, int(count))
    rng = np.random.RandomState(int(seed))
    replace = n > train_rows
    idx = rng.choice(train_rows, size=n, replace=replace)
    lines = "\n".join(str(int(i)) for i in idx.tolist())
    out_path.write_text(lines + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CausalDiffTab 自动化训练+采样管线")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "baseline_tabddpm",
            "ablation_no_causal",
            "ablation_no_hierarchy",
            "ours_full_model",
            "ours_stage2_causal",
            "our_model_no_h3",
            "macro_soft_2024",
            "our_model",
        ],
        help="与 configs/experiments/*.yaml 中 model_name 一致",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "full"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="跳过训练，仅采样（需已有对应 checkpoint）",
    )
    parser.add_argument(
        "--skip_sample",
        action="store_true",
        help="仅训练，不采样",
    )
    parser.add_argument(
        "--impute_index_seed",
        type=int,
        default=42,
        help="impute_stage3 采样时训练行索引的随机种子（避免总是取前 N 行）",
    )
    parser.add_argument(
        "--debug_compare_unconditional",
        action="store_true",
        help="若当前模式为 impute_stage3，额外导出一份 unconditional 对照样本",
    )
    parser.add_argument(
        "--dataname",
        type=str,
        default="nyc_crash",
        help="Stage 3 训练/采样数据目录名，位于 data/ 下，例如 nyc_crash_2024",
    )
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default=None,
        help="合成结果输出目录；默认 results/synthetic",
    )
    parser.add_argument("--road_graphml", type=str, default=None,
                        help="OSM graphml 路径，启用路网 snap")
    parser.add_argument("--road_signals", type=str, default=None,
                        help="信号灯 geojson 路径（可选）")
    parser.add_argument("--snap_max_dist_m", type=float, default=300.0)
    parser.add_argument("--no_recompute_osm", action="store_true")
    args = parser.parse_args()

    cfg_path = CDT_ROOT / "configs" / "experiments" / f"{args.model}.yaml"
    if not cfg_path.is_file():
        raise SystemExit(
            f"未找到 {cfg_path}，请先运行: python src/generate_experiment_configs.py"
        )

    cfg = _load_yaml(cfg_path)
    experiment_id = cfg["experiment_id"]
    lambda_causal = float(cfg.get("lambda_causal", 1.0))
    use_causal_masks = bool(cfg.get("use_causal_masks", True))
    hierarchical = bool(cfg.get("hierarchical", False))
    train_stage2 = bool(cfg.get("train_stage2", False))
    sampling = cfg.get("sampling") or {}
    mode = sampling.get("mode", "unconditional")

    syn_dir = Path(args.synthetic_dir) if args.synthetic_dir else CDT_ROOT / "results" / "synthetic"
    if not syn_dir.is_absolute():
        syn_dir = CDT_ROOT / syn_dir
    syn_dir.mkdir(parents=True, exist_ok=True)
    final_csv = syn_dir / f"{args.model}_{args.tier}.csv"

    if not args.skip_train:
        if hierarchical:
            _run_train_stage(
                1, args.tier, experiment_id, lambda_causal, use_causal_masks, args.device, args.dataname
            )
        if train_stage2:
            _run_train_stage(
                2, args.tier, experiment_id, lambda_causal, use_causal_masks, args.device, args.dataname
            )
        _run_train_stage(
            3, args.tier, experiment_id, lambda_causal, use_causal_masks, args.device, args.dataname
        )

    if args.skip_sample:
        print("[skip_sample] 训练结束，未生成合成 CSV")
        return

    if mode == "chain_stage123":
        stage1_dataname = args.dataname.replace("nyc_crash", "nyc_stage1", 1) if args.dataname.startswith("nyc_crash_") else "nyc_stage1"
        stage2_dataname = args.dataname.replace("nyc_crash", "nyc_stage2", 1) if args.dataname.startswith("nyc_crash_") else "nyc_stage2"
        stage1_ckpt_dir = CDT_ROOT / "ckpt" / stage1_dataname / f"stage1_spatial_{args.tier}_{experiment_id}"
        stage2_ckpt_dir = CDT_ROOT / "ckpt" / stage2_dataname / f"stage2_context_{args.tier}_{experiment_id}"
        stage3_ckpt_dir = CDT_ROOT / "ckpt" / args.dataname / f"stage3_full_{args.tier}_{experiment_id}"
        stage1_data_root = CDT_ROOT / "data" / stage1_dataname
        stage2_data_root = CDT_ROOT / "data" / stage2_dataname
        data_root = CDT_ROOT / "data" / args.dataname
        impute_stage = "stage3"
        for ckpt_dir in [stage1_ckpt_dir, stage2_ckpt_dir, stage3_ckpt_dir]:
            if not (ckpt_dir / "config.pkl").is_file():
                raise SystemExit(f"缺少 checkpoint 目录或 config.pkl: {ckpt_dir}")
    elif mode == "impute_stage2":
        stage2_dataname = args.dataname.replace("nyc_crash", "nyc_stage2", 1) if args.dataname.startswith("nyc_crash_") else "nyc_stage2"
        ckpt_dir = CDT_ROOT / "ckpt" / stage2_dataname / f"stage2_context_{args.tier}_{experiment_id}"
        data_root = CDT_ROOT / "data" / stage2_dataname
        impute_stage = "stage2"
    else:
        ckpt_dir = CDT_ROOT / "ckpt" / args.dataname / f"stage3_full_{args.tier}_{experiment_id}"
        data_root = CDT_ROOT / "data" / args.dataname
        impute_stage = "stage3"
    if mode != "chain_stage123" and not (ckpt_dir / "config.pkl").is_file():
        raise SystemExit(f"缺少 checkpoint 目录或 config.pkl: {ckpt_dir}")

    num_samples = _default_num_samples(cfg, args.tier)
    data_dir = str(data_root)
    raw_out = syn_dir / f"_{args.model}_{args.tier}_samples.csv"

    sys.path.insert(0, str(CDT_ROOT))
    from src.sample_conditional import run_sampling, run_hierarchical_chain_sampling

    cond_indices = None
    if mode in {"impute_stage2", "impute_stage3"}:
        n_idx = int(sampling.get("impute_indices_count", num_samples))
        idx_file = CDT_ROOT / "results" / "_cache" / f"impute_base_{args.model}_{args.tier}.txt"
        train_rows = _infer_train_rows(data_root)
        _write_random_impute_indices_file(n_idx, train_rows, args.impute_index_seed, idx_file)
        print(
            f"[note] mode={mode}: 上游条件仍来自训练行索引；"
            "该设置用于 inpainting 质量评估，不等同于最终未来闭环生成。"
        )
        print(
            f"[impute] index_file={idx_file}, count={n_idx}, train_rows={train_rows}, seed={args.impute_index_seed}"
        )
        cond_indices = str(idx_file)

    if mode == "chain_stage123":
        run_hierarchical_chain_sampling(
            stage1_ckpt_dir=str(stage1_ckpt_dir),
            stage1_data_dir=str(stage1_data_root),
            stage2_ckpt_dir=str(stage2_ckpt_dir),
            stage2_data_dir=str(stage2_data_root),
            stage3_ckpt_dir=str(stage3_ckpt_dir),
            stage3_data_dir=str(data_root),
            num_samples=num_samples,
            batch_size=min(500, num_samples),
            device=args.device,
            output_csv=str(raw_out),
            do_postprocess=True,
            road_graphml=args.road_graphml,
            road_signals=args.road_signals,
            snap_max_dist_m=args.snap_max_dist_m,
            recompute_osm_after_snap=not args.no_recompute_osm,
        )
    else:
        run_sampling(
            ckpt_dir=str(ckpt_dir),
            data_dir=data_dir,
            condition_train_indices=cond_indices,
            num_samples=num_samples,
            batch_size=min(500, num_samples),
            device=args.device,
            output_csv=str(raw_out),
            do_postprocess=(mode != "impute_stage2"),
            road_graphml=args.road_graphml,
            road_signals=args.road_signals,
            snap_max_dist_m=args.snap_max_dist_m,
            recompute_osm_after_snap=not args.no_recompute_osm,
            impute_stage=impute_stage,
        )

    if mode == "impute_stage2":
        shutil.copy2(raw_out, final_csv)
        print(f"[done] Stage2 context samples: {final_csv}")
    else:
        physical = raw_out.with_name(raw_out.stem + "_physical" + raw_out.suffix)
        if not physical.is_file():
            raise SystemExit(f"后处理未生成 {physical}，请检查 postprocess_samples")
        shutil.copy2(physical, final_csv)
        print(f"[done] 合成数据: {final_csv}")

    target_year = _infer_year_from_dataname(args.dataname)
    if target_year is not None and mode != "impute_stage2":
        from export_raw_style_synthetic import export_raw_style

        raw_style_csv = syn_dir / f"{args.model}_{args.tier}_raw_style.csv"
        export_raw_style(
            input_csv=final_csv,
            output_csv=raw_style_csv,
            target_year=target_year,
            raw_csv=None,
            seed=args.impute_index_seed,
        )
        print(f"[done] 原始+API补全样式数据: {raw_style_csv}")

    if args.debug_compare_unconditional and mode == "impute_stage3":
        debug_raw = syn_dir / f"_{args.model}_{args.tier}_unconditional_debug_samples.csv"
        debug_final = syn_dir / f"{args.model}_{args.tier}_unconditional_debug.csv"
        print("[debug] 生成 unconditional 对照样本，用于解释 impute_stage3 的性能增益来源")
        run_sampling(
            ckpt_dir=str(ckpt_dir),
            data_dir=data_dir,
            condition_train_indices=None,
            num_samples=num_samples,
            batch_size=min(500, num_samples),
            device=args.device,
            output_csv=str(debug_raw),
            do_postprocess=True,
            road_graphml=args.road_graphml,
            road_signals=args.road_signals,
            snap_max_dist_m=args.snap_max_dist_m,
            recompute_osm_after_snap=not args.no_recompute_osm,
        )
        debug_physical = debug_raw.with_name(debug_raw.stem + "_physical" + debug_raw.suffix)
        if debug_physical.is_file():
            shutil.copy2(debug_physical, debug_final)
            print(f"[debug] unconditional 对照数据: {debug_final}")

    meta = {
        "model": args.model,
        "tier": args.tier,
        "mode": mode,
        "num_samples": int(num_samples),
        "dataname": args.dataname,
        "synthetic_dir": str(syn_dir),
        "impute_index_seed": int(args.impute_index_seed),
        "used_condition_indices": bool(cond_indices),
    }
    meta_path = syn_dir / f"_{args.model}_{args.tier}_sampling_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[meta] {meta_path}")


if __name__ == "__main__":
    main()
