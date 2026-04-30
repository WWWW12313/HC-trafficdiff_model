"""
run_traffic_structured.py — Phase 1 最小可行版本管线入口
=========================================================
实现 Section 2.1 第一阶段：

1. 对已有训练好的扩散模型生成的原始合成样本执行：
   a. 物理值还原（scaler 逆变换 + 分类解码）
   b. 道路约束经纬度后处理（road_snap 候选点吸附）
   c. 上下文补全（historical_lookup 天气 + OSM 属性）

2. 执行五大语义变量专项评估：
   a. 时间评估
   b. 空间评估（含道路距离）
   c. 车辆类型评估
   d. 事故原因评估
   e. 伤亡评估
   f. 路网/天气上下文评估

3. 与现有 baseline 对比（可选）

4. 输出 Markdown 评估报告

用法
----
# 完整 Phase 1 流程（生成 + snap + context + 评估）
python pipeline/run_traffic_structured.py \\
    --experiment_id macro_soft_2024 \\
    --year 2024 \\
    --context_mode historical_lookup \\
    --num_samples 10000

# 仅评估（已有合成样本）
python pipeline/run_traffic_structured.py \\
    --synth_csv results/synth_macro_soft_2024_n10000_physical.csv \\
    --real_csv  data/nyc_crash/test.csv \\
    --year 2024 \\
    --eval_only

# 2025 迁移评估
python pipeline/run_traffic_structured.py \\
    --synth_csv results/synth_macro_soft_2024_n10000_physical.csv \\
    --real_csv  data/nyc_crash_2025/test.csv \\
    --year 2025 \\
    --eval_only \\
    --out_report results/structured_eval_2025.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 1：采样 / 加载已有合成样本
# ──────────────────────────────────────────────────────────────────────────────

def _sample_or_load(args) -> Path:
    """返回原始合成样本 CSV 路径（已是物理值），如不存在则先采样。"""
    if args.synth_csv:
        p = Path(args.synth_csv)
        if p.exists():
            print(f"[run] 使用已有合成样本: {p}")
            return p
        else:
            print(f"[run] 指定的 synth_csv 不存在: {p}")

    # 自动推断路径
    exp_id = args.experiment_id or "macro_soft_2024"
    n      = args.num_samples
    raw_csv = ROOT / "results" / f"synth_{exp_id}_n{n}.csv"
    phy_csv = ROOT / "results" / f"synth_{exp_id}_n{n}_physical.csv"

    # 如果已有 physical CSV 直接用
    if phy_csv.exists():
        print(f"[run] 找到已有 physical CSV: {phy_csv}")
        return phy_csv

    # 先采样
    if not raw_csv.exists():
        ckpt_dir = _find_ckpt(exp_id, args)
        data_dir = args.data_dir or str(ROOT / "data" / "nyc_crash")
        print(f"[run] 采样 {n} 个样本 (ckpt={ckpt_dir}) ...")
        import subprocess
        cmd = [
            sys.executable,
            str(ROOT / "src" / "sample_conditional.py"),
            "--ckpt_dir",   str(ckpt_dir),
            "--data_dir",   str(data_dir),
            "--num_samples", str(n),
            "--device",     args.device,
            "--output_csv", str(raw_csv),
        ]
        ret = subprocess.run(cmd, check=True)
        if ret.returncode != 0:
            raise RuntimeError(f"采样失败: {ret}")

    return raw_csv


def _find_ckpt(exp_id: str, args) -> Path:
    """自动查找 stage3_full_balanced_{exp_id} 目录。"""
    explicit = getattr(args, "ckpt_dir", None)
    if explicit and Path(explicit).exists():
        return Path(explicit)
    candidates = [
        ROOT / "ckpt" / "nyc_crash" / f"stage3_full_balanced_{exp_id}",
        ROOT / "ckpt" / "nyc_crash" / f"stage3_full_full_{exp_id}",
        ROOT / "ckpt" / "nyc_crash" / exp_id,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"找不到 {exp_id} 的 checkpoint 目录，请用 --ckpt_dir 指定")


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 2：后处理（道路 snap + context_lookup）
# ──────────────────────────────────────────────────────────────────────────────

def _postprocess(raw_csv: Path, args) -> Path:
    """执行物理还原 + road snap + context_lookup，返回最终 CSV 路径。"""
    n      = args.num_samples
    exp_id = args.experiment_id or "macro_soft_2024"
    year   = args.year

    suffix = f"_snap_ctx{year}" if args.context_mode else "_snap"
    out_csv = ROOT / "results" / f"synth_{exp_id}_n{n}{suffix}.csv"

    if out_csv.exists() and not args.force_reprocess:
        print(f"[run] 后处理结果已存在: {out_csv}")
        return out_csv

    # 推断 OSM 路径
    osm_path = _find_osm_graphml(year)

    # 推断天气 CSV 路径
    weather_csv = args.weather_csv
    if weather_csv is None:
        w_dir = ROOT / "raw_data" / "weather" / str(year)
        w_files = list(w_dir.glob("open-meteo-*.csv")) if w_dir.exists() else []
        weather_csv = str(w_files[0]) if w_files else None

    from src.postprocess_samples import postprocess
    postprocess(
        samples_csv          = str(raw_csv),
        output_csv           = str(out_csv),
        road_graphml         = str(osm_path) if osm_path else None,
        snap_max_dist_m      = args.snap_max_dist_m,
        recompute_osm_after_snap = True,
        use_candidate_snap   = args.use_candidate_snap,
        road_snap_cache      = args.road_snap_cache,
        context_mode         = args.context_mode,
        weather_csv          = weather_csv,
        context_year         = year,
    )
    return out_csv


def _find_osm_graphml(year: int) -> Path | None:
    """按年份查 OSM graphml，若不存在回退到全局。"""
    yr_path = ROOT / "raw_data" / "osm" / str(year) / "nyc_drive_graph.graphml"
    if yr_path.exists():
        return yr_path
    global_path = ROOT / "raw_data" / "osm" / "nyc_drive_graph.graphml"
    if global_path.exists():
        return global_path
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 3：五大语义变量专项评估
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate(synth_csv: Path, real_csv: Path, args) -> dict:
    """执行专项评估，返回指标 dict。"""
    from src.structured_eval import eval_all, report_markdown

    print(f"\n[eval] 加载合成样本: {synth_csv.name}")
    print(f"[eval] 加载真实测试集: {real_csv}")
    synth_df = pd.read_csv(synth_csv)
    real_df  = pd.read_csv(real_csv)
    print(f"  合成: {len(synth_df)} 行 × {synth_df.shape[1]} 列")
    print(f"  真实: {len(real_df)} 行 × {real_df.shape[1]} 列")

    # 构建道路候选集（用于空间评估）
    rcs = None
    if not args.no_road_eval:
        osm_path = _find_osm_graphml(args.year)
        if osm_path:
            try:
                from src.road_snap import build_road_candidate_set
                print("[eval] 构建道路候选集（用于空间评估）...")
                rcs = build_road_candidate_set(
                    graphml_path=str(osm_path),
                    interp_step_m=100.0,  # 评估时用更粗的插值步长以加速
                    cache_path=args.road_snap_cache,
                    verbose=True,
                )
            except Exception as e:
                print(f"[eval] 道路候选集构建失败（跳过空间评估）: {e}")

    context_mode = args.context_mode or "historical_lookup"
    metrics = eval_all(real_df, synth_df, rcs=rcs, context_mode=context_mode)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 4：格式化输出
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(metrics: dict):
    """在控制台打印关键指标摘要。"""
    print("\n" + "=" * 65)
    print("=== Structured Evaluation Summary ===")
    print("=" * 65)

    # 时间
    t = metrics.get("time", {})
    print("\n[时间 - TimeDiff]")
    if "season_js" in t:
        print(f"  SEASON JS:          {t['season_js']:.4f}")
    for k in ["is_weekend", "is_am_peak", "is_pm_peak"]:
        v = t.get(k, {})
        if v:
            print(f"  {k.upper():<20} real={v.get('real_rate', 'N/A'):.3f}  "
                  f"synth={v.get('synth_rate', 'N/A'):.3f}  "
                  f"diff={v.get('rate_diff', 'N/A'):.3f}")
    if "both_peak_ratio" in t:
        print(f"  AM+PM同时为1比例:   {t['both_peak_ratio']:.4f}")

    # 空间
    s = metrics.get("spatial", {})
    print("\n[空间 - GeoRoadDiff]")
    for coord in ("latitude", "longitude"):
        wk = f"wasserstein_{coord}"
        if wk in s:
            print(f"  Wasserstein({coord.upper():<10}): {s[wk]:.6f}")
    rd_s = s.get("road_dist_synth", {})
    rd_r = s.get("road_dist_real", {})
    if rd_s:
        print(f"  合成点距道路(均值): {rd_s.get('mean_m', 'N/A'):.1f}m  "
              f"P95={rd_s.get('p95_m', 'N/A'):.1f}m  "
              f"落路率={rd_s.get('on_road_pct', 'N/A'):.1%}")
    if rd_r:
        print(f"  真实点距道路(均值): {rd_r.get('mean_m', 'N/A'):.1f}m  "
              f"P95={rd_r.get('p95_m', 'N/A'):.1f}m  "
              f"落路率={rd_r.get('on_road_pct', 'N/A'):.1%}")

    # 车辆
    v = metrics.get("vehicle", {})
    print("\n[车辆类型 - VehicleMultiLabel]")
    if "vehicle_jaccard_mean" in v:
        print(f"  多标签 Jaccard 均值: {v['vehicle_jaccard_mean']:.4f}")
    tv = v.get("total_vehicles", {})
    if tv:
        print(f"  TOTAL_VEHICLES:     real_mean={tv.get('real_mean', 'N/A'):.2f}  "
              f"synth_mean={tv.get('synth_mean', 'N/A'):.2f}  "
              f"MAE={tv.get('mae', 'N/A'):.3f}")

    # 原因
    c = metrics.get("cause", {})
    print("\n[事故原因 - CauseMultiLabel]")
    if "cause_cooccurrence_frobenius" in c:
        print(f"  共现矩阵 Frobenius: {c['cause_cooccurrence_frobenius']:.4f}")
    if "cause_vehicle_joint_js_mean" in c:
        print(f"  原因×车辆联合JS均值: {c['cause_vehicle_joint_js_mean']:.4f}")

    # 伤亡
    inj = metrics.get("injury", {})
    print("\n[伤亡 - InjuryCountSeverity]")
    if "injury_bin_js_mean" in inj:
        print(f"  BIN 列 JS 均值:    {inj['injury_bin_js_mean']:.4f}")
    ic = inj.get("injury_count", {})
    if ic:
        print(f"  NUMBER OF PERSONS INJURED:")
        print(f"    real_mean={ic.get('real_mean', 'N/A'):.3f}  "
              f"synth_mean={ic.get('synth_mean', 'N/A'):.3f}  "
              f"MAE={ic.get('mae', 'N/A'):.3f}")
        print(f"    Wasserstein={ic.get('wasserstein', 'N/A'):.4f}  "
              f"zero_diff={ic.get('zero_ratio_diff', 'N/A'):.3f}  "
              f"neg_ratio={ic.get('neg_ratio', 'N/A'):.3f}")

    # 上下文
    ctx = metrics.get("context", {})
    print(f"\n[上下文 - Context ({ctx.get('mode', '?')})]")
    wc = ctx.get("weather_coverage", {})
    for col, cov in wc.items():
        print(f"  {col:<25} 覆盖率={cov:.1%}")

    print("=" * 65)


def _save_report(metrics: dict, out_path: Path, extra_meta: dict = None):
    """保存 JSON 指标 + Markdown 报告。"""
    from src.structured_eval import report_markdown

    # JSON
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] 指标 JSON: {json_path}")

    # Markdown
    title_suffix = ""
    if extra_meta:
        title_suffix = f" | {extra_meta.get('experiment_id', '')} | {extra_meta.get('year', '')}"
    md = report_markdown(metrics, title=f"Structured Evaluation{title_suffix}")

    # 追加元信息
    if extra_meta:
        md += "\n\n---\n## 元信息\n"
        for k, v in extra_meta.items():
            md += f"- **{k}**: {v}\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[saved] Markdown 报告: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Traffic-Structured Diffusion 管线（road snap + context + structured eval）"
    )

    # 模型与数据
    parser.add_argument("--experiment_id", default="macro_soft_2024",
                        help="实验 ID（用于查 ckpt 和命名输出文件）")
    parser.add_argument("--ckpt_dir",      default=None,
                        help="Checkpoint 目录（可选，会自动推断）")
    parser.add_argument("--data_dir",      default=None,
                        help="data/nyc_crash 目录（采样时使用）")
    parser.add_argument("--num_samples",   type=int, default=10000)
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--year",          type=int, default=2024,
                        help="目标年份（决定使用哪个 OSM/天气数据）")

    # 输入（eval_only 模式）
    parser.add_argument("--synth_csv",  default=None,
                        help="已有合成样本 CSV（跳过采样和后处理时使用）")
    parser.add_argument("--real_csv",   default=None,
                        help="真实测试集 CSV（默认 data/nyc_crash/test.csv）")
    parser.add_argument("--eval_only",  action="store_true",
                        help="仅执行评估（不采样、不后处理）")

    # 后处理选项
    parser.add_argument("--context_mode", default=None,
                        choices=["historical_lookup", "future_simulation", "correction"],
                        help="上下文补全模式（None=跳过）")
    parser.add_argument("--weather_csv",  default=None,
                        help="天气 CSV 路径（auto 推断时不需要）")
    parser.add_argument("--snap_max_dist_m", type=float, default=300.0)
    parser.add_argument("--use_candidate_snap", action="store_true",
                        help="使用预构建候选点集 snap（比 nearest_edges 更快）")
    parser.add_argument("--road_snap_cache", default=None,
                        help="候选点集缓存 .npz 路径（可选）")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="即使已有后处理 CSV 也重新执行")

    # 评估选项
    parser.add_argument("--no_road_eval", action="store_true",
                        help="跳过道路距离评估（加速评估）")
    parser.add_argument("--out_report", default=None,
                        help="输出 Markdown 报告路径（默认自动生成）")

    args = parser.parse_args()

    # 推断 real_csv
    if args.real_csv is None:
        if args.year == 2024:
            args.real_csv = str(ROOT / "data" / "nyc_crash" / "test.csv")
        else:
            args.real_csv = str(ROOT / f"data/nyc_crash_{args.year}/test.csv")
    real_csv = Path(args.real_csv)
    if not real_csv.exists():
        print(f"[warn] 真实测试集不存在: {real_csv}")

    # ── Step 1 & 2: 采样 + 后处理（除非 eval_only）────────────────────────
    if args.eval_only:
        if not args.synth_csv:
            parser.error("--eval_only 需要 --synth_csv 参数")
        synth_csv = Path(args.synth_csv)
    else:
        raw_csv   = _sample_or_load(args)
        synth_csv = _postprocess(raw_csv, args)

    # ── Step 3: 评估 ──────────────────────────────────────────────────────
    metrics = _evaluate(synth_csv, real_csv, args)

    # ── Step 4: 输出 ──────────────────────────────────────────────────────
    _print_summary(metrics)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = args.experiment_id or "unknown"
    if args.out_report is None:
        out_md = ROOT / "results" / f"structured_eval_{exp_id}_{args.year}_{ts}.md"
    else:
        out_md = Path(args.out_report)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    _save_report(
        metrics,
        out_md,
        extra_meta={
            "experiment_id": exp_id,
            "year":          args.year,
            "synth_csv":     str(synth_csv),
            "real_csv":      str(real_csv),
            "context_mode":  args.context_mode,
            "timestamp":     ts,
        },
    )


if __name__ == "__main__":
    main()
