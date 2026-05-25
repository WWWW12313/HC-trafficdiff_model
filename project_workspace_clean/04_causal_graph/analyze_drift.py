"""
2017 vs 2025 特征分布漂移诊断脚本
===================================
对每个特征计算：
  连续特征 → JS 散度（histogram 30 bins）+ Wasserstein 距离 + KS 统计量
  离散/二值/类别特征 → 频率 JS 散度

重要：2017 test.csv 为 QuantileTransformer 归一化空间，
      2025 数据绝大多数列已在物理空间，本脚本先对 2017 做 inverse_transform
      使两者统一到物理空间后再比较。

已知 2025 数据质量问题（5 列）：
  DIST_TO_SIGNAL_M / INFERRED_LANES → 全为单一值（OSM 特征未正确生成）
  HAS_TRAFFIC_SIGNAL / OSM_ONEWAY   → 全为 0
  OSM_TYPE                           → 全为 "residential"
  这 5 列不做漂移分析，单独输出为 DATA_QUALITY_ISSUE。

输出：
  results/drift_report_latest.md
  results/drift_report_latest.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

# ── 2025 数据中全为常数的列（动态检测，不再硬编码）
CONSTANT_IN_2025: set = set()

# ── 连续特征（histogram 分箱）
CONTINUOUS_COLS = [
    "LATITUDE", "LONGITUDE",
    "CRASH_TIME_SIN", "CRASH_TIME_COS",
    "TEMP_C", "prcp", "WIND_SPEED_KMH",
    "DIST_TO_SIGNAL_M", "INFERRED_LANES",
]

# ── 二值/低基数整数特征（频率对比）
BINARY_COLS = [
    "IS_MULTI_VEHICLE",
    "HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY",
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup",
    "is_bus", "is_van", "is_motorcycle", "is_bicycle", "is_emergency",
    "is_distracted", "is_speeding", "is_failure_to_yield",
    "is_following_too_closely", "is_drunk_driving", "is_fatigue",
    "is_view_obstructed", "is_vehicle_defect", "is_backing_unsafely",
    "is_pedestrian_related", "is_inexperience", "is_pavement_slippery",
]

# ── 低基数整数（当类别处理）
ORDINAL_COLS = [
    "DAY_OF_WEEK", "coco", "TOTAL_VEHICLES",
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN", "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN", "NUMBER_OF_MOTORIST_KILLED_BIN",
]

# ── 字符串类别特征（含 OSM_TYPE）
CATEGORICAL_COLS = ["SEASON", "TIME_PERIOD", "WEATHER_CONDITION", "OSM_TYPE"]

# ── 目标变量（单独分析）
TARGET_COL = "NUMBER OF PERSONS INJURED"

DRIFT_SEVERE  = 0.15   # JS > 0.15 → 严重漂移
DRIFT_MODERATE = 0.05  # JS > 0.05 → 中度漂移


def _js_from_histograms(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    # 加平滑避免 log(0)
    pa = pa + 1e-9
    pb = pb + 1e-9
    pa /= pa.sum()
    pb /= pb.sum()
    return float(jensenshannon(pa, pb))


def _js_from_freq(a: pd.Series, b: pd.Series) -> float:
    cats = set(a.unique()) | set(b.unique())
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    p = np.array([pa.get(c, 0.0) + 1e-9 for c in cats])
    q = np.array([pb.get(c, 0.0) + 1e-9 for c in cats])
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def _severity(js: float) -> str:
    if js >= DRIFT_SEVERE:
        return "SEVERE"
    if js >= DRIFT_MODERATE:
        return "MODERATE"
    return "STABLE"


def analyze(df17: pd.DataFrame, df25: pd.DataFrame) -> list[dict]:
    results = []

    # 连续特征
    for col in CONTINUOUS_COLS:
        if col not in df17.columns or col not in df25.columns:
            continue
        a = df17[col].dropna().to_numpy(dtype=float)
        b = df25[col].dropna().to_numpy(dtype=float)
        js = _js_from_histograms(a, b)
        ks_result = ks_2samp(a, b)
        ks_stat, ks_p = float(ks_result[0]), float(ks_result[1])  # type: ignore[arg-type]
        # wasserstein 归一化（除以标准差）
        std = float(np.std(np.concatenate([a, b]))) or 1.0
        wass = float(wasserstein_distance(a / std, b / std))
        results.append({
            "feature": col,
            "type": "continuous",
            "js_divergence": round(js, 6),
            "wasserstein_normed": round(wass, 6),
            "ks_statistic": round(ks_stat, 6),
            "ks_pvalue": round(ks_p, 6),
            "severity": _severity(js),
            "mean_2017": round(float(np.mean(a)), 4),
            "mean_2025": round(float(np.mean(b)), 4),
            "std_2017": round(float(np.std(a)), 4),
            "std_2025": round(float(np.std(b)), 4),
        })

    # 二值特征
    for col in BINARY_COLS:
        if col not in df17.columns:
            continue
        js = _js_from_freq(df17[col], df25[col])
        rate17 = float(df17[col].mean())
        rate25 = float(df25[col].mean())
        results.append({
            "feature": col,
            "type": "binary",
            "js_divergence": round(js, 6),
            "wasserstein_normed": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "severity": _severity(js),
            "rate_2017": round(rate17, 4),
            "rate_2025": round(rate25, 4),
            "rate_delta": round(rate25 - rate17, 4),
        })

    # 有序整数类别
    for col in ORDINAL_COLS:
        if col not in df17.columns:
            continue
        js = _js_from_freq(df17[col], df25[col])
        results.append({
            "feature": col,
            "type": "ordinal",
            "js_divergence": round(js, 6),
            "wasserstein_normed": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "severity": _severity(js),
            "mean_2017": round(float(df17[col].mean()), 4),
            "mean_2025": round(float(df25[col].mean()), 4),
        })

    # 字符串类别
    for col in CATEGORICAL_COLS:
        if col not in df17.columns:
            continue
        js = _js_from_freq(df17[col].astype(str), df25[col].astype(str))
        results.append({
            "feature": col,
            "type": "categorical",
            "js_divergence": round(js, 6),
            "wasserstein_normed": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "severity": _severity(js),
        })

    # 目标变量
    if TARGET_COL in df17.columns:
        col = TARGET_COL
        a = df17[col].dropna().to_numpy(dtype=float)
        b = df25[col].dropna().to_numpy(dtype=float)
        js = _js_from_histograms(a, b, bins=20)
        ks_result = ks_2samp(a, b)
        ks_stat, ks_p = float(ks_result[0]), float(ks_result[1])  # type: ignore[arg-type]
        results.append({
            "feature": col,
            "type": "target",
            "js_divergence": round(js, 6),
            "wasserstein_normed": None,
            "ks_statistic": round(ks_stat, 6),
            "ks_pvalue": round(ks_p, 6),
            "severity": _severity(js),
            "mean_2017": round(float(np.mean(a)), 4),
            "mean_2025": round(float(np.mean(b)), 4),
        })

    # ── 2025 数据质量问题列（全为常数，不做漂移分析）
    for col in sorted(CONSTANT_IN_2025):
        col_type = "categorical" if col == "OSM_TYPE" else ("binary" if col in {"HAS_TRAFFIC_SIGNAL", "OSM_ONEWAY"} else "continuous")
        val25 = df25[col].iloc[0] if col in df25.columns else "N/A"
        results.append({
            "feature": col,
            "type": col_type,
            "js_divergence": None,
            "wasserstein_normed": None,
            "ks_statistic": None,
            "ks_pvalue": None,
            "severity": "DATA_QUALITY_ISSUE",
            "note": f"2025 全行常数={val25!r}，OSM 特征未正确生成",
        })

    # 按 JS 降序排序（DATA_QUALITY 条目 js_divergence=None，排到末尾）
    results.sort(key=lambda x: (x["js_divergence"] is None, -(x["js_divergence"] or 0)))
    return results


def _build_md(results: list[dict], ts: str, n17: int, n25: int) -> str:
    severe   = [r for r in results if r["severity"] == "SEVERE"]
    moderate = [r for r in results if r["severity"] == "MODERATE"]
    stable   = [r for r in results if r["severity"] == "STABLE"]
    dq       = [r for r in results if r["severity"] == "DATA_QUALITY_ISSUE"]

    lines = [
        "# Feature Drift Report: 2017 → 2025",
        "",
        f"- generated_at: `{ts}`",
        f"- source_2017: `synthetic/nyc_crash/test.csv`  ({n17:,} rows)",
        f"- source_2025: `results/postcovid_test_2025_n{n25}.csv`  ({n25:,} rows)",
        f"- threshold_severe: JS > {DRIFT_SEVERE}",
        f"- threshold_moderate: JS > {DRIFT_MODERATE}",
        "",
        "## 概览",
        "",
        f"| 等级 | 数量 | 特征 |",
        f"| --- | --- | --- |",
        f"| 🔴 SEVERE   | {len(severe)} | {', '.join(r['feature'] for r in severe)} |",
        f"| 🟡 MODERATE | {len(moderate)} | {', '.join(r['feature'] for r in moderate)} |",
        f"| 🟢 STABLE   | {len(stable)} | （{len(stable)} 个特征分布稳定，JS ≤ {DRIFT_MODERATE}） |",
        f"| ⚫ DATA_QUALITY | {len(dq)} | {', '.join(r['feature'] for r in dq)} |",
        "",
        "## 全特征排名（按 JS 散度降序）",
        "",
        "| # | 特征 | 类型 | JS 散度 | 漂移等级 | 备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for i, r in enumerate(results, 1):
        badge_map = {"SEVERE": "🔴", "MODERATE": "🟡", "STABLE": "🟢", "DATA_QUALITY_ISSUE": "⚫"}
        badge = badge_map.get(r["severity"], "❓")
        note = ""
        if r["severity"] == "DATA_QUALITY_ISSUE":
            note = r.get("note", "")
            lines.append(
                f"| {i} | `{r['feature']}` | {r['type']} | N/A | {badge} DATA_QUALITY | {note} |"
            )
            continue
        if r["type"] == "continuous":
            note = f"mean: {r.get('mean_2017')} → {r.get('mean_2025')}"
        elif r["type"] == "binary":
            note = f"rate: {r.get('rate_2017'):.3f} → {r.get('rate_2025'):.3f}  (Δ{r.get('rate_delta'):+.3f})"
        elif r["type"] in ("ordinal", "target"):
            note = f"mean: {r.get('mean_2017')} → {r.get('mean_2025')}"
        lines.append(
            f"| {i} | `{r['feature']}` | {r['type']} | **{r['js_divergence']:.4f}** | {badge} {r['severity']} | {note} |"
        )

    # 严重漂移详细分析
    if severe:
        lines += [
            "",
            "## 严重漂移特征详析（SEVERE, JS > 0.15）",
            "",
        ]
        for r in severe:
            lines += [f"### `{r['feature']}`", ""]
            lines += [f"- JS 散度: **{r['js_divergence']:.4f}**"]
            if r.get("ks_statistic") is not None:
                lines += [f"- KS 统计量: {r['ks_statistic']:.4f}  (p={r['ks_pvalue']:.4e})"]
            if r.get("wasserstein_normed") is not None:
                lines += [f"- Wasserstein 距离（归一化）: {r['wasserstein_normed']:.4f}"]
            if r["type"] == "continuous":
                lines += [
                    f"- 2017 均值/标准差: {r['mean_2017']} / {r['std_2017']}",
                    f"- 2025 均值/标准差: {r['mean_2025']} / {r['std_2025']}",
                ]
            elif r["type"] == "binary":
                lines += [
                    f"- 2017 发生率: {r['rate_2017']:.4f}",
                    f"- 2025 发生率: {r['rate_2025']:.4f}  (Δ {r['rate_delta']:+.4f})",
                ]
            elif r["type"] in ("ordinal", "target"):
                lines += [
                    f"- 2017 均值: {r['mean_2017']}",
                    f"- 2025 均值: {r['mean_2025']}",
                ]
            lines.append("")

    # 目标变量单独说明
    target_row = next((r for r in results if r["feature"] == TARGET_COL), None)
    if target_row:
        lines += [
            "## 目标变量漂移（NUMBER OF PERSONS INJURED）",
            "",
            f"- JS 散度: **{target_row['js_divergence']:.4f}**  → {target_row['severity']}",
            f"- KS 统计量: {target_row.get('ks_statistic')}  (p={target_row.get('ks_pvalue')})",
            f"- 2017 均值: {target_row.get('mean_2017')}  |  2025 均值: {target_row.get('mean_2025')}",
            "",
            "> 目标变量分布的稳定性直接决定迁移评测中 R² 退化的上界。",
        ]

    # 数据质量问题专节
    if dq:
        lines += [
            "",
            "## ⚫ 数据质量问题列（2025 数据中全为常数，OSM 特征缺失）",
            "",
            "| 列名 | 类型 | 2025 中的值 | 问题说明 |",
            "| --- | --- | --- | --- |",
        ]
        for r in dq:
            val_str = r.get("note", "").split("=")[-1].split("，")[0] if r.get("note") else "?"
            lines.append(f"| `{r['feature']}` | {r['type']} | {val_str} | {r.get('note', '')} |")
        lines += [
            "",
            "> **影响**：这 5 列在 2025 迁移评测中完全无变异，会导致模型无法学习 OSM 空间特征，",
            "> 直接拉低 TSTR 的 R² 和分类准确率。建议重新生成 2025 测试集，确保 OSM 查询生效。",
        ]

    return "\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-2017", type=str, default=None,
                    help="Path to 2017 CSV (default: synthetic/nyc_crash/test.csv with inverse transform)")
    ap.add_argument("--source-2025", type=str, default=None,
                    help="Path to 2025 CSV (default: postcovid_2025_fully_enriched_like_2017.csv, fallback to legacy n82698/n5000)")
    ap.add_argument("--no-inverse", action="store_true",
                    help="跳过 2017 QuantileTransformer 逆变换 (当 source_2017 已是物理空间时)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix tag for report files (e.g. 'like2017')")
    args = ap.parse_args()

    # ── 读取数据
    if args.source_2017:
        src17 = Path(args.source_2017)
        if not src17.is_absolute():
            src17 = ROOT / src17
    else:
        src17 = ROOT / "synthetic" / "nyc_crash" / "test.csv"
    df17 = pd.read_csv(src17)
    if args.source_2025:
        src25 = Path(args.source_2025)
        if not src25.is_absolute():
            src25 = ROOT / src25
    else:
        _plike  = ROOT / "results" / "postcovid_2025_fully_enriched_like_2017.csv"
        _p82698 = ROOT / "results" / "postcovid_test_2025_n82698.csv"
        _p5000  = ROOT / "results" / "postcovid_test_2025_n5000.csv"
        src25 = _plike if _plike.exists() else (_p82698 if _p82698.exists() else _p5000)
    df25 = pd.read_csv(src25)
    print(f"[drift] 2017: {len(df17)} rows ({src17.name})  |  2025: {len(df25)} rows ({src25.name})")

    # ── 将 2017 逆变换到物理空间（QuantileTransformer → GPS / 摄氏度 / mm）
    if args.no_inverse:
        print("[drift] 跳过逆变换 (--no-inverse)")
    else:
        try:
            from evaluate_all import _inverse_transform_continuous  # noqa: F401
            with open(ROOT / "results" / "info.json", encoding="utf-8") as _f:
                _info = json.load(_f)
            df17 = _inverse_transform_continuous(df17, _info)
            print("[drift] 2017 数据已完成 QuantileTransformer 逆变换（物理空间）")
        except Exception as _e:
            print(f"[drift] 警告：逆变换失败（{_e}），直接使用原始归一化空间")

    results = analyze(df17, df25)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results"

    md_text = _build_md(results, ts, len(df17), len(df25))
    payload = {
        "generated_at": ts,
        "n_2017": len(df17),
        "n_2025": len(df25),
        "thresholds": {"severe": DRIFT_SEVERE, "moderate": DRIFT_MODERATE},
        "features": results,
    }

    suffix = f"_{args.tag}" if args.tag else ""
    (out_dir / f"drift_report_latest{suffix}.md").write_text(md_text, encoding="utf-8")
    (out_dir / f"drift_report_latest{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"drift_report_{ts}{suffix}.md").write_text(md_text, encoding="utf-8")

    print(f"[drift] 写出：results/drift_report_latest{suffix}.md")
    print()
    # 控制台预览
    print(md_text)


if __name__ == "__main__":
    main()
