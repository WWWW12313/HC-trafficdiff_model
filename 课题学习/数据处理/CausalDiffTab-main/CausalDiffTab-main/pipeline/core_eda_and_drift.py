"""
core_eda_and_drift.py
=====================
针对 4 份关键数据表，做 **核心字段** 的单表 EDA + 三组分布对比 + 路网热点图。

四张表（全部映射到物理空间）：
  T17R_TRAIN = data/nyc_crash/train.csv          (2017 真实, QuantileTransformer 空间, 需逆变换)
  T17R_TEST  = data/nyc_crash/test.csv           (2017 真实 holdout, 同上)
  T25R       = results/postcovid_2025_fully_enriched_like_2017.csv   (2025 真实, 物理空间)
  T_SYN      = results/synthetic/_ours_full_model_full_samples_physical.csv (生成样本, 物理空间)

核心字段组（其他列视为派生副产物，**不参与漂移诊断**）：
  A. crash_time → CRASH_TIME_SIN/COS（恢复小时）
  B. accident_cause → is_distracted / is_speeding / is_failure_to_yield / is_following_too_closely
                       / is_drunk_driving / is_fatigue / is_view_obstructed / is_vehicle_defect
                       / is_backing_unsafely / is_pedestrian_related / is_inexperience / is_pavement_slippery
  C. geo → LATITUDE, LONGITUDE
  D. vehicle_type → is_sedan / is_suv / is_taxi / is_truck / is_pickup / is_bus / is_van
                     / is_motorcycle / is_bicycle / is_emergency
  E. casualty → NUMBER OF PERSONS INJURED, NUMBER_OF_PEDESTRIANS_INJURED_BIN / KILLED_BIN,
                NUMBER_OF_CYCLIST_INJURED_BIN / KILLED_BIN, NUMBER_OF_MOTORIST_INJURED_BIN / KILLED_BIN

输出：
  results/eda_core_summary.json                  四表关键字段单表统计
  results/three_way_distribution_comparison.json 3-way JS/PSI/KS/Wasserstein
  results/three_way_distribution_comparison.md   人类可读
  results/figures/hotspot_<tag>.png              四份表 hexbin 热点图
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

# ---------------------------- core feature groups ----------------------------
GROUP_A_TIME = ["CRASH_TIME_SIN", "CRASH_TIME_COS"]
GROUP_B_CAUSE = [
    "is_distracted", "is_speeding", "is_failure_to_yield", "is_following_too_closely",
    "is_drunk_driving", "is_fatigue", "is_view_obstructed", "is_vehicle_defect",
    "is_backing_unsafely", "is_pedestrian_related", "is_inexperience", "is_pavement_slippery",
]
GROUP_C_GEO = ["LATITUDE", "LONGITUDE"]
GROUP_D_VEH = [
    "is_sedan", "is_suv", "is_taxi", "is_truck", "is_pickup", "is_bus",
    "is_van", "is_motorcycle", "is_bicycle", "is_emergency",
]
GROUP_E_CAS = [
    "NUMBER OF PERSONS INJURED",
    "NUMBER_OF_PEDESTRIANS_INJURED_BIN", "NUMBER_OF_PEDESTRIANS_KILLED_BIN",
    "NUMBER_OF_CYCLIST_INJURED_BIN",   "NUMBER_OF_CYCLIST_KILLED_BIN",
    "NUMBER_OF_MOTORIST_INJURED_BIN",  "NUMBER_OF_MOTORIST_KILLED_BIN",
]

CONTINUOUS_CORE = GROUP_A_TIME + GROUP_C_GEO + ["NUMBER OF PERSONS INJURED"]
BINARY_CORE = GROUP_B_CAUSE + GROUP_D_VEH
ORDINAL_CORE = [c for c in GROUP_E_CAS if c != "NUMBER OF PERSONS INJURED"]

ALL_CORE = GROUP_A_TIME + GROUP_B_CAUSE + GROUP_C_GEO + GROUP_D_VEH + GROUP_E_CAS

# severity thresholds
JS_SEVERE, JS_MODERATE = 0.15, 0.05

# ---------------------------- IO -----------------------------------------------
def load_info() -> dict:
    return json.loads((ROOT / "results" / "info.json").read_text(encoding="utf-8"))


def load_table(path: Path, do_inverse: bool, info: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    if do_inverse:
        from evaluate_all import _inverse_transform_continuous
        df = _inverse_transform_continuous(df, info)
    return df


# ---------------------------- single-table EDA --------------------------------
def eda_one(df: pd.DataFrame, tag: str) -> dict:
    out: Dict[str, dict] = {"tag": tag, "n_rows": int(len(df)), "groups": {}}

    def _stats_continuous(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return {"n": 0}
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        return {
            "n": int(len(s)),
            "missing": int(df.shape[0] - len(s)),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(s.skew()), 4),
            "kurt": round(float(s.kurt()), 4),
            "p01": round(float(q[0.01]), 4),
            "p05": round(float(q[0.05]), 4),
            "p25": round(float(q[0.25]), 4),
            "p50": round(float(q[0.5]), 4),
            "p75": round(float(q[0.75]), 4),
            "p95": round(float(q[0.95]), 4),
            "p99": round(float(q[0.99]), 4),
        }

    def _stats_binary(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce")
        n = int(s.notna().sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "missing": int(df.shape[0] - n),
            "rate_pos": round(float((s > 0).mean()), 6),
            "n_pos": int((s > 0).sum()),
        }

    def _stats_ordinal(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return {"n": 0}
        vc = s.value_counts(normalize=True).sort_index().head(20).to_dict()
        return {
            "n": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "max": int(s.max()),
            "value_freq": {str(int(k) if float(k).is_integer() else k): round(float(v), 6) for k, v in vc.items()},
        }

    # group A time: also recover hour from sin/cos
    grpA = {}
    for c in GROUP_A_TIME:
        if c in df.columns:
            grpA[c] = _stats_continuous(df[c])
    if {"CRASH_TIME_SIN", "CRASH_TIME_COS"}.issubset(df.columns):
        sin = pd.to_numeric(df["CRASH_TIME_SIN"], errors="coerce")
        cos = pd.to_numeric(df["CRASH_TIME_COS"], errors="coerce")
        ang = np.arctan2(sin, cos)
        # angle in [-pi, pi] → fraction of day → hour
        frac = (ang / (2 * math.pi)) % 1.0
        hour = (frac * 24.0)
        grpA["__hour_recovered"] = _stats_continuous(pd.Series(hour))
        # hour histogram (24 bins)
        h_int = hour.dropna().astype(int).clip(0, 23)
        hist = h_int.value_counts(normalize=True).sort_index().to_dict()
        grpA["__hour_hist24"] = {int(k): round(float(v), 6) for k, v in hist.items()}
    out["groups"]["A_time"] = grpA

    out["groups"]["B_cause"] = {c: _stats_binary(df[c]) for c in GROUP_B_CAUSE if c in df.columns}

    grpC = {}
    for c in GROUP_C_GEO:
        if c in df.columns:
            grpC[c] = _stats_continuous(df[c])
    out["groups"]["C_geo"] = grpC

    out["groups"]["D_vehicle"] = {c: _stats_binary(df[c]) for c in GROUP_D_VEH if c in df.columns}

    grpE = {}
    for c in GROUP_E_CAS:
        if c not in df.columns:
            continue
        if c == "NUMBER OF PERSONS INJURED":
            grpE[c] = _stats_continuous(df[c])
        else:
            grpE[c] = _stats_ordinal(df[c])
    out["groups"]["E_casualty"] = grpE

    return out


# ---------------------------- distribution distances --------------------------
def _js_continuous(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    pa = pa + 1e-9; pb = pb + 1e-9
    pa = pa / pa.sum(); pb = pb / pb.sum()
    return float(jensenshannon(pa, pb))


def _psi_continuous(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """PSI = sum((p_b - p_a) * ln(p_b / p_a)) over `a`-quantile bins."""
    eps = 1e-6
    qs = np.quantile(a, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    pa, _ = np.histogram(a, bins=qs)
    pb, _ = np.histogram(b, bins=qs)
    pa = pa / max(pa.sum(), 1) + eps
    pb = pb / max(pb.sum(), 1) + eps
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _js_categorical(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a.dropna().unique()) | set(b.dropna().unique()))
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    p = np.array([pa.get(c, 0) + 1e-9 for c in cats])
    q = np.array([pb.get(c, 0) + 1e-9 for c in cats])
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def _psi_categorical(a: pd.Series, b: pd.Series) -> float:
    eps = 1e-6
    cats = sorted(set(a.dropna().unique()) | set(b.dropna().unique()))
    pa = a.value_counts(normalize=True); pb = b.value_counts(normalize=True)
    pa_arr = np.array([pa.get(c, 0) + eps for c in cats])
    pb_arr = np.array([pb.get(c, 0) + eps for c in cats])
    return float(np.sum((pb_arr - pa_arr) * np.log(pb_arr / pa_arr)))


def compare_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> List[dict]:
    rows: List[dict] = []
    for col in CONTINUOUS_CORE:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        a = pd.to_numeric(df_a[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(df_b[col], errors="coerce").dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        js = _js_continuous(a, b)
        psi = _psi_continuous(a, b)
        ks = ks_2samp(a, b)
        std = float(np.std(np.concatenate([a, b]))) or 1.0
        wd = float(wasserstein_distance(a / std, b / std))
        rows.append({
            "feature": col, "type": "continuous",
            "JS": round(js, 6), "PSI": round(psi, 6),
            "KS_stat": round(float(ks[0]), 6), "KS_p": round(float(ks[1]), 6),
            "Wasserstein_normed": round(wd, 6),
            f"mean_{name_a}": round(float(np.mean(a)), 4),
            f"mean_{name_b}": round(float(np.mean(b)), 4),
            "severity": "SEVERE" if js >= JS_SEVERE else ("MODERATE" if js >= JS_MODERATE else "STABLE"),
        })

    for col in BINARY_CORE + ORDINAL_CORE:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        a = pd.to_numeric(df_a[col], errors="coerce")
        b = pd.to_numeric(df_b[col], errors="coerce")
        js = _js_categorical(a.fillna(-1).astype(int), b.fillna(-1).astype(int))
        psi = _psi_categorical(a.fillna(-1).astype(int), b.fillna(-1).astype(int))
        rows.append({
            "feature": col, "type": "binary" if col in BINARY_CORE else "ordinal",
            "JS": round(js, 6), "PSI": round(psi, 6),
            "KS_stat": None, "KS_p": None, "Wasserstein_normed": None,
            f"rate_{name_a}": round(float((a > 0).mean()), 6),
            f"rate_{name_b}": round(float((b > 0).mean()), 6),
            "severity": "SEVERE" if js >= JS_SEVERE else ("MODERATE" if js >= JS_MODERATE else "STABLE"),
        })

    rows.sort(key=lambda r: -r["JS"])
    return rows


# ---------------------------- hotspot map -------------------------------------
def hotspot_map(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat = pd.to_numeric(df.get("LATITUDE"), errors="coerce").dropna()
    lon = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
    lon = lon.loc[lat.index].dropna()
    lat = lat.loc[lon.index]
    if len(lat) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=120)
    hb = ax.hexbin(lon, lat, gridsize=120, mincnt=1, bins="log", cmap="magma",
                   extent=(-74.30, -73.65, 40.45, 40.95))
    ax.set_title(f"{title}  (n={len(lat):,})")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(-74.30, -73.65); ax.set_ylim(40.45, 40.95)
    ax.set_aspect("equal")
    cb = fig.colorbar(hb, ax=ax, shrink=0.85)
    cb.set_label("log10(crash count)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


# ---------------------------- markdown reporter -------------------------------
def render_pair_md(name_a: str, name_b: str, rows: List[dict]) -> str:
    lines = [f"### {name_a}  vs  {name_b}", ""]
    sev = sum(1 for r in rows if r["severity"] == "SEVERE")
    mod = sum(1 for r in rows if r["severity"] == "MODERATE")
    sta = sum(1 for r in rows if r["severity"] == "STABLE")
    lines += [f"- counts: SEVERE={sev}  MODERATE={mod}  STABLE={sta}", ""]
    lines += ["| # | feature | type | JS | PSI | KS | Wass | sev | note |",
              "|---|---------|------|----|-----|----|------|-----|------|"]
    for i, r in enumerate(rows, 1):
        ks = "" if r["KS_stat"] is None else f"{r['KS_stat']:.3f}"
        wd = "" if r["Wasserstein_normed"] is None else f"{r['Wasserstein_normed']:.3f}"
        if r["type"] == "continuous":
            note = f"mean: {r.get(f'mean_{name_a}')} → {r.get(f'mean_{name_b}')}"
        else:
            note = f"rate: {r.get(f'rate_{name_a}'):.3f} → {r.get(f'rate_{name_b}'):.3f}"
        lines.append(
            f"| {i} | `{r['feature']}` | {r['type']} | **{r['JS']:.4f}** | {r['PSI']:.3f} | {ks} | {wd} | {r['severity']} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------- main --------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_syn", type=str, default=None,
                    help="覆盖 T_SYN 文件路径（默认 _ours_full_model_full_samples_physical.csv）")
    ap.add_argument("--tag", type=str, default="",
                    help="输出后缀，例如 'zsuncond' → results/three_way_distribution_comparison_zsuncond.{json,md}")
    args, _unknown = ap.parse_known_args()
    out_suffix = f"_{args.tag}" if args.tag else ""

    info = load_info()
    t_syn_path = (Path(args.t_syn) if args.t_syn else
                  ROOT / "results" / "synthetic" / "_ours_full_model_full_samples_physical.csv")
    if not t_syn_path.is_absolute():
        t_syn_path = ROOT / t_syn_path
    paths = {
        "T17R_TRAIN": (ROOT / "data" / "nyc_crash" / "train.csv", True),
        "T17R_TEST":  (ROOT / "data" / "nyc_crash" / "test.csv",  True),
        "T25R":       (ROOT / "results" / "postcovid_2025_fully_enriched_like_2017.csv", False),
        "T_SYN":      (t_syn_path, False),
    }
    print(f"[load] loading 4 tables ... (T_SYN={t_syn_path.name}, tag='{args.tag}')")
    tables = {tag: load_table(p, inv, info) for tag, (p, inv) in paths.items()}
    for tag, df in tables.items():
        print(f"  {tag:11s} rows={len(df):,}  cols={df.shape[1]}")

    # --- 1) single-table EDA on core fields
    print("[eda] single-table EDA on core groups A-E ...")
    eda = {tag: eda_one(df, tag) for tag, df in tables.items()}
    (ROOT / "results" / f"eda_core_summary{out_suffix}.json").write_text(
        json.dumps(eda, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("  -> results/eda_core_summary.json")

    # --- 2) 3-way distribution comparison
    print("[3way] distribution comparison ...")
    pairs = [
        ("T17R_TRAIN", "T25R"),     # 真年份漂移
        ("T17R_TRAIN", "T_SYN"),    # 训练拟合度
        ("T25R",       "T_SYN"),    # 跨年生成保真度
    ]
    pair_results = {}
    for a, b in pairs:
        pair_results[f"{a}__vs__{b}"] = compare_pair(tables[a], tables[b], a, b)
    (ROOT / "results" / f"three_way_distribution_comparison{out_suffix}.json").write_text(
        json.dumps(pair_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# Three-Way Distribution Comparison (core fields only)", ""]
    md += [f"- T17R_TRAIN: {len(tables['T17R_TRAIN']):,}  rows (2017 真实, 已逆变换)",
           f"- T17R_TEST : {len(tables['T17R_TEST']):,}  rows",
           f"- T25R      : {len(tables['T25R']):,}  rows (2025 真实, 同 2017 富化口径)",
           f"- T_SYN     : {len(tables['T_SYN']):,}  rows (生成样本, physical)",
           f"- core groups: A_time / B_cause / C_geo / D_vehicle / E_casualty",
           f"- severity:  JS>={JS_SEVERE} SEVERE | JS>={JS_MODERATE} MODERATE | else STABLE",
           "",]
    for (a, b), rows in zip(pairs, pair_results.values()):
        md.append(render_pair_md(a, b, rows))
    (ROOT / "results" / f"three_way_distribution_comparison{out_suffix}.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(f"  -> results/three_way_distribution_comparison{out_suffix}.[json|md]")

    # --- 2.5) 加权迁移退化指数（按用户明确口径：5 大核心组等权，其他字段不计入）
    print("[degradation] weighted transfer-degradation index ...")
    GROUP_OF = {}
    for c in GROUP_A_TIME:    GROUP_OF[c] = "A_time"
    for c in GROUP_B_CAUSE:   GROUP_OF[c] = "B_cause"
    for c in GROUP_C_GEO:     GROUP_OF[c] = "C_geo"
    for c in GROUP_D_VEH:     GROUP_OF[c] = "D_vehicle"
    for c in GROUP_E_CAS:     GROUP_OF[c] = "E_casualty"
    GROUP_WEIGHTS = {  # 用户口径: 5 组等权, 其他字段权重=0 (不在 core_eda 范围内)
        "A_time": 0.20, "B_cause": 0.20, "C_geo": 0.20,
        "D_vehicle": 0.20, "E_casualty": 0.20,
    }

    deg_summary: Dict[str, dict] = {"weights": GROUP_WEIGHTS, "pairs": {}}
    md_deg = ["", "## 加权迁移退化指数（核心字段口径）", "",
              "口径：crash_time / accident_cause / lat-lon / vehicle_type / casualty 五组等权（each 0.20），",
              "其他字段（OSM/天气/SEASON 等）视为反向映射副产物，**不计入退化率**。", "",
              "退化指标定义：",
              "- `mean_JS_group` = 该组核心字段 JS 散度均值",
              "- `weighted_JS`   = Σ w_g · mean_JS_group   (越小越好)",
              "- `severe_count`  = 组内 JS≥0.15 的字段数", "",
              "| 配对 | A_time | B_cause | C_geo | D_vehicle | E_casualty | **加权 JS** | SEVERE 字段数 |",
              "|---|---|---|---|---|---|---|---|"]
    for (a, b), rows in zip(pairs, pair_results.values()):
        per_group: Dict[str, List[float]] = {g: [] for g in GROUP_WEIGHTS}
        severe_n = 0
        for r in rows:
            g = GROUP_OF.get(r["feature"])
            if g is None:
                continue
            per_group[g].append(r["JS"])
            if r["severity"] == "SEVERE":
                severe_n += 1
        means = {g: (round(float(np.mean(v)), 4) if v else None) for g, v in per_group.items()}
        weighted = round(float(sum(GROUP_WEIGHTS[g] * (means[g] or 0.0) for g in GROUP_WEIGHTS)), 4)
        deg_summary["pairs"][f"{a}__vs__{b}"] = {
            "mean_JS_per_group": means,
            "weighted_JS": weighted,
            "severe_count": severe_n,
        }
        cells = [f"{means[g]:.4f}" if means[g] is not None else "—" for g in
                 ["A_time", "B_cause", "C_geo", "D_vehicle", "E_casualty"]]
        md_deg.append(f"| `{a}` ↔ `{b}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | **{weighted:.4f}** | {severe_n} |")

    # 退化率 = T25R↔T_SYN 的加权JS / T17R_TRAIN↔T_SYN 的加权JS - 1
    base = deg_summary["pairs"].get("T17R_TRAIN__vs__T_SYN", {}).get("weighted_JS")
    cross = deg_summary["pairs"].get("T25R__vs__T_SYN", {}).get("weighted_JS")
    if base and cross:
        deg_rate = round((cross - base) / max(base, 1e-9), 4)
        deg_summary["transfer_degradation_rate"] = deg_rate
        md_deg += ["",
                   f"**跨年迁移退化率** = (`T25R↔T_SYN` − `T17R_TRAIN↔T_SYN`) / `T17R_TRAIN↔T_SYN` "
                   f"= ({cross:.4f} − {base:.4f}) / {base:.4f} = **{deg_rate*100:+.2f}%**",
                   "",
                   "> 数值含义：模型在 2025 上加权 JS 相对于 2017 拟合 JS 的相对增长率；",
                   "> < 0% 表示迁移好于拟合（不可能在无适应情况下出现），> 100% 表示加权漂移翻倍以上。"]
    (ROOT / "results" / f"transfer_degradation_index{out_suffix}.json").write_text(
        json.dumps(deg_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 追加到三组对比 md 末尾
    with (ROOT / "results" / f"three_way_distribution_comparison{out_suffix}.md").open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(md_deg) + "\n")
    print(f"  -> results/transfer_degradation_index{out_suffix}.json (+追加到 three_way_*{out_suffix}.md)")

    # --- 3) hotspot maps
    print("[hotspot] hexbin maps ...")
    fig_dir = ROOT / "results" / "figures"
    for tag, df in tables.items():
        hotspot_map(df, fig_dir / f"hotspot_{tag}.png", title=f"NYC crash hotspot — {tag}")
        print(f"  -> {fig_dir.name}/hotspot_{tag}.png")

    print("DONE.")


if __name__ == "__main__":
    main()
