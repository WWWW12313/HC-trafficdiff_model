"""
遍历 results/synthetic/*.csv，与真实测试集对比，输出 TSTR / 统计距离 / 逻辑违背率。

示例:
  python pipeline/evaluate_all.py --real_test synthetic/nyc_crash/test.csv
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportAttributeAccessIssue=false, reportArgumentType=false

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from benchmark_evaluator import BenchmarkEvaluator
from evaluate_joint_metrics import (
    CMIPairSpec,
    ConditionalMutualInformationEvaluator,
    evaluate_shd_from_files,
)

CDT_ROOT = Path(__file__).resolve().parent.parent


def _to_numeric_series(values: pd.Series) -> pd.Series:
    """Convert to numeric and always return a Series for static type safety."""
    converted = pd.to_numeric(values, errors="coerce")
    if isinstance(converted, pd.Series):
        return converted
    return pd.Series(converted, index=values.index)


def _safe_finite_float_array(values: np.ndarray) -> np.ndarray:
    """Convert to float64 and replace NaN/Inf values for stable distance computation."""
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, 0.0)


def _sanitize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost 等要求特征名为字符串且不含 [, ]、< 等字符（get_dummies 可能产生）。"""
    out = df.copy()
    out.columns = out.columns.astype(str).str.replace(r"[<\[\]]", "", regex=True)
    return out


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map: Dict[str, str] = {_normalize_col_name(c): c for c in df.columns}
    cand_norm = [_normalize_col_name(x) for x in candidates]
    for n in cand_norm:
        if n in norm_map:
            return norm_map[n]
    for n in cand_norm:
        for k, v in norm_map.items():
            if n in k or k in n:
                return v
    return None


def _infer_coordinate_domain(df: pd.DataFrame) -> str:
    """Infer whether coordinates are physical NYC lat/lon or transformed space."""
    lat_col = _find_first_col(df, ["LATITUDE", "lat"])
    lon_col = _find_first_col(df, ["LONGITUDE", "lon", "lng"])
    if lat_col is None or lon_col is None:
        return "missing"
    lat = _to_numeric_series(df[lat_col]).dropna()
    lon = _to_numeric_series(df[lon_col]).dropna()
    if len(lat) < 10 or len(lon) < 10:
        return "missing"
    in_nyc = ((lat >= 40.40) & (lat <= 40.95) & (lon >= -74.30) & (lon <= -73.70)).mean()
    return "physical" if float(in_nyc) > 0.95 else "transformed"


def _load_continuous_transformer() -> Tuple[Optional[Any], List[str]]:
    p = CDT_ROOT / "data" / "processed" / "continuous_scaler.pkl"
    if not p.is_file():
        return None, []
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "scaler" in obj:
            scaler = obj.get("scaler")
            cols = list(obj.get("columns", []))
            return scaler, cols
        return obj, []
    except Exception:
        return None, []


def _inverse_transform_continuous(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Inverse-transform continuous features back to physical space if they appear transformed."""
    out = df.copy()
    space = _infer_coordinate_domain(out)
    if space == "physical":
        return out

    scaler, scaler_cols = _load_continuous_transformer()
    if scaler is None:
        return out

    num_cols = list(info.get("num_col_names", [])) if info else []
    if not num_cols:
        return out

    cols = [c for c in scaler_cols if c in out.columns and c in num_cols]
    if not cols:
        cols = [c for c in num_cols if c in out.columns]
    if not cols:
        return out

    vals = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    # Align with postprocess logic used by this project.
    vals = np.clip(vals, -5.2, 5.2)
    try:
        restored = scaler.inverse_transform(vals)
    except Exception:
        return out
    out[cols] = restored
    return out



def _jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    from scipy.spatial.distance import jensenshannon

    p = np.asarray(p, dtype=float).ravel() + eps
    q = np.asarray(q, dtype=float).ravel() + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0))


def _statistical_distances(
    syn: pd.DataFrame, real: pd.DataFrame, info: Optional[dict]
) -> Dict[str, float]:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError as e:
        raise SystemExit("统计距离需要 scipy: python -m pip install scipy") from e

    num_names = list(info.get("num_col_names", [])) if info else []
    cat_names = list(info.get("cat_col_names", [])) if info else []
    target = info.get("target_col") if info else None

    if not num_names and not cat_names:
        num_names = [
            c
            for c in syn.columns
            if c in real.columns and pd.api.types.is_numeric_dtype(syn[c])
        ]
        cat_names = [
            c
            for c in syn.columns
            if c in real.columns and not pd.api.types.is_numeric_dtype(syn[c])
        ]

    if target and target in num_names:
        num_names = [c for c in num_names if c != target]

    w_vals: List[float] = []
    for c in num_names:
        if c not in syn.columns or c not in real.columns:
            continue
        a = _to_numeric_series(syn[c]).dropna()
        b = _to_numeric_series(real[c]).dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        av = _safe_finite_float_array(a.values)
        bv = _safe_finite_float_array(b.values)
        w_vals.append(wasserstein_distance(av, bv))
    mean_w = float(np.mean(w_vals)) if w_vals else float("nan")

    js_vals: List[float] = []
    for c in cat_names:
        if c not in syn.columns or c not in real.columns:
            continue
        vs = syn[c].astype(str).fillna("_nan_")
        vr = real[c].astype(str).fillna("_nan_")
        cats = sorted(set(vs.unique()) | set(vr.unique()))
        if len(cats) < 2:
            continue
        idx = {k: i for i, k in enumerate(cats)}
        ps = np.zeros(len(cats), dtype=float)
        pr = np.zeros(len(cats), dtype=float)
        for x in vs:
            ps[idx[x]] += 1
        for x in vr:
            pr[idx[x]] += 1
        ps /= max(ps.sum(), 1)
        pr /= max(pr.sum(), 1)
        js_vals.append(_jensen_shannon(ps, pr))
    mean_js = float(np.mean(js_vals)) if js_vals else float("nan")

    return {
        "mean_wasserstein_numeric": mean_w,
        "n_numeric_cols_used": float(len(w_vals)),
        "mean_js_divergence_categorical": mean_js,
        "n_categorical_cols_used": float(len(js_vals)),
    }


def _prepare_xy(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    sub = df[feature_cols + [target_col]].dropna()
    X = pd.get_dummies(sub[feature_cols], dummy_na=False)
    X = _sanitize_feature_columns(X)
    y = sub[target_col].values
    return X, y


def _load_info(info_json: Optional[str] = None) -> dict:
    p = Path(info_json) if info_json else CDT_ROOT / "data" / "nyc_crash" / "info.json"
    if not p.is_absolute():
        p = CDT_ROOT / p
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "| (empty) |\n| --- |\n"
    keys = list(rows[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


def _parse_joint_specs(specs_json_path: Optional[str]) -> List[CMIPairSpec]:
    if not specs_json_path:
        return []
    p = Path(specs_json_path)
    if not p.is_file():
        p = CDT_ROOT / specs_json_path
    if not p.is_file():
        raise SystemExit(f"joint specs json 不存在: {specs_json_path}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit("joint specs json 必须是 list[object]")

    specs: List[CMIPairSpec] = []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        x = str(obj.get("x_col", "")).strip()
        y = str(obj.get("y_col", "")).strip()
        cond = obj.get("cond_cols", [])
        cond_cols = [str(c) for c in cond] if isinstance(cond, list) else []
        if x and y:
            specs.append(CMIPairSpec(x_col=x, y_col=y, cond_cols=cond_cols))
    return specs


def _default_joint_specs(real_df: pd.DataFrame) -> List[CMIPairSpec]:
    """Fallback specs for CMI evaluation when user does not provide JSON."""
    candidates = [
        ("LATITUDE", "LONGITUDE", ["TEMP_C", "prcp"]),
        ("TEMP_C", "prcp", ["coco"]),
        ("DIST_TO_SIGNAL_M", "CTX_DIST_TO_INTERSECTION", ["TEMP_C"]),
    ]
    specs: List[CMIPairSpec] = []
    cols = set(real_df.columns)
    for x, y, cond in candidates:
        if x in cols and y in cols and all(c in cols for c in cond):
            specs.append(CMIPairSpec(x_col=x, y_col=y, cond_cols=cond))
    return specs


def _compute_ranked_summary(
    results: List[Dict[str, Any]],
    task_type: str,
    profile: str,
) -> List[Dict[str, Any]]:
    """Compute ranked model summary based on primary_metrics_profile.

    Tier definitions:
      structural  -> CMI abs error (lower better) + SHD normalized (lower better)
      downstream  -> TSTR avg_score / R2 / F1 (higher better)

    Composite score for ranking: 1.0 = best, 0.0 = worst.
    """
    if not results:
        return []

    def _safe(v: Any) -> float:
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    def _norm_lower(vals: List[float]) -> List[float]:
        """Normalize so that lower raw value → higher score [0,1]."""
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return [float("nan")] * len(vals)
        lo, hi = min(finite), max(finite)
        rng = hi - lo if hi != lo else 1.0
        return [float((hi - v) / rng) if np.isfinite(v) else float("nan") for v in vals]

    def _norm_higher(vals: List[float]) -> List[float]:
        """Normalize so that higher raw value → higher score [0,1]."""
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return [float("nan")] * len(vals)
        lo, hi = min(finite), max(finite)
        rng = hi - lo if hi != lo else 1.0
        return [float((v - lo) / rng) if np.isfinite(v) else float("nan") for v in vals]

    n = len(results)

    # Collect raw metric vectors
    cmi_errs = [_safe(r.get("cmi_abs_error_mean")) for r in results]
    shd_norms = [_safe(r.get("shd_normalized")) for r in results]
    avg_scores = [_safe(r.get("tstr_avg_score")) for r in results]
    if task_type == "regression":
        task_primary = [_safe(r.get("tstr_r2")) for r in results]
    else:
        task_primary = [_safe(r.get("tstr_accuracy")) for r in results]
    # Normalized scores (all in [0,1], higher = better)
    nc_cmi = _norm_lower(cmi_errs)
    nc_shd = _norm_lower(shd_norms)
    nc_avg = _norm_higher(avg_scores)
    nc_task = _norm_higher(task_primary)

    def _avg_finite(*vals: float) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    composite: List[float] = []
    for i in range(n):
        if profile == "structural":
            s = _avg_finite(nc_cmi[i], nc_shd[i])
        elif profile == "downstream":
            s = _avg_finite(nc_avg[i], nc_task[i])
        elif profile == "no_rule":
            s = _avg_finite(nc_cmi[i], nc_shd[i], nc_avg[i], nc_task[i])
        elif profile == "full":
            s = _avg_finite(nc_cmi[i], nc_shd[i], nc_avg[i], nc_task[i])
        else:
            s = _avg_finite(nc_avg[i], nc_task[i])
        composite.append(s)

    ranked = []
    order = sorted(range(n), key=lambda i: (-composite[i] if np.isfinite(composite[i]) else 1.0))
    for rank_pos, idx in enumerate(order, 1):
        r = results[idx]
        ranked.append({
            "rank": rank_pos,
            "file": r.get("file", ""),
            "composite_score": round(composite[idx], 4) if np.isfinite(composite[idx]) else None,
            "cmi_abs_error_mean": r.get("cmi_abs_error_mean"),
            "shd_normalized": r.get("shd_normalized"),
            "tstr_avg_score": r.get("tstr_avg_score"),
            "tstr_r2_or_accuracy": task_primary[idx] if np.isfinite(task_primary[idx]) else None,
        })
    return ranked


def main():
    parser = argparse.ArgumentParser(description="合成数据多维度评估")
    parser.add_argument(
        "--real_test",
        type=str,
        default=str(CDT_ROOT / "synthetic" / "nyc_crash" / "test.csv"),
        help="真实测试集 CSV（可与 synthetic/nyc_crash/test.csv 相同格式）",
    )
    parser.add_argument(
        "--synthetic_dir",
        type=str,
        default=str(CDT_ROOT / "results" / "synthetic"),
        help="合成 CSV 所在目录（评估其中非下划线开头的 .csv）",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="默认从 data/nyc_crash/info.json 读取 target_col",
    )
    parser.add_argument(
        "--info_json",
        type=str,
        default=None,
        help="评估 schema/info.json 路径；2024 源域实验应传 data/nyc_crash_2024/info.json",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        default="*.csv",
        help="仅评估 synthetic_dir 下匹配该 glob 的文件，例如 *_balanced.csv、*_full.csv",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=["regression", "classification"],
        help="可选覆盖任务类型（默认从 data/nyc_crash/info.json 读取）",
    )
    parser.add_argument(
        "--enable_joint_metrics",
        action="store_true",
        help="启用模块1联合分布与因果结构评估（CMI误差 + SHD）",
    )
    parser.add_argument(
        "--joint_specs_json",
        type=str,
        default=None,
        help="CMI规格配置 JSON 路径，格式: [{x_col,y_col,cond_cols:[...]}]",
    )
    parser.add_argument(
        "--real_adj_path",
        type=str,
        default=None,
        help="真实因果邻接矩阵路径（.npy 或 .csv），用于 SHD",
    )
    parser.add_argument(
        "--syn_adj_path",
        type=str,
        default=None,
        help="合成因果邻接矩阵路径（.npy 或 .csv），用于 SHD（固定单文件）",
    )
    parser.add_argument(
        "--syn_adj_pattern",
        type=str,
        default=None,
        help="合成因果邻接路径模板，支持 {file_stem}，用于逐样本 SHD",
    )
    parser.add_argument(
        "--primary_metrics_profile",
        type=str,
        default="auto",
        choices=["auto", "structural", "downstream", "no_rule", "full"],
        help=(
            "主指标配置（决定输出的综合排名依据）：\n"
            "  structural  - 结构层优先（CMI误差 + SHD）\n"
            "  downstream  - 任务层优先（TSTR R2/F1 + avg_score）\n"
            "  no_rule     - 结构层 + 任务层（等效 no_rule）\n"
            "  full        - 结构层 + 任务层均等权重\n"
            "  auto        - 向后兼容旧行为（不生成综合排名）"
        ),
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="输出文件名标签（指定后用 eval_report_{tag}.md 替代时间戳文件名）",
    )
    args = parser.parse_args()

    info = _load_info(args.info_json)
    task_type = str(args.task_type or info.get("task_type", "classification")).lower()
    if task_type not in {"regression", "classification"}:
        raise SystemExit(f"不支持的 task_type: {task_type}，仅支持 regression/classification")
    target_col = args.target_col or info.get("target_col", "NUMBER OF PERSONS INJURED")

    print(f"[config] task_type={task_type}, target_col={target_col}")

    cmi_eval = ConditionalMutualInformationEvaluator(random_state=42)

    real_path = Path(args.real_test)
    if not real_path.is_file():
        real_path = CDT_ROOT / args.real_test
    if not real_path.is_file():
        raise SystemExit(f"找不到测试集: {args.real_test}")

    real_df = _sanitize_feature_columns(pd.read_csv(real_path, low_memory=False))
    print("[real_check] Applying inverse transformation to physical space...")
    real_df = _inverse_transform_continuous(real_df, info)
    syn_folder = Path(args.synthetic_dir)
    if not syn_folder.is_dir():
        syn_folder = CDT_ROOT / args.synthetic_dir
    if not syn_folder.is_dir():
        raise SystemExit(f"合成目录不存在: {args.synthetic_dir}")
    files = sorted(
        f
        for f in syn_folder.glob(args.file_glob)
        if f.is_file() and not f.name.startswith("_")
    )
    if not files:
        print(f"[warn] 目录中无可用合成 CSV: {syn_folder}（已跳过以下划线开头的临时文件）")

    results: List[Dict[str, Any]] = []
    benchmark_details: Dict[str, Dict[str, Any]] = {}
    joint_specs = _parse_joint_specs(args.joint_specs_json)
    if args.enable_joint_metrics and not joint_specs:
        joint_specs = _default_joint_specs(real_df)
        if joint_specs:
            print(f"[joint_metrics] Using {len(joint_specs)} default CMI specs")
        else:
            print("[joint_metrics] No valid default CMI specs found in current columns")

    for fp in files:
        if not fp.is_file():
            continue
        syn = _sanitize_feature_columns(pd.read_csv(fp, low_memory=False))
        print(f"  [inverse_transform] {fp.name}...")
        syn = _inverse_transform_continuous(syn, info)
        stats = _statistical_distances(syn, real_df, info)
        row = {
            "file": fp.name,
            "n_rows": len(syn),
            "mean_wasserstein_numeric": round(stats["mean_wasserstein_numeric"], 6),
            "mean_js_categorical": round(stats["mean_js_divergence_categorical"], 6),
        }

        if args.enable_joint_metrics:
            try:
                if joint_specs:
                    cmi_df = cmi_eval.evaluate_cmi_error(real_df, syn, joint_specs)
                    row["cmi_abs_error_mean"] = round(float(cmi_df["cmi_abs_error"].mean()), 6)
                    row["cmi_rel_error_mean"] = round(float(cmi_df["cmi_rel_error"].mean()), 6)
                else:
                    row["cmi_abs_error_mean"] = float("nan")
                    row["cmi_rel_error_mean"] = float("nan")

                shd_done = False
                if args.real_adj_path and (args.syn_adj_path or args.syn_adj_pattern):
                    syn_adj_path = args.syn_adj_path
                    if args.syn_adj_pattern:
                        syn_adj_path = args.syn_adj_pattern.format(file_stem=fp.stem)
                    if syn_adj_path:
                        shd_info = evaluate_shd_from_files(args.real_adj_path, syn_adj_path)
                        row["shd"] = int(shd_info["shd"])
                        row["shd_normalized"] = round(float(shd_info["shd_normalized"]), 6)
                        shd_done = True
                if not shd_done:
                    row["shd"] = float("nan")
                    row["shd_normalized"] = float("nan")
            except Exception as e:
                row["cmi_abs_error_mean"] = float("nan")
                row["cmi_rel_error_mean"] = float("nan")
                row["shd"] = float("nan")
                row["shd_normalized"] = float("nan")
                row["joint_metrics_error"] = str(e)

        try:
            common = [c for c in syn.columns if c in real_df.columns and c != target_col]
            if target_col not in syn.columns or target_col not in real_df.columns:
                raise ValueError(f"目标列 {target_col} 在合成集或测试集中缺失")

            Xs, ys = _prepare_xy(syn, target_col, common)
            Xt, yt = _prepare_xy(real_df, target_col, common)
            all_cols = sorted(set(Xs.columns) | set(Xt.columns))
            Xs = Xs.reindex(columns=all_cols, fill_value=0)
            Xt = Xt.reindex(columns=all_cols, fill_value=0)

            print(f"[tstr_benchmark] {fp.name}: task_type={task_type}, models=xgboost/random_forest/mlp")
            evaluator = BenchmarkEvaluator(task_type=task_type, random_state=42)
            bench_df, bench_summary = evaluator.evaluate(Xs, ys, Xt, yt)

            benchmark_details[fp.name] = {
                "summary": bench_summary,
                "models": bench_df.to_dict(orient="records"),
            }

            row["tstr_avg_score"] = round(float(bench_summary.get("avg_score_mean", float("nan"))), 6)
            row["tstr_best_model"] = str(bench_summary.get("best_model", ""))
            row["tstr_best_model_score"] = round(
                float(bench_summary.get("best_model_avg_score", float("nan"))),
                6,
            )

            # Keep backward-compatible headline metrics using xgboost row if available.
            xgb_row = bench_df[bench_df["model"] == "xgboost"]
            if not xgb_row.empty and str(xgb_row.iloc[0].get("status", "")) == "ok":
                if task_type == "regression":
                    row["tstr_r2"] = round(float(xgb_row.iloc[0].get("r2", float("nan"))), 6)
                    row["tstr_mse"] = round(float(xgb_row.iloc[0].get("mse", float("nan"))), 6)
                    row["tstr_mae"] = round(float(xgb_row.iloc[0].get("mae", float("nan"))), 6)
                else:
                    row["tstr_f1_macro"] = round(float(xgb_row.iloc[0].get("f1_macro", float("nan"))), 6)
                    row["tstr_f1_micro"] = round(float(xgb_row.iloc[0].get("f1_micro", float("nan"))), 6)
                    row["tstr_accuracy"] = round(float(xgb_row.iloc[0].get("accuracy", float("nan"))), 6)
                    auroc_val = xgb_row.iloc[0].get("auroc", float("nan"))
                    if auroc_val is not None and not (isinstance(auroc_val, float) and np.isnan(auroc_val)):
                        row["tstr_auroc"] = round(float(auroc_val), 6)
            elif task_type == "regression":
                row["tstr_r2"] = float("nan")
                row["tstr_mse"] = float("nan")
                row["tstr_mae"] = float("nan")
            else:
                row["tstr_f1_macro"] = float("nan")
                row["tstr_f1_micro"] = float("nan")
                row["tstr_accuracy"] = float("nan")
            tstr_err = ""
        except Exception as e:
            if task_type == "regression":
                row["tstr_r2"] = float("nan")
                row["tstr_mse"] = float("nan")
                row["tstr_mae"] = float("nan")
            else:
                row["tstr_f1_macro"] = float("nan")
                row["tstr_f1_micro"] = float("nan")
                row["tstr_accuracy"] = float("nan")
            row["tstr_avg_score"] = float("nan")
            row["tstr_best_model"] = ""
            row["tstr_best_model_score"] = float("nan")
            benchmark_details[fp.name] = {
                "summary": {},
                "models": [],
            }
            tstr_err = str(e)

        if tstr_err:
            row["tstr_error"] = tstr_err
        results.append(row)

    out_dir = CDT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _stem = args.output_tag if getattr(args, "output_tag", None) else ts
    md_path = out_dir / f"eval_report_{_stem}.md"
    json_path = out_dir / f"eval_report_{_stem}.json"

    md_body = (
        f"# Synthetic evaluation report\n\n"
        f"- real_test: `{real_path}`\n"
        f"- task_type: `{task_type}`\n"
        f"- target_col: `{target_col}`\n"
        f"- file_glob: `{args.file_glob}`\n"
        f"- primary_metrics_profile: `{args.primary_metrics_profile}`\n\n"
        f"## 三层指标体系说明\n\n"
        f"| 层级 | 指标 | 优先级 | 说明 |\n"
        f"| --- | --- | --- | --- |\n"
        f"| **结构层** | `cmi_abs_error_mean`, `shd_normalized` | 主（structural/no_rule/full 模式） | 联合分布与因果结构一致性（较低=更好） |\n"
        f"| **任务层** | `tstr_avg_score`, `tstr_r2`/`tstr_accuracy` | 主（downstream/no_rule/full 模式） | 下游 TSTR 迁移学习效果（较高=更好） |\n\n"
    )
    md_body += _markdown_table(results)
    md_body += "\n## TSTR Benchmark Details\n\n"
    for fname, detail in benchmark_details.items():
        md_body += f"### {fname}\n\n"
        summary = detail.get("summary", {})
        if summary:
            md_body += "- summary: " + json.dumps(summary, ensure_ascii=False) + "\n\n"
        rows = detail.get("models", [])
        md_body += _markdown_table(rows if isinstance(rows, list) else [])
        md_body += "\n"

    # Add ranked summary section when profile is not 'auto'
    ranked_summary: List[Dict[str, Any]] = []
    if args.primary_metrics_profile != "auto" and results:
        ranked_summary = _compute_ranked_summary(results, task_type, args.primary_metrics_profile)
        md_body += f"\n## 综合排名（profile={args.primary_metrics_profile}）\n\n"
        md_body += (
            "综合分 = 各层归一化分数的平均值（1.0=最优，0.0=最差）。"
            "排名依据由 `--primary_metrics_profile` 决定。\n\n"
        )
        md_body += _markdown_table(ranked_summary)

    md_path.write_text(md_body, encoding="utf-8")

    payload = {
        "generated_utc": ts,
        "primary_metrics_profile": args.primary_metrics_profile,
        "two_tier_methodology": {
            "structural_layer": ["cmi_abs_error_mean", "shd_normalized"],
            "task_layer": ["tstr_avg_score", "tstr_r2", "tstr_accuracy"],
        },
        "real_test": str(real_path),
        "task_type": task_type,
        "target_col": target_col,
        "file_glob": args.file_glob,
        "joint_metrics_enabled": bool(args.enable_joint_metrics),
        "joint_specs": [
            {"x_col": s.x_col, "y_col": s.y_col, "cond_cols": list(s.cond_cols)}
            for s in joint_specs
        ],
        "rows": results,
        "ranked_summary": ranked_summary,
        "tstr_benchmark_details": benchmark_details,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 同时写入固定文件名便于论文迭代覆盖（只增不删：保留带时间戳文件）
    latest_md = out_dir / "eval_report_latest.md"
    latest_json = out_dir / "eval_report_latest.json"
    latest_md.write_text(md_body, encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[write] {md_path}\n[write] {json_path}\n[write] {latest_md}")


if __name__ == "__main__":
    main()
