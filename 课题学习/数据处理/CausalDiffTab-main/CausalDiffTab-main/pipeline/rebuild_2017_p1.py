"""
P1-1 + P1-2 2017 数据集修复管线
=====================================
P1-1: 从本地 PBF 重新提取 OSM lanes tag，修复 INFERRED_LANES 退化问题
P1-2: 对 DIST_TO_SIGNAL_M 做 P99.9 winsorize，消除极端异常值

操作链：
  1. 加载 nyc_2017_pristine_v9.csv（物理空间）
  2. [P1-1] 用 pyrosm 从 PBF 提取道路 lanes 属性，用 BallTree 最近邻匹配事故点
  3. [P1-2] DIST_TO_SIGNAL_M P99.9 winsorize
  4. 保存修复后的 pristine CSV（保留原文件备份）
  5. 重跑 src/data_processor.py（归一化 → processed_hierarchical.csv）
  6. 重跑 src/prepare_dataset.py（→ data/nyc_crash/train.csv + test.csv）

用法:
  cd CausalDiffTab-main
  python pipeline/rebuild_2017_p1.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # CausalDiffTab-main/
TABDDPM = Path(__file__).resolve().parents[3] / "tab-ddpm-main"  # ../tab-ddpm-main/
PRISTINE = TABDDPM / "nyc_2017_pristine_v9.csv"
PBF_PATH = TABDDPM / "osmdata" / "new-york-180101-internal.osm.pbf"
PYTHON   = sys.executable

# ──────────────────────────────────────────────────────────────────────────────
# Step 1：PBF → OSM lanes tag（pyrosm + BallTree 最近邻匹配）
# ──────────────────────────────────────────────────────────────────────────────

def extract_osm_lanes(pbf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    用 pyrosm 从 PBF 提取所有 driving 道路的中心坐标与 lanes tag。
    返回 DataFrame: ['lon', 'lat', 'lanes_raw', 'lanes_int']
    """
    import pyrosm
    import geopandas as gpd

    print(f"  [PBF] 加载 {pbf_path.name} ...")
    osm = pyrosm.OSM(str(pbf_path))
    # 提取 driving 路网 edges（包含 lanes tag）
    network = osm.get_network(network_type="driving", nodes=True)
    if network is None:
        raise RuntimeError("PBF 中未提取到 driving 路网")
    _, edges = network
    if edges is None or len(edges) == 0:
        raise RuntimeError("PBF 中未提取到 driving 路网 edges")

    edges = edges.to_crs("EPSG:4326")
    print(f"  [PBF] 提取到 {len(edges):,} 条道路边")

    # 提取 lanes tag
    lanes_raw = edges.get("lanes", pd.Series([None]*len(edges), index=edges.index))
    # 有效 lanes tag 的比例
    valid_lanes = lanes_raw.notna() & (lanes_raw != "") & (lanes_raw != "None")
    print(f"  [PBF] lanes tag 非空率: {valid_lanes.mean():.1%} ({valid_lanes.sum():,}/{len(edges):,})")

    def _parse_lanes(v) -> int | None:
        """解析 lanes tag 为整数（支持 '2', '2;3', [2,3] 等）"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip()
        if not s or s.lower() in ("none", "nan", ""):
            return None
        # 取第一个数字
        import re
        nums = re.findall(r"\d+", s)
        if nums:
            return int(nums[0])
        return None

    # 取每条边的中心点坐标
    centroids = edges.geometry.centroid
    df_roads = pd.DataFrame({
        "lon": centroids.x.values,
        "lat": centroids.y.values,
        "lanes_raw": lanes_raw.values,
    })
    df_roads["lanes_int"] = df_roads["lanes_raw"].apply(_parse_lanes)

    # 只保留有有效 lanes tag 的行（减少 BallTree 匹配成本）
    df_with_lanes = df_roads[df_roads["lanes_int"].notna()].copy()
    print(f"  [PBF] 有效 lanes tag 道路: {len(df_with_lanes):,}")
    return df_roads, df_with_lanes


def match_lanes_to_accidents(
    df_crash: pd.DataFrame,
    df_roads: pd.DataFrame,
    df_road_lanes: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    用 BallTree(haversine) 为每个事故点找最近道路。
    优先找最近的有 lanes tag 的道路；若最近 1km 内无 lanes tag，则回退到推断值。
    返回: (OSM_LANES_TAG_series, INFERRED_LANES_series)
    """
    lat_r = np.deg2rad(df_crash["LATITUDE"].values).reshape(-1, 1)
    lon_r = np.deg2rad(df_crash["LONGITUDE"].values).reshape(-1, 1)
    crash_coords = np.hstack([lat_r, lon_r])

    # BallTree on roads with valid lanes
    if len(df_road_lanes) > 0:
        road_lat_r = np.deg2rad(df_road_lanes["lat"].values).reshape(-1, 1)
        road_lon_r = np.deg2rad(df_road_lanes["lon"].values).reshape(-1, 1)
        tree_lanes = BallTree(np.hstack([road_lat_r, road_lon_r]), metric="haversine")
        dist_lanes, idx_lanes = tree_lanes.query(crash_coords, k=1)
        dist_lanes_m = dist_lanes[:, 0] * 6371000  # 转米
        matched_lanes = df_road_lanes["lanes_int"].iloc[idx_lanes[:, 0]].values
    else:
        dist_lanes_m = np.full(len(df_crash), 99999.0)
        matched_lanes = np.full(len(df_crash), np.nan)

    # BallTree on all roads (for highway type inference when no lanes tag nearby)
    if "osm_type" in df_roads.columns or "highway" in df_roads.columns:
        hw_col = "highway" if "highway" in df_roads.columns else "osm_type"
        df_all = df_roads.dropna(subset=["lon", "lat"]).copy()
    else:
        df_all = df_roads.copy()
    road_lat_r_all = np.deg2rad(df_all["lat"].values).reshape(-1, 1)
    road_lon_r_all = np.deg2rad(df_all["lon"].values).reshape(-1, 1)
    tree_all = BallTree(np.hstack([road_lat_r_all, road_lon_r_all]), metric="haversine")
    dist_all, idx_all = tree_all.query(crash_coords, k=1)

    def _infer_lanes(hw: str | None) -> int:
        if hw is None:
            return 1
        h = str(hw).lower()
        if "motorway" in h or "trunk" in h:
            return 3
        if "primary" in h or "secondary" in h:
            return 2
        return 1

    hw_col = None
    for c in ("highway", "osm_type", "lanes_raw"):
        if c in df_all.columns:
            hw_col = c
            break

    # 组装结果
    osm_lanes_tag = np.full(len(df_crash), np.nan, dtype=float)
    inferred_lanes = np.zeros(len(df_crash), dtype=int)

    MAX_DIST_M = 500  # 如果最近有 lanes tag 的道路 <= 500m，就用它

    for i in range(len(df_crash)):
        if dist_lanes_m[i] <= MAX_DIST_M and not np.isnan(matched_lanes[i]):
            osm_lanes_tag[i] = matched_lanes[i]
            inferred_lanes[i] = int(matched_lanes[i])
        else:
            # 没有近距离的 lanes tag，用 highway type 推断
            near_row = df_all.iloc[idx_all[i, 0]] if len(df_all) > 0 else {}
            hw = near_row.get(hw_col, "") if hw_col else ""
            inferred_lanes[i] = _infer_lanes(hw)

    tag_rate = np.isfinite(osm_lanes_tag).mean()
    print(f"  [match] OSM_LANES_TAG 非空率: {tag_rate:.1%}")
    print(f"  [match] INFERRED_LANES 分布: {pd.Series(inferred_lanes).value_counts().sort_index().to_dict()}")

    return pd.Series(osm_lanes_tag, index=df_crash.index), pd.Series(inferred_lanes, index=df_crash.index)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2：DIST_TO_SIGNAL_M winsorize
# ──────────────────────────────────────────────────────────────────────────────

def winsorize_dist_signal(series: pd.Series, quantile: float = 0.999) -> tuple[pd.Series, dict]:
    """P99.9 winsorize，返回处理后的 Series 和统计摘要"""
    q_val = series.quantile(quantile)
    before = {
        "mean":   round(series.mean(), 2),
        "median": round(series.median(), 2),
        "p99":    round(series.quantile(0.99), 2),
        "p999":   round(q_val, 2),
        "max":    round(series.max(), 2),
        "n_extreme": int((series > q_val).sum()),
    }
    clipped = series.clip(upper=q_val)
    after = {
        "mean":   round(clipped.mean(), 2),
        "median": round(clipped.median(), 2),
        "p99":    round(clipped.quantile(0.99), 2),
        "p999":   round(clipped.quantile(quantile), 2),
        "max":    round(clipped.max(), 2),
        "n_extreme": 0,
    }
    print(f"  [P1-2] DIST_TO_SIGNAL_M winsorize @ P{quantile*100:.1f}={before['p999']:.1f}m")
    print(f"    Before: mean={before['mean']}, median={before['median']}, p99={before['p99']}, max={before['max']} ({before['n_extreme']} extreme)")
    print(f"    After:  mean={after['mean']}, median={after['median']}, p99={after['p99']}, max={after['max']}")
    return clipped, {"before": before, "after": after}


# ──────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  P1-1 + P1-2 2017 数据修复管线")
    print("=" * 64)

    # 备份原始 pristine
    backup = PRISTINE.with_name("nyc_2017_pristine_v9_backup_p1.csv")
    if not backup.exists():
        print(f"\n[0] 备份原始 pristine → {backup.name}")
        shutil.copy2(str(PRISTINE), str(backup))
    else:
        print(f"\n[0] 备份已存在: {backup.name}")

    # 加载 pristine
    print(f"\n[1] 加载 pristine: {PRISTINE.name} ...")
    df = pd.read_csv(str(PRISTINE))
    print(f"    → {len(df):,} 行")

    # ── P1-1: 提取 lanes ──────────────────────────────────────────────────────
    print(f"\n[2] P1-1: 从 PBF 提取 OSM lanes tag ...")
    df_roads, df_road_lanes = extract_osm_lanes(PBF_PATH)

    osm_lanes_tag_before = df["OSM_LANES_TAG"].notna().mean() if "OSM_LANES_TAG" in df.columns else 0.0
    inferred_before = df["INFERRED_LANES"].value_counts().sort_index().to_dict() if "INFERRED_LANES" in df.columns else {}
    print(f"    修复前 OSM_LANES_TAG 非空率: {osm_lanes_tag_before:.1%}")
    print(f"    修复前 INFERRED_LANES 分布: {inferred_before}")

    valid_mask = df["LATITUDE"].notna() & df["LONGITUDE"].notna()
    df_valid = df[valid_mask].copy()

    print(f"    匹配 {len(df_valid):,} 个有效坐标点...")
    osm_tag, inferred = match_lanes_to_accidents(df_valid, df_roads, df_road_lanes)

    df.loc[valid_mask, "OSM_LANES_TAG"]   = osm_tag.values
    df.loc[valid_mask, "INFERRED_LANES"]  = inferred.values

    print(f"    修复后 OSM_LANES_TAG 非空率: {df['OSM_LANES_TAG'].notna().mean():.1%}")
    print(f"    修复后 INFERRED_LANES 分布: {df['INFERRED_LANES'].value_counts().sort_index().to_dict()}")

    # ── P1-2: winsorize DIST_TO_SIGNAL_M ─────────────────────────────────────
    print(f"\n[3] P1-2: DIST_TO_SIGNAL_M winsorize ...")
    if "DIST_TO_SIGNAL_M" in df.columns:
        dist_series = pd.to_numeric(df["DIST_TO_SIGNAL_M"], errors="coerce")
        clipped, stats = winsorize_dist_signal(dist_series)
        df["DIST_TO_SIGNAL_M"] = clipped
    else:
        print("    ⚠ 列不存在，跳过")
        stats = {}

    # ── 修复已知 BOM/前缀问题的列名 ──────────────────────────────────────────
    # pristine_v9 的第一列有 BOM 或 "cc" 前缀变成 "ccCRASH DATE"，
    # data_processor.py 的 process_temporal_features 直接用 "CRASH DATE"，需修正。
    col_renames = {}
    for c in df.columns:
        clean = c.lstrip("\ufeff").lstrip("\xcc").strip()
        if clean != c:
            col_renames[c] = clean
    # 也处理 ccCRASH DATE 这种情况
    if "ccCRASH DATE" in df.columns:
        col_renames["ccCRASH DATE"] = "CRASH DATE"
    if col_renames:
        df.rename(columns=col_renames, inplace=True)
        print(f"    列名修复: {col_renames}")

    # ── 保存修复后 pristine ───────────────────────────────────────────────────
    print(f"\n[4] 保存修复后 pristine → {PRISTINE.name} ...")
    df.to_csv(str(PRISTINE), index=False)
    print(f"    ✓ 已保存 ({len(df):,} 行)")

    # ── 重跑 data_processor.py ───────────────────────────────────────────────
    data_proc = ROOT / "src" / "data_processor.py"
    out_dir   = ROOT / "data" / "processed"
    print(f"\n[5] 重跑 data_processor.py ...")
    # data_processor.py 默认 input 为 tab-ddpm-main/data/processed/nyc_2017_pristine_v9.csv
    # 但实际 pristine 在 tab-ddpm-main/nyc_2017_pristine_v9.csv，需显式指定
    cmd = [
        PYTHON, str(data_proc),
        "--input_csv",   str(PRISTINE),
        "--output_dir",  str(out_dir),
        "--norm_method", "quantile",
        "--no_export_npy",  # 跳过 npy（prepare_dataset.py 会单独处理）
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    # ── 重跑 prepare_dataset.py ───────────────────────────────────────────────
    prep_ds = ROOT / "src" / "prepare_dataset.py"
    notears = ROOT / "configs" / "causal_matrix_notears_mlp.npy"
    print(f"\n[6] 重跑 prepare_dataset.py ...")
    cmd2 = [
        PYTHON, str(prep_ds),
        "--input_csv",          str(out_dir / "processed_hierarchical.csv"),
        "--column_groups_json", str(out_dir / "column_groups.json"),
        "--notears_npy",        str(notears),
        "--output_base",        str(ROOT),
    ]
    subprocess.run(cmd2, cwd=str(ROOT), check=True)

    # ── 验证最终结果 ──────────────────────────────────────────────────────────
    print(f"\n[7] 验证 data/nyc_crash/train.csv ...")
    train = pd.read_csv(str(ROOT / "data" / "nyc_crash" / "train.csv"))
    print(f"    train.csv: {len(train):,} 行")
    if "INFERRED_LANES" in train.columns:
        print(f"    INFERRED_LANES unique values (scaled): {sorted(train['INFERRED_LANES'].unique())[:10]}")
        print(f"    INFERRED_LANES std: {train['INFERRED_LANES'].std():.4f}")
    if "DIST_TO_SIGNAL_M" in train.columns:
        print(f"    DIST_TO_SIGNAL_M range (scaled): [{train['DIST_TO_SIGNAL_M'].min():.4f}, {train['DIST_TO_SIGNAL_M'].max():.4f}]")

    print("\n" + "=" * 64)
    print("  P1 修复完成！")
    print("  请重跑 pipeline/analyze_drift.py 查看 JS 散度变化")
    print("=" * 64)

    # 输出摘要供日志记录
    print("\n### P1-1 修复摘要")
    print(f"  修复前 OSM_LANES_TAG 非空率: {osm_lanes_tag_before:.1%}")
    print(f"  修复后 OSM_LANES_TAG 非空率: {df['OSM_LANES_TAG'].notna().mean():.1%}")
    print(f"  修复前 INFERRED_LANES 分布: {inferred_before}")
    print(f"  修复后 INFERRED_LANES 分布: {df['INFERRED_LANES'].value_counts().sort_index().to_dict()}")
    print("\n### P1-2 修复摘要")
    if stats:
        b, a = stats["before"], stats["after"]
        print(f"  Before: mean={b['mean']}, median={b['median']}, P99.9={b['p999']}, max={b['max']}, n_extreme={b['n_extreme']}")
        print(f"  After:  mean={a['mean']}, median={a['median']}, P99.9={a['p999']}, max={a['max']}")


if __name__ == "__main__":
    main()
