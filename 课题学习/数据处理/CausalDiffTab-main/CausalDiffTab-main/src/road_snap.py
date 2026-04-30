"""
road_snap.py — 道路约束空间候选集与事故点吸附模块
=====================================================
实现 Section 1.4 中的道路约束经纬度生成策略（Phase 1: 后处理吸附）：

1. 从 OSM GraphML 加载路网，提取候选点集合：
   - 路口节点 (intersection nodes)
   - 每条 segment 中心点
   - 沿 segment 中心线插值点（可选）

2. build_road_candidate_set(graphml_path, ...)
   → 返回候选点数组 + BallTree 索引

3. snap_points_to_road(lats, lons, tree, candidates, ...)
   → 将任意经纬度吸附到最近候选点，支持小范围高斯扰动

4. validate_points(lats, lons, tree, candidates, ...)
   → 空间合法性校验：距最近道路距离、NYC 边界

5. postprocess_latlon_df(df, ...)
   → 对 DataFrame 中 LATITUDE/LONGITUDE 列执行完整后处理

用法（Phase 1 后处理，不修改扩散模型）：
    from src.road_snap import build_road_candidate_set, postprocess_latlon_df
    tree, candidates, meta = build_road_candidate_set(graphml_path)
    df = postprocess_latlon_df(df, tree, candidates, jitter_m=10.0)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "RoadCandidateSet",
    "build_road_candidate_set",
    "snap_points_to_road",
    "validate_points",
    "postprocess_latlon_df",
]

# NYC 地理边界（宽松）
NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.65

# UTM 18N（纽约）投影 EPSG 代码
NYC_UTM_CRS = "EPSG:32618"
WGS84_CRS   = "EPSG:4326"

# 沿 segment 插值步长（米）：每 50m 生成一个插值候选点
INTERP_STEP_M = 50.0

# 吸附距离阈值（米）：超过此距离视为"偏离道路"
SNAP_FAR_THRESHOLD_M = 200.0


@dataclass
class RoadCandidateSet:
    """道路候选点集合（经纬度坐标 + BallTree 索引）。"""

    # shape (N, 2)，列顺序：[lat, lon]（用于 BallTree haversine 距离）
    latlon: np.ndarray

    # BallTree 索引（haversine 距离，单位 radian）
    tree: object  # sklearn BallTree

    # UTM 坐标 (米制)，用于欧式距离计算
    utm_xy: np.ndarray  # shape (N, 2), [x_east, y_north]

    # 每个候选点对应的 OSM edge_id（(u,v,key) 三元组的索引，-1 表示路口节点）
    edge_idx: np.ndarray  # shape (N,), int

    # 元信息
    meta: dict = field(default_factory=dict)


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """向量化 Haversine 距离（米）。"""
    R = 6_371_000.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _deg_to_rad_latlon(latlon_deg: np.ndarray) -> np.ndarray:
    """(N,2) lat/lon 度数 → 弧度，供 BallTree haversine 用。"""
    return np.radians(latlon_deg)


def build_road_candidate_set(
    graphml_path: str | Path,
    interp_step_m: float = INTERP_STEP_M,
    cache_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> RoadCandidateSet:
    """
    从 OSM GraphML 构建道路候选点集合。

    参数
    ----
    graphml_path : OSM 路网 .graphml 文件路径
    interp_step_m : 沿 segment 中心线插值步长（米），0 = 只取路口 + 中心点
    cache_path : 若指定，则加载/保存 .npz 缓存以加速重复调用
    verbose : 是否打印进度

    返回
    ----
    RoadCandidateSet
    """
    from sklearn.neighbors import BallTree
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import osmnx as ox

    graphml_path = Path(graphml_path)

    # ── 缓存读取 ───────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"[road_snap] 加载候选点缓存: {cache_path.name}")
            data = np.load(cache_path, allow_pickle=True)
            latlon   = data["latlon"]
            utm_xy   = data["utm_xy"]
            edge_idx = data["edge_idx"]
            tree     = BallTree(_deg_to_rad_latlon(latlon), metric="haversine")
            meta     = json.loads(str(data["meta"]))
            return RoadCandidateSet(latlon=latlon, tree=tree, utm_xy=utm_xy,
                                    edge_idx=edge_idx, meta=meta)

    if verbose:
        print(f"[road_snap] 加载路网: {graphml_path.name} ...")

    # ── 加载路网 ──────────────────────────────────────────────────────────
    def _compat_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("yes", "true", "1", "on")

    G = ox.load_graphml(
        str(graphml_path),
        edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
        graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
    )
    G_proj = ox.project_graph(G, to_crs=NYC_UTM_CRS)

    # ── 路口节点候选点 ────────────────────────────────────────────────────
    node_lats, node_lons = [], []
    node_xs, node_ys     = [], []
    node_edge_idx        = []

    for nid, attrs in G_proj.nodes(data=True):
        # 原始地理坐标（WGS84）
        orig = G.nodes[nid]
        lat  = float(orig.get("y", 0))
        lon  = float(orig.get("x", 0))
        if not (NYC_LAT_MIN <= lat <= NYC_LAT_MAX and NYC_LON_MIN <= lon <= NYC_LON_MAX):
            continue
        node_lats.append(lat)
        node_lons.append(lon)
        node_xs.append(float(attrs.get("x", 0)))
        node_ys.append(float(attrs.get("y", 0)))
        node_edge_idx.append(-1)  # -1 = 路口节点

    if verbose:
        print(f"[road_snap] 路口节点: {len(node_lats):,} 个")

    # ── Segment 候选点（中心点 + 插值点）────────────────────────────────
    seg_lats, seg_lons = [], []
    seg_xs, seg_ys     = [], []
    seg_edge_idx       = []

    edges_data = list(G.edges(data=True, keys=True))
    for eidx, (u, v, key, attrs) in enumerate(edges_data):
        geom = attrs.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            coords_wgs84 = list(geom.coords)  # [(lon, lat), ...]
        else:
            # 直接用起止节点
            u_attrs = G.nodes[u]
            v_attrs = G.nodes[v]
            lon_u, lat_u = float(u_attrs.get("x", 0)), float(u_attrs.get("y", 0))
            lon_v, lat_v = float(v_attrs.get("x", 0)), float(v_attrs.get("y", 0))
            coords_wgs84 = [(lon_u, lat_u), (lon_v, lat_v)]

        if len(coords_wgs84) < 2:
            continue

        # 投影坐标（用于米制插值）
        try:
            u_proj = G_proj.nodes[u]
            v_proj = G_proj.nodes[v]
            x_u, y_u = float(u_proj.get("x", 0)), float(u_proj.get("y", 0))
            x_v, y_v = float(v_proj.get("x", 0)), float(v_proj.get("y", 0))
        except KeyError:
            continue

        # segment 中心点（WGS84 近似中点）
        n = len(coords_wgs84)
        mid = coords_wgs84[n // 2]
        mid_lon, mid_lat = mid[0], mid[1]
        if (NYC_LAT_MIN <= mid_lat <= NYC_LAT_MAX and
                NYC_LON_MIN <= mid_lon <= NYC_LON_MAX):
            seg_lats.append(mid_lat)
            seg_lons.append(mid_lon)
            seg_xs.append((x_u + x_v) / 2)
            seg_ys.append((y_u + y_v) / 2)
            seg_edge_idx.append(eidx)

        # 沿 segment 插值点
        if interp_step_m > 0:
            seg_len = np.hypot(x_v - x_u, y_v - y_u)
            if seg_len > interp_step_m * 2:
                n_pts = int(seg_len // interp_step_m) - 1
                ts = np.linspace(0, 1, n_pts + 2)[1:-1]
                # 沿 WGS84 线性插值（NYC 区域误差可接受）
                lons_arr = np.array([c[0] for c in coords_wgs84])
                lats_arr = np.array([c[1] for c in coords_wgs84])
                seg_fracs = np.linspace(0, 1, len(coords_wgs84))
                for t in ts:
                    # 按 t 在 segment 中的比例插值
                    interp_lon = float(np.interp(t, seg_fracs, lons_arr))
                    interp_lat = float(np.interp(t, seg_fracs, lats_arr))
                    interp_x   = x_u + t * (x_v - x_u)
                    interp_y   = y_u + t * (y_v - y_u)
                    if (NYC_LAT_MIN <= interp_lat <= NYC_LAT_MAX and
                            NYC_LON_MIN <= interp_lon <= NYC_LON_MAX):
                        seg_lats.append(interp_lat)
                        seg_lons.append(interp_lon)
                        seg_xs.append(interp_x)
                        seg_ys.append(interp_y)
                        seg_edge_idx.append(eidx)

    if verbose:
        print(f"[road_snap] Segment 候选点: {len(seg_lats):,} 个（含中心+插值）")

    # ── 合并所有候选点 ────────────────────────────────────────────────────
    all_lats     = np.array(node_lats + seg_lats,     dtype=np.float64)
    all_lons     = np.array(node_lons + seg_lons,     dtype=np.float64)
    all_xs       = np.array(node_xs   + seg_xs,       dtype=np.float64)
    all_ys       = np.array(node_ys   + seg_ys,       dtype=np.float64)
    all_edge_idx = np.array(node_edge_idx + seg_edge_idx, dtype=np.int32)

    latlon   = np.column_stack([all_lats, all_lons])
    utm_xy   = np.column_stack([all_xs,   all_ys])

    # ── 构建 BallTree ────────────────────────────────────────────────────
    tree = BallTree(_deg_to_rad_latlon(latlon), metric="haversine")

    meta = {
        "graphml": str(graphml_path),
        "n_nodes": len(node_lats),
        "n_seg":   len(seg_lats),
        "n_total": len(all_lats),
        "interp_step_m": interp_step_m,
    }

    if verbose:
        print(f"[road_snap] 总候选点: {len(all_lats):,}，BallTree 构建完成")

    # ── 可选缓存保存 ───────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            latlon=latlon,
            utm_xy=utm_xy,
            edge_idx=all_edge_idx,
            meta=json.dumps(meta),
        )
        if verbose:
            print(f"[road_snap] 候选点缓存已保存: {cache_path}")

    return RoadCandidateSet(latlon=latlon, tree=tree, utm_xy=utm_xy,
                            edge_idx=all_edge_idx, meta=meta)


def snap_points_to_road(
    lats: np.ndarray,
    lons: np.ndarray,
    rcs: RoadCandidateSet,
    jitter_m: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将经纬度点吸附到最近道路候选点，并添加小范围高斯扰动。

    参数
    ----
    lats, lons : 输入坐标（各 shape (N,)）
    rcs        : RoadCandidateSet
    jitter_m   : 高斯扰动标准差（米），0 = 不扰动
    seed       : 随机种子

    返回
    ----
    snapped_lats, snapped_lons : 吸附后坐标
    snap_dist_m                : 每个点到最近候选点的距离（米）
    """
    lats  = np.asarray(lats,  dtype=np.float64).ravel()
    lons  = np.asarray(lons,  dtype=np.float64).ravel()
    query = _deg_to_rad_latlon(np.column_stack([lats, lons]))

    dists_rad, idxs = rcs.tree.query(query, k=1)
    idxs      = idxs.ravel()
    dists_rad = dists_rad.ravel()

    # 弧度 → 米
    R = 6_371_000.0
    snap_dist_m = dists_rad * R

    snapped_lats = rcs.latlon[idxs, 0].copy()
    snapped_lons = rcs.latlon[idxs, 1].copy()

    # 高斯扰动（UTM 米制坐标下扰动，再转回 WGS84 近似）
    if jitter_m > 0:
        rng = np.random.default_rng(seed)
        dx  = rng.normal(0, jitter_m, size=len(idxs))
        dy  = rng.normal(0, jitter_m, size=len(idxs))
        utm_x = rcs.utm_xy[idxs, 0] + dx
        utm_y = rcs.utm_xy[idxs, 1] + dy
        # 近似转换 UTM → WGS84（NYC 区域误差 < 1m）
        # 1° lat ≈ 111_000 m；1° lon ≈ 111_000 * cos(lat) m
        lat_ref = snapped_lats
        d_lat   = dy / 111_000.0
        d_lon   = dx / (111_000.0 * np.cos(np.radians(lat_ref)))
        snapped_lats = snapped_lats + d_lat
        snapped_lons = snapped_lons + d_lon

    return snapped_lats, snapped_lons, snap_dist_m


def validate_points(
    lats: np.ndarray,
    lons: np.ndarray,
    rcs: RoadCandidateSet,
    far_threshold_m: float = SNAP_FAR_THRESHOLD_M,
) -> dict:
    """
    空间合法性校验，返回统计字典。

    校验项目：
    - in_nyc_bounds     : 是否在 NYC 地理矩形边界内
    - dist_to_road_m    : 到最近候选点距离（米）
    - is_on_road        : dist < far_threshold_m
    - off_road_ratio    : 偏离比例
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()

    in_bounds = (
        (lats >= NYC_LAT_MIN) & (lats <= NYC_LAT_MAX) &
        (lons >= NYC_LON_MIN) & (lons <= NYC_LON_MAX)
    )

    query = _deg_to_rad_latlon(np.column_stack([lats, lons]))
    dists_rad, _ = rcs.tree.query(query, k=1)
    R = 6_371_000.0
    dist_m = dists_rad.ravel() * R

    is_on_road = dist_m < far_threshold_m

    return {
        "n_total":        len(lats),
        "in_nyc_bounds":  int(in_bounds.sum()),
        "in_nyc_pct":     float(in_bounds.mean()),
        "dist_to_road_m": dist_m,
        "mean_dist_m":    float(dist_m.mean()),
        "median_dist_m":  float(np.median(dist_m)),
        "p95_dist_m":     float(np.percentile(dist_m, 95)),
        "on_road_n":      int(is_on_road.sum()),
        "on_road_pct":    float(is_on_road.mean()),
        "off_road_ratio": float((~is_on_road).mean()),
    }


def postprocess_latlon_df(
    df: pd.DataFrame,
    rcs: RoadCandidateSet,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    jitter_m: float = 10.0,
    snap_threshold_m: float = SNAP_FAR_THRESHOLD_M,
    seed: int = 42,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    对 DataFrame 中的 LATITUDE/LONGITUDE 列执行道路吸附后处理。

    只吸附超过 snap_threshold_m 的点（已在路网附近的点保持轻微扰动）。

    返回
    ----
    后处理后的 DataFrame（含新列 SNAP_DIST_M）
    """
    if not inplace:
        df = df.copy()

    lats = df[lat_col].to_numpy(dtype=float, na_value=np.nan)
    lons = df[lon_col].to_numpy(dtype=float, na_value=np.nan)

    valid = np.isfinite(lats) & np.isfinite(lons)
    if valid.sum() == 0:
        df["SNAP_DIST_M"] = np.nan
        return df

    snapped_lats, snapped_lons, dist_m = snap_points_to_road(
        lats[valid], lons[valid], rcs, jitter_m=jitter_m, seed=seed
    )

    new_lats = lats.copy()
    new_lons = lons.copy()
    new_lats[valid] = snapped_lats
    new_lons[valid] = snapped_lons

    snap_dist_full = np.full(len(df), np.nan)
    snap_dist_full[valid] = dist_m

    df[lat_col]       = new_lats
    df[lon_col]       = new_lons
    df["SNAP_DIST_M"] = snap_dist_full

    if verbose:
        n_valid  = int(valid.sum())
        far_mask = dist_m > snap_threshold_m
        print(
            f"[road_snap] 吸附 {n_valid:,} 个有效点 | "
            f"偏离道路(>{snap_threshold_m:.0f}m): {far_mask.sum():,} ({far_mask.mean():.1%}) | "
            f"吸附后均值距离: {snapped_lats.mean():.5f}° (lat)"
        )
        stats = validate_points(new_lats[valid], new_lons[valid], rcs)
        print(
            f"[road_snap] 吸附后 | 均值距道路: {stats['mean_dist_m']:.1f}m | "
            f"落路率: {stats['on_road_pct']:.1%} | 偏离率: {stats['off_road_ratio']:.1%}"
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI 独立运行：验证候选集构建
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="构建并验证道路候选点集合")
    p.add_argument("--graphml",  required=True, help="OSM .graphml 文件路径")
    p.add_argument("--cache",    default=None,  help="候选点缓存 .npz 路径（可选）")
    p.add_argument("--csv",      default=None,  help="输入 CSV（含 LATITUDE/LONGITUDE）做验证")
    p.add_argument("--out_csv",  default=None,  help="后处理后输出 CSV")
    p.add_argument("--jitter_m", type=float, default=10.0, help="高斯扰动标准差（米）")
    p.add_argument("--interp_step_m", type=float, default=50.0, help="插值步长（米），0=不插值")
    args = p.parse_args()

    rcs = build_road_candidate_set(
        graphml_path=args.graphml,
        interp_step_m=args.interp_step_m,
        cache_path=args.cache,
        verbose=True,
    )
    print(f"\n候选点集合统计: {rcs.meta}")

    if args.csv:
        df = pd.read_csv(args.csv)
        stats_before = validate_points(df["LATITUDE"].values, df["LONGITUDE"].values, rcs)
        print(f"\n吸附前: 均值距道路 {stats_before['mean_dist_m']:.1f}m, "
              f"落路率 {stats_before['on_road_pct']:.1%}")
        df = postprocess_latlon_df(df, rcs, jitter_m=args.jitter_m)
        stats_after = validate_points(df["LATITUDE"].values, df["LONGITUDE"].values, rcs)
        print(f"吸附后: 均值距道路 {stats_after['mean_dist_m']:.1f}m, "
              f"落路率 {stats_after['on_road_pct']:.1%}")
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"已保存: {args.out_csv}")
