"""
将各年份的 OSM PBF 文件转换为 GraphML 格式 + CSV 导出
=======================================================
用法:
  # 转换所有年份（2024, 2025）
  python pipeline/convert_pbf_to_year_graphml.py

  # 转换指定年份
  python pipeline/convert_pbf_to_year_graphml.py --years 2024

  # 强制重新生成（覆盖已有 graphml）
  python pipeline/convert_pbf_to_year_graphml.py --force

输入目录结构:
  raw_data/osm/2024/alabama-250101.osm.pbf
  raw_data/osm/2025/new-york-260101.osm.pbf  (或已存在 nyc_drive_graph.graphml)

输出（每年份目录下）:
  raw_data/osm/{year}/nyc_drive_graph.graphml   ← osmnx 需要的路网图
  raw_data/osm/{year}/nyc_nodes.csv             ← 节点CSV（osmid, lat, lon, highway）
  raw_data/osm/{year}/nyc_edges.csv             ← 边CSV（u, v, highway, lanes, oneway）
  raw_data/osm/{year}/nyc_traffic_signals.geojson ← 信号灯（可选）
  raw_data/osm/{year}/meta.json                 ← 元信息
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
OSM_ROOT = ROOT / "raw_data" / "osm"

# NYC 边界（含5个行政区，适度放宽）
BBOX = dict(north=40.92, south=40.48, east=-73.68, west=-74.28)


# ──────────────────────────────────────────────────────────────────────────────
# 核心转换函数
# ──────────────────────────────────────────────────────────────────────────────

def convert_pbf_to_graphml(
    pbf_path: Path,
    out_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    """
    将单个 PBF 文件转换为 GraphML 并导出节点/边 CSV。
    返回输出 graphml 路径，失败时返回 None。
    """
    import osmnx as ox

    graphml_path = out_dir / "nyc_drive_graph.graphml"
    nodes_csv    = out_dir / "nyc_nodes.csv"
    edges_csv    = out_dir / "nyc_edges.csv"
    signals_path = out_dir / "nyc_traffic_signals.geojson"

    if graphml_path.exists() and not force:
        print(f"  [skip] GraphML 已存在: {graphml_path.name}（使用 --force 强制重建）")
        return graphml_path

    print(f"  [PBF→GraphML] 读取: {pbf_path.name} ...")
    try:
        from pyrosm import OSM as PyrosmOSM
        bbox_list = [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]]
        osm_obj = PyrosmOSM(str(pbf_path), bounding_box=bbox_list)
        result = osm_obj.get_network(network_type="driving", nodes=True)
        if result is None:
            raise RuntimeError("pyrosm.get_network() 返回了 None")
        nodes_gdf, edges_gdf = result
        print(f"    pyrosm 提取: {len(nodes_gdf)} 节点, {len(edges_gdf)} 条边")

        G = osm_obj.to_graph(
            nodes_gdf, edges_gdf,
            graph_type="networkx",
            retain_all=False,
            osmnx_compatible=True,
        )
        if G is None:
            raise RuntimeError("pyrosm.to_graph() 返回了 None")
        print(f"    NetworkX 图: {len(G.nodes)} 节点, {len(G.edges)} 条边")

    except ImportError:
        print("    ⚠ pyrosm 未安装，尝试使用 osmnx.graph_from_xml() ...")
        # osmnx 1.x 可以直接读取 osm.pbf（部分版本支持）
        try:
            G = ox.graph_from_xml(str(pbf_path), retain_all=False)
            print(f"    osmnx.graph_from_xml 图: {len(G.nodes)} 节点, {len(G.edges)} 条边")
        except Exception as e2:
            print(f"    ✗ osmnx.graph_from_xml 也失败: {e2}")
            print("    请先安装 pyrosm: pip install pyrosm")
            return None

    # ── 保存 GraphML ──────────────────────────────────────────────────────────
    print(f"    保存 GraphML → {graphml_path.name} ...")
    ox.save_graphml(G, filepath=str(graphml_path))
    print(f"    ✓ {graphml_path.name}")

    # ── 导出节点 CSV ──────────────────────────────────────────────────────────
    import pandas as pd
    import numpy as np
    node_rows = []
    for nid, attrs in G.nodes(data=True):
        tags = attrs.get("tags", {}) if isinstance(attrs.get("tags"), dict) else {}
        hw = tags.get("highway", attrs.get("highway", ""))
        node_rows.append({
            "osmid":    nid,
            "lat":      attrs.get("y", np.nan),
            "lon":      attrs.get("x", np.nan),
            "highway":  str(hw) if hw else "",
        })
    pd.DataFrame(node_rows).to_csv(str(nodes_csv), index=False)
    print(f"    ✓ {nodes_csv.name} ({len(node_rows)} 行)")

    # ── 导出边 CSV ────────────────────────────────────────────────────────────
    edge_rows = []
    for u, v, key, attrs in G.edges(data=True, keys=True):
        tags = attrs.get("tags", {}) if isinstance(attrs.get("tags"), dict) else {}
        hw = attrs.get("highway", tags.get("highway", "residential"))
        lanes = attrs.get("lanes", tags.get("lanes", None))
        oneway = attrs.get("oneway", tags.get("oneway", False))
        length = attrs.get("length", attrs.get("geometry", None))
        # 取第一个值（可能是 list）
        if isinstance(hw, list):      hw = hw[0]
        if isinstance(lanes, list):   lanes = lanes[0]
        if isinstance(oneway, list):  oneway = oneway[0]
        if hasattr(length, "__len__") and not isinstance(length, str):
            length = None
        try:
            length_m = float(length) if length is not None else None
        except (TypeError, ValueError):
            length_m = None
        edge_rows.append({
            "u":        u,
            "v":        v,
            "key":      key,
            "highway":  str(hw),
            "lanes":    str(lanes) if lanes is not None else "",
            "oneway":   str(oneway),
            "length_m": length_m,
        })
    pd.DataFrame(edge_rows).to_csv(str(edges_csv), index=False)
    print(f"    ✓ {edges_csv.name} ({len(edge_rows)} 行)")

    # ── 提取信号灯 GeoJSON ────────────────────────────────────────────────────
    _extract_signals_geojson(G, signals_path)

    return graphml_path


def _extract_signals_geojson(G, out_path: Path) -> None:
    """从路网图中提取信号灯节点并保存为 GeoJSON。"""
    try:
        import osmnx as ox
        import geopandas as gpd

        G_proj = ox.project_graph(G, to_crs="EPSG:32618")
        nodes_gdf, _ = ox.graph_to_gdfs(G_proj)

        def _is_sig(v) -> bool:
            return "traffic_signals" in str(v)

        if "highway" in nodes_gdf.columns:
            sig_nodes = nodes_gdf[nodes_gdf["highway"].apply(_is_sig)].copy()
        else:
            # 尝试从 tags 属性中提取
            sig_ids = []
            for nid, attrs in G.nodes(data=True):
                tags = attrs.get("tags", {})
                hw = tags.get("highway", "") if isinstance(tags, dict) else str(tags)
                if "traffic_signals" in str(hw):
                    sig_ids.append(nid)
            sig_nodes = nodes_gdf.loc[nodes_gdf.index.isin(sig_ids)] if sig_ids else nodes_gdf.iloc[0:0]

        if len(sig_nodes) == 0:
            print("    ⚠ 未找到信号灯节点，跳过 GeoJSON 导出")
            return

        sig_nodes = sig_nodes[["geometry"]].to_crs("EPSG:4326")
        sig_nodes.to_file(str(out_path), driver="GeoJSON")
        print(f"    ✓ {out_path.name} ({len(sig_nodes)} 个信号灯)")
    except Exception as e:
        print(f"    ⚠ 信号灯提取失败: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def process_year(year: int, force: bool = False) -> None:
    year_dir = OSM_ROOT / str(year)
    if not year_dir.exists():
        print(f"[{year}] 目录不存在，跳过: {year_dir}")
        return

    # 查找 PBF 文件
    pbf_files = sorted(year_dir.glob("*.osm.pbf")) + sorted(year_dir.glob("*.pbf"))
    graphml_path = year_dir / "nyc_drive_graph.graphml"

    if graphml_path.exists() and not force:
        print(f"[{year}] GraphML 已存在，跳过 PBF 转换（使用 --force 重建）")
        # 即使 graphml 已存在，如果 CSV 不存在，也重新导出 CSV
        nodes_csv = year_dir / "nyc_nodes.csv"
        edges_csv = year_dir / "nyc_edges.csv"
        if not nodes_csv.exists() or not edges_csv.exists():
            print(f"[{year}] 从已有 GraphML 补充导出 CSV ...")
            _export_csv_from_graphml(graphml_path, year_dir)
        # 写元信息
        _write_meta(year_dir, graphml_path)
        return

    if not pbf_files:
        print(f"[{year}] 未找到 PBF 文件（{year_dir}），跳过")
        return

    pbf = pbf_files[0]
    print(f"\n{'='*60}")
    print(f"[{year}] 处理 PBF 文件: {pbf.name}")
    print(f"{'='*60}")

    result = convert_pbf_to_graphml(pbf, year_dir, force=force)
    if result:
        _write_meta(year_dir, result)
        print(f"[{year}] ✓ 转换完成")
    else:
        print(f"[{year}] ✗ 转换失败")


def _export_csv_from_graphml(graphml_path: Path, out_dir: Path) -> None:
    """从已有 GraphML 补充导出节点/边 CSV。"""
    import osmnx as ox
    import pandas as pd
    import numpy as np

    def _compat_bool(v) -> bool:
        if isinstance(v, bool): return v
        return str(v).lower() in ("yes", "true", "1", "on")

    G = ox.load_graphml(
        str(graphml_path),
        edge_dtypes={"oneway": _compat_bool, "reversed": _compat_bool},
        graph_dtypes={"consolidated": _compat_bool, "simplified": _compat_bool},
    )

    node_rows = []
    for nid, attrs in G.nodes(data=True):
        hw = attrs.get("highway", "")
        node_rows.append({
            "osmid": nid,
            "lat": attrs.get("y", np.nan),
            "lon": attrs.get("x", np.nan),
            "highway": str(hw) if hw else "",
        })
    pd.DataFrame(node_rows).to_csv(str(out_dir / "nyc_nodes.csv"), index=False)

    edge_rows = []
    for u, v, key, attrs in G.edges(data=True, keys=True):
        hw = attrs.get("highway", "residential")
        lanes = attrs.get("lanes", None)
        oneway = attrs.get("oneway", False)
        if isinstance(hw, list):     hw = hw[0]
        if isinstance(lanes, list):  lanes = lanes[0]
        if isinstance(oneway, list): oneway = oneway[0]
        edge_rows.append({
            "u": u, "v": v, "key": key,
            "highway": str(hw),
            "lanes": str(lanes) if lanes is not None else "",
            "oneway": str(oneway),
        })
    pd.DataFrame(edge_rows).to_csv(str(out_dir / "nyc_edges.csv"), index=False)
    print(f"    ✓ CSV 导出完成: {len(node_rows)} 节点, {len(edge_rows)} 边")


def _write_meta(year_dir: Path, graphml_path: Path) -> None:
    """写入元信息 JSON。"""
    meta = {
        "graphml":  str(graphml_path),
        "signals":  str(year_dir / "nyc_traffic_signals.geojson")
                    if (year_dir / "nyc_traffic_signals.geojson").exists() else None,
        "nodes_csv": str(year_dir / "nyc_nodes.csv")
                     if (year_dir / "nyc_nodes.csv").exists() else None,
        "edges_csv": str(year_dir / "nyc_edges.csv")
                     if (year_dir / "nyc_edges.csv").exists() else None,
        "bbox":     BBOX,
        "crs_proj": "EPSG:32618",
    }
    (year_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    ✓ meta.json 写入")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将各年份 OSM PBF 文件转换为 GraphML + CSV"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2024, 2025],
        help="要处理的年份列表（默认: 2024 2025）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新生成，即使 GraphML 已存在",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OSM PBF → GraphML + CSV 转换工具")
    print(f"  年份: {args.years}")
    print(f"  force: {args.force}")
    print("=" * 60)

    for year in args.years:
        process_year(year, force=args.force)

    print("\n所有年份处理完毕。")
    print("输出目录:", OSM_ROOT)


if __name__ == "__main__":
    main()
