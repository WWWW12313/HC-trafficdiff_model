"""
一次性：从本地 OSM 文件构建并缓存 NYC 路网 + 交通信号灯节点
（离线模式：你已手动下载 .osm.pbf 文件时使用）

使用方式：
    1. 从 Geofabrik 下载 NYC 区域文件（见 README / 实验日志），
       放到 raw_data/osm/new-york-city.osm.pbf（或 .osm.xml）
    2. 运行本脚本构建本地 GraphML 缓存：
       C:/Users/Admin/anaconda3/envs/crashgen/python.exe pipeline/download_osm_cache.py

    若本地没有 .osm 文件，可改用 --online 参数（需要联网）：
       C:/Users/Admin/anaconda3/envs/crashgen/python.exe pipeline/download_osm_cache.py --online
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OSM_DIR = ROOT / "raw_data" / "osm"
OSM_DIR.mkdir(parents=True, exist_ok=True)

GRAPHML_PATH = OSM_DIR / "nyc_drive_graph.graphml"
SIGNALS_PATH = OSM_DIR / "nyc_traffic_signals.geojson"

# NYC 边界（含5个行政区，适度放宽）
BBOX = dict(north=40.92, south=40.48, east=-73.68, west=-74.28)


def _extract_signals(G_proj) -> None:
    import osmnx as ox
    import geopandas as gpd

    nodes, _ = ox.graph_to_gdfs(G_proj)
    if "highway" not in nodes.columns:
        print("    → 路网无 highway 节点属性，跳过信号灯提取")
        return

    def _is_sig(v) -> bool:
        return "traffic_signals" in v if isinstance(v, list) else str(v) == "traffic_signals"

    sig_nodes = nodes[nodes["highway"].apply(_is_sig)].copy()
    if len(sig_nodes) == 0:
        print("    → 未找到信号灯节点")
        return

    sig_nodes = sig_nodes[["geometry"]].to_crs("EPSG:4326")
    sig_nodes.to_file(str(SIGNALS_PATH), driver="GeoJSON")
    print(f"    → {len(sig_nodes)} 个信号灯 → {SIGNALS_PATH}")


def build_from_pbf(pbf_path: Path) -> None:
    """从本地 .osm.pbf 文件构建路网（使用 pyrosm，osmnx 2.x 不支持直接读取 PBF）。"""
    import osmnx as ox
    from pyrosm import OSM

    print(f"[1/3] 用 pyrosm 从 PBF 提取 NYC 路网: {pbf_path.name} ...")
    bbox_list = [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]]
    osm = OSM(str(pbf_path), bounding_box=bbox_list)
    _net = osm.get_network(network_type="driving", nodes=True)
    if _net is None:
        raise RuntimeError("pyrosm.get_network() 返回了 None，请检查 PBF 文件和 bbox 是否正确")
    nodes, edges = _net
    assert nodes is not None and edges is not None, "pyrosm.get_network() 中 nodes/edges 为 None"
    print(f"    → pyrosm 提取: {len(nodes)} 节点, {len(edges)} 条边")

    G = osm.to_graph(nodes, edges, graph_type="networkx", retain_all=False, osmnx_compatible=True)
    if G is None:
        raise RuntimeError("pyrosm.to_graph() 返回了 None")
    print(f"    → osmnx 图: {len(G.nodes)} 节点, {len(G.edges)} 条边")

    print("[2/3] 保存 GraphML ...")
    ox.save_graphml(G, filepath=str(GRAPHML_PATH))
    print(f"    → {GRAPHML_PATH}")

    print("[3/3] 提取信号灯节点 ...")
    G_proj = ox.project_graph(G, to_crs="EPSG:32618")
    _extract_signals(G_proj)


def build_from_online() -> None:
    """联网下载并缓存（备用）。"""
    import osmnx as ox

    print("[1/3] 联网下载 NYC 驾车路网（约5分钟）...")
    G = ox.graph_from_bbox(
        bbox=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
        network_type="drive",
    )
    print(f"    → {len(G.nodes)} 节点，{len(G.edges)} 条边")

    print("[2/3] 保存 GraphML ...")
    ox.save_graphml(G, filepath=str(GRAPHML_PATH))
    print(f"    → {GRAPHML_PATH}")

    print("[3/3] 提取信号灯节点 ...")
    G_proj = ox.project_graph(G, to_crs="EPSG:32618")
    _extract_signals(G_proj)


def main() -> None:
    parser = argparse.ArgumentParser()
    _pbf_candidates = sorted(OSM_DIR.glob("*.pbf"), reverse=True)
    _default_pbf = str(_pbf_candidates[0]) if _pbf_candidates else str(OSM_DIR / "new-york-city.osm.pbf")
    parser.add_argument(
        "--pbf",
        default=_default_pbf,
        help="本地 OSM PBF 文件路径（Geofabrik 下载）",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="改用联网下载（无需 PBF 文件）",
    )
    args = parser.parse_args()

    if GRAPHML_PATH.exists():
        print(f"路网已存在，跳过: {GRAPHML_PATH}")
        print("如需重新生成，删除该文件后重新运行。")
    elif args.online:
        build_from_online()
    else:
        pbf = Path(args.pbf)
        if not pbf.exists():
            print(f"⚠ 未找到 PBF 文件: {pbf}")
            print("请先下载（见下方 URL），或改用 --online 参数。")
            return
        build_from_pbf(pbf)

    # 写入元信息
    meta = {
        "graphml": str(GRAPHML_PATH),
        "signals": str(SIGNALS_PATH) if SIGNALS_PATH.exists() else None,
        "bbox": BBOX,
        "crs_proj": "EPSG:32618",
    }
    (OSM_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("完成，元信息写入 raw_data/osm/meta.json")


if __name__ == "__main__":
    main()
