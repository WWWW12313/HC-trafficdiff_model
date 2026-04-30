"""Quick debug: test nearest_edges with geographic graph (EPSG:4326) + lon/lat."""
import osmnx as ox
import numpy as np

def _compat(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("yes", "true", "1", "on")

print("Loading graphml...")
G = ox.load_graphml(
    "raw_data/osm/nyc_drive_graph.graphml",
    edge_dtypes={"oneway": _compat, "reversed": _compat},
    graph_dtypes={"consolidated": _compat, "simplified": _compat},
)
print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

# 5 diverse NYC locations: Midtown, Lower East Side, Brooklyn, Staten Island, Queens
test_lons = [-73.9857, -73.9712, -73.9441, -74.0060, -73.8542]
test_lats  = [ 40.7580,  40.7144,  40.6592,  40.6501,  40.7282]

ne = ox.nearest_edges(G, X=np.array(test_lons), Y=np.array(test_lats))
print("nearest_edges result (should differ for each point):")
for i, (u, v, k) in enumerate(ne):
    ed = G.get_edge_data(u, v, k) or {}
    hw = ed.get("highway", "MISSING")
    lanes = ed.get("lanes", None)
    ow = ed.get("oneway", None)
    print(f"  pt{i} ({test_lons[i]:.4f},{test_lats[i]:.4f}) -> edge({u},{v}) highway={hw} lanes={lanes} oneway={ow}")
