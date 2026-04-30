import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from meteostat import Hourly, Stations
from datetime import datetime
import warnings
import os
import requests
import time
from sklearn.neighbors import BallTree
import urllib3
import networkx as nx
import re  # <--- 关键修复：确保导入 re 模块

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 1. 代理配置 (端口 7897)
# ==============================================================================
PROXY_PORT = "7897"
PROXIES = {
    "http": f"http://127.0.0.1:{PROXY_PORT}",
    "https": f"http://127.0.0.1:{PROXY_PORT}"
}
print(f"🌍 代理已锁定: {PROXY_PORT}")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.openstreetmap.org/'
}

# 全球镜像源列表
OVERPASS_SERVERS = [
    "https://api.openstreetmap.fr/oapi/interpreter",          # 法国 (优先)
    "https://overpass.kumi.systems/api/interpreter",          # Kumi
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",# 俄罗斯
    "https://overpass-api.de/api/interpreter"                 # 德国
]

# NYC 标准配置
NYC_CRS = "EPSG:32618"
NYC_BOUNDS = {
    'min_lat': 40.45, 'max_lat': 41.15,
    'min_lon': -74.30, 'max_lon': -73.65
}

# ==============================================================================
# 🛠️ 智能下载引擎
# ==============================================================================
def manual_download_chunk_smart(north, south, east, west, chunk_id):
    query = f"""
    [out:xml][timeout:180];
    (
      way["highway"]( {south},{west},{north},{east} );
    );
    (._;>;);
    out;
    """
    
    filename = f"temp_chunk_{chunk_id}.osm"
    
    for server_url in OVERPASS_SERVERS:
        for attempt in range(2):
            try:
                response = requests.post(
                    server_url, 
                    data=query, 
                    proxies=PROXIES, 
                    headers=HEADERS,
                    timeout=180, 
                    verify=False
                )
                
                if response.status_code == 200:
                    content = response.content
                    # 校验 HTML 报错
                    if b"<html" in content[:100].lower() or b"<!doctype html" in content[:100].lower():
                        continue 
                    # 校验 XML
                    if b"<osm" not in content[:100]:
                        continue
                    
                    if len(content) < 500: return None
                    
                    with open(filename, 'wb') as f:
                        f.write(content)
                    return filename
                    
                elif response.status_code == 429:
                    time.sleep(5)
                    
            except:
                pass
    return None

def download_nyc_graph_multi_source(north, south, east, west):
    print(f"  ⚡ 启动智能多源下载 (5x5 网格)")
    
    lat_steps = np.linspace(south, north, 6)
    lon_steps = np.linspace(west, east, 6)
    
    graphs = []
    total = 25
    count = 0
    
    for i in range(len(lat_steps)-1):
        for j in range(len(lon_steps)-1):
            count += 1
            s_c, n_c = lat_steps[i], lat_steps[i+1]
            w_c, e_c = lon_steps[j], lon_steps[j+1]
            margin = 0.001
            
            print(f"    -> [{count}/{total}] 下载...", end="")
            
            osm_file = manual_download_chunk_smart(n_c+margin, s_c-margin, e_c+margin, w_c-margin, count)
            
            if osm_file:
                try:
                    G_chunk = ox.graph_from_xml(osm_file)
                    if len(G_chunk.edges) > 0:
                        graphs.append(G_chunk)
                        print(f" ✅ 成功 ({len(G_chunk.edges)} 边)")
                    else:
                        print(f" ⚠️ 无道路")
                    
                    if os.path.exists(osm_file): os.remove(osm_file)
                except:
                    print(f" ❌ XML解析失败")
            else:
                print(f" ⚠️ 无数据 (跳过)")

    if not graphs:
        raise ValueError("\n❌ 所有源均无法连接！")
        
    print(f"  🧩 合并 {len(graphs)} 个分块...")
    
    # 使用 networkx 的 compose_all 避免 osmnx 版本报错
    if len(graphs) == 1:
        G_final = graphs[0]
    else:
        G_final = nx.compose_all(graphs)
    
    return G_final

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def enrich_nyc_final_v8(input_file, output_file):
    print("="*60)
    print("🚦 NYC 数据补全 (v8: 修复 re 模块缺失)")
    print("="*60)
    
    # 0. 代理测试
    try:
        print("  🔍 测试代理...", end="")
        requests.get("https://www.google.com", proxies=PROXIES, timeout=5, verify=False)
        print(" ✅ 通畅")
    except:
        print("\n❌ 代理不通！")
        return

    # 1. 读取
    try:
        df = pd.read_csv(input_file)
    except:
        print("❌ 文件不存在")
        return

    df['CRASH_FULL_TIME'] = pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'], errors='coerce')
    df['merge_hour'] = df['CRASH_FULL_TIME'].dt.floor('H')
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

    # 坐标清洗：保留原始 NYC 范围内的数据，剔除异常值
    df_valid = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'CRASH_FULL_TIME']).copy()
    mask_nyc = (
        (df_valid['LATITUDE'] >= NYC_BOUNDS['min_lat']) & 
        (df_valid['LATITUDE'] <= NYC_BOUNDS['max_lat']) & 
        (df_valid['LONGITUDE'] >= NYC_BOUNDS['min_lon']) & 
        (df_valid['LONGITUDE'] <= NYC_BOUNDS['max_lon'])
    )
    df_clean = df_valid[mask_nyc].copy()
    
    dropped_count = len(df_valid) - len(df_clean)
    print(f"\n[Step 1] 数据清洗: 保留 {len(df_clean)} 行 (剔除 {dropped_count} 条异常坐标)")

    if len(df_clean) == 0: return

    # 2. 天气
    print("\n[Step 2] 获取天气...", end="")
    try:
        avg_lat, avg_lon = df_clean['LATITUDE'].mean(), df_clean['LONGITUDE'].mean()
        stations = Stations().nearby(avg_lat, avg_lon)
        station = stations.fetch(1)
        if not station.empty:
            start, end = df_clean['CRASH_FULL_TIME'].min(), df_clean['CRASH_FULL_TIME'].max()
            weather = Hourly(station.index[0], start, end).fetch().reset_index()
            for c in ['time', 'temp', 'prcp', 'wspd', 'visib', 'coco']:
                if c not in weather.columns: weather[c] = np.nan
            
            df_merged = pd.merge(df_clean, weather[['time', 'temp', 'prcp', 'wspd', 'visib', 'coco']], 
                                 left_on='merge_hour', right_on='time', how='left')
            
            coco_map = {1:'Clear', 2:'Fair', 3:'Cloudy', 4:'Overcast', 5:'Fog', 6:'Freezing Fog', 
                        7:'Light Rain', 8:'Rain', 9:'Heavy Rain', 10:'Freezing Rain', 14:'Light Snow', 15:'Snow'}
            df_merged['WEATHER_CONDITION'] = df_merged['coco'].map(coco_map)
            
            def refine_weather(row):
                if pd.notna(row['WEATHER_CONDITION']): return row['WEATHER_CONDITION']
                if pd.notna(row['visib']) and row['visib'] < 2.0: return 'Fog/Haze (Inferred)'
                if pd.notna(row['prcp']) and row['prcp'] > 0:
                    return 'Snow (Inferred)' if row['temp'] <= 0 else 'Rain (Inferred)'
                return 'Clear/Cloudy'
            
            df_merged['REAL_WEATHER'] = df_merged.apply(refine_weather, axis=1)
            df_clean = df_merged.drop(columns=['time', 'merge_hour'])
            df_clean.rename(columns={'temp':'TEMP_C', 'visib':'VISIBILITY_KM', 'wspd':'WIND_SPEED_KMH'}, inplace=True)
            print(" ✅ 完成")
    except Exception as e:
        print(f" ⚠️ 出错: {e}")

    # 3. 路网
    print("\n[Step 3] 下载路网 (自动换源)...")
    north = df_clean['LATITUDE'].max() + 0.005
    south = df_clean['LATITUDE'].min() - 0.005
    east = df_clean['LONGITUDE'].max() + 0.005
    west = df_clean['LONGITUDE'].min() - 0.005
    
    try:
        G = download_nyc_graph_multi_source(north, south, east, west)
        
        print(f"  - 投影路网...")
        G_proj = ox.project_graph(G, to_crs=NYC_CRS)
        
        geometry = [Point(xy) for xy in zip(df_clean['LONGITUDE'], df_clean['LATITUDE'])]
        gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
        gdf_proj = gdf.to_crs(NYC_CRS)
        
        # A. 红绿灯
        print("  - 分析红绿灯...")
        nodes = ox.graph_to_gdfs(G_proj, nodes=True, edges=False)
        if 'highway' in nodes.columns:
            signal_nodes = nodes[nodes['highway'] == 'traffic_signals']
            if not signal_nodes.empty:
                tree = BallTree(signal_nodes.geometry.apply(lambda p: [p.x, p.y]).tolist(), leaf_size=15)
                acc_points = gdf_proj.geometry.apply(lambda p: [p.x, p.y]).tolist()
                dist, _ = tree.query(acc_points, k=1)
                df_clean['DIST_TO_SIGNAL_M'] = dist
                df_clean['HAS_TRAFFIC_SIGNAL'] = (df_clean['DIST_TO_SIGNAL_M'] < 30).astype(int)
            else:
                df_clean['HAS_TRAFFIC_SIGNAL'] = 0
        else:
            df_clean['HAS_TRAFFIC_SIGNAL'] = 0

        # B. 匹配
        print("  - 匹配最近道路...")
        ne_edges = ox.distance.nearest_edges(G_proj, X=gdf_proj.geometry.x, Y=gdf_proj.geometry.y)
        
        osm_res = {'maxspeed':[], 'lanes':[], 'highway':[], 'oneway':[]}
        for u, v, key in ne_edges:
            edge = G_proj.get_edge_data(u, v, key)
            def get_v(k, default=np.nan):
                val = edge.get(k, default)
                if isinstance(val, list): return val[0]
                return val
            osm_res['maxspeed'].append(get_v('maxspeed'))
            osm_res['lanes'].append(get_v('lanes'))
            osm_res['highway'].append(get_v('highway', 'residential'))
            osm_res['oneway'].append(get_v('oneway', False))
            
        df_clean['OSM_TYPE'] = osm_res['highway']
        df_clean['OSM_SPEED_TAG'] = osm_res['maxspeed']
        df_clean['OSM_LANES_TAG'] = osm_res['lanes']
        df_clean['OSM_ONEWAY'] = osm_res['oneway']

        # C. 推断
        print("  - 法规推断...")
        
        # 【修复点】: 确保 re 模块已导入
        def infer_speed_2017(row):
            raw = str(row['OSM_SPEED_TAG'])
            # 使用 re.findall 安全提取数字
            digits = re.findall(r'\d+', raw)
            if digits: return float(digits[0])
            
            h_type = str(row['OSM_TYPE']).lower()
            if 'motorway' in h_type: return 50
            if 'trunk' in h_type: return 35
            if 'primary' in h_type: return 25 
            if 'living_street' in h_type: return 15
            return 25 
        
        df_clean['REAL_SPEED_LIMIT'] = df_clean.apply(infer_speed_2017, axis=1)

        major_roads = ['motorway', 'trunk', 'primary']
        def infer_divider(row):
            h_type = str(row['OSM_TYPE']).lower()
            if 'motorway' in h_type: return 1
            if any(r in h_type for r in major_roads) and row['OSM_ONEWAY']: return 1
            return 0
        df_clean['HAS_DIVIDER'] = df_clean.apply(infer_divider, axis=1)
        
        def infer_lanes(row):
            raw = str(row['OSM_LANES_TAG'])
            if raw.isdigit(): return int(raw)
            h_type = str(row['OSM_TYPE']).lower()
            if 'motorway' in h_type: return 3
            if 'trunk' in h_type: return 3
            if 'primary' in h_type: return 2
            return 1
        df_clean['INFERRED_LANES'] = df_clean.apply(infer_lanes, axis=1)

    except Exception as e:
        print(f"❌ 严重错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 保存
    print("\n[Step 4] 保存结果...")
    v_cols = [c for c in df_clean.columns if 'CONTRIBUTING FACTOR VEHICLE' in c]
    df_clean['TOTAL_VEHICLES'] = df_clean[v_cols].notna().sum(axis=1)
    df_clean['IS_MULTI_VEHICLE'] = (df_clean['TOTAL_VEHICLES'] >= 3).astype(int)
    
    df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 处理完成！文件已保存: {output_file}")
    
    if 'OSM_TYPE' in df_clean.columns:
        print("\n--- 最终检查: 道路类型分布 ---")
        print(df_clean['OSM_TYPE'].value_counts().head())

if __name__ == "__main__":
    input_csv = 'nyc_accidents_2017.csv'
    output_csv = 'nyc_2017_final_v8.csv'
    
    if os.path.exists(input_csv):
        enrich_nyc_final_v8(input_csv, output_csv)
    else:
        print("找不到文件")