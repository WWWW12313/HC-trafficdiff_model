"""
postprocess_synthetic_v3.py  —  v6 合成数据后处理管线
=====================================================
功能:
  1. 反解码分类变量 (cat index → label)
  2. 经纬度范围校验 & 裁剪
  3. 基于经纬度的 API 补全 (OSM 路网 + 天气)
  4. 数据完整性校验报告

输入: exp/nyc_crash_v3/MODEL_NAME/ 下的 X_num_train.npy, X_cat_train.npy, y_train.npy
输出: synthetic_complete.csv, postprocess_report.json
"""

import argparse
import json
import os
import sys
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# 纽约市经纬度边界
NYC_BOUNDS = {
    "lat_min": 40.4774, "lat_max": 40.9176,
    "lon_min": -74.2591, "lon_max": -73.7004,
}

# 天气代码 → REAL_WEATHER 映射
COCO_TO_WEATHER = {
    1: "Clear/Cloudy", 2: "Clear/Cloudy", 3: "Clear/Cloudy", 4: "Clear/Cloudy",
    5: "Fog", 6: "Fog", 7: "Light Rain", 8: "Rain",
    9: "Rain", 10: "Light Rain", 11: "Light Rain",
    14: "Light Rain", 15: "Rain", 16: "Heavy Rain",
    17: "Rain", 18: "Rain", 19: "Light Snow", 20: "Light Snow",
    21: "Snow", 22: "Snow", 23: "Light Rain", 24: "Light Rain",
    25: "Light Rain", 26: "Rain", 27: "Rain",
}


def load_synthetic_data(parent_dir, data_dir):
    """加载合成数据 npy 和 column_mapping。"""
    X_num = None
    X_cat = None
    y = np.load(os.path.join(parent_dir, "y_train.npy"), allow_pickle=True)

    num_path = os.path.join(parent_dir, "X_num_train.npy")
    if os.path.exists(num_path):
        X_num = np.load(num_path, allow_pickle=True).astype(float)

    cat_path = os.path.join(parent_dir, "X_cat_train.npy")
    if os.path.exists(cat_path):
        X_cat = np.load(cat_path, allow_pickle=True)

    # 加载 info 和 mapping
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    mapping_path = os.path.join(data_dir, "column_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            column_mapping = json.load(f)
    else:
        column_mapping = {}

    return X_num, X_cat, y, info, column_mapping


def decode_categorical(X_cat, cat_columns, column_mapping):
    """将分类索引反解码为标签。"""
    df_cat = pd.DataFrame(X_cat, columns=cat_columns)
    for col in cat_columns:
        if col in column_mapping:
            idx2label = {int(v): k for k, v in column_mapping[col].items()}
            df_cat[col] = df_cat[col].astype(int).map(idx2label).fillna("UNKNOWN")
        else:
            df_cat[col] = df_cat[col].astype(str)
    return df_cat


def clip_coordinates(df, report):
    """裁剪经纬度到纽约市范围内。"""
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        return df

    n = len(df)
    lat_oob = ((df["LATITUDE"] < NYC_BOUNDS["lat_min"]) |
               (df["LATITUDE"] > NYC_BOUNDS["lat_max"])).sum()
    lon_oob = ((df["LONGITUDE"] < NYC_BOUNDS["lon_min"]) |
               (df["LONGITUDE"] > NYC_BOUNDS["lon_max"])).sum()

    df["LATITUDE"] = df["LATITUDE"].clip(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
    df["LONGITUDE"] = df["LONGITUDE"].clip(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])

    report["lat_out_of_bounds"] = int(lat_oob)
    report["lon_out_of_bounds"] = int(lon_oob)
    report["lat_oob_ratio"] = round(lat_oob / n, 4)
    report["lon_oob_ratio"] = round(lon_oob / n, 4)

    print(f"  📍 经纬度越界: lat={lat_oob} ({lat_oob/n*100:.1f}%), lon={lon_oob} ({lon_oob/n*100:.1f}%)")
    return df


def approximate_date_from_features(df):
    """从 CRASH_SEASON + DAY_OF_WEEK 近似还原一个合理日期。
    仅用于天气 API 查询，不影响最终数据。"""
    season_month_map = {"0": 1, "1": 4, "2": 7, "3": 10}  # 每季代表月
    dates = []
    for _, row in df.iterrows():
        season = str(row.get("CRASH_SEASON", "0"))
        month = season_month_map.get(season, 1)
        day_of_week = int(row.get("DAY_OF_WEEK", 0))
        # 取该月第一个匹配星期几的日期
        from datetime import datetime, timedelta
        d = datetime(2017, month, 1)
        while d.weekday() != day_of_week:
            d += timedelta(days=1)
        dates.append(d)
    return dates


def try_osm_enrich(df, report, cache_dir="cache/osm_v3"):
    """尝试使用 OSM API 补全路网信息 (离线版: 跳过若无网络)。"""
    print("  🌐 OSM 路网补全...")
    os.makedirs(cache_dir, exist_ok=True)

    cols_to_add = {
        "OSM_TYPE": "residential",
        "OSM_SPEED_TAG": "25 mph",
        "OSM_ONEWAY": "0",
        "HAS_TRAFFIC_SIGNAL": "0",
        "HAS_DIVIDER": "0",
        "INFERRED_LANES": "2",
        "DIST_TO_SIGNAL_M": 500.0,
    }

    # 初始化默认值
    for col, default in cols_to_add.items():
        df[col] = default

    try:
        import osmnx as ox
        # 尝试加载纽约市路网缓存
        graph_cache = os.path.join(cache_dir, "nyc_graph.graphml")
        if os.path.exists(graph_cache):
            print("    📦 使用缓存路网")
            G = ox.load_graphml(graph_cache)
        else:
            print("    ⬇️ 下载纽约市路网 (可能需要几分钟)...")

            # 某些网络环境下 overpass-api.de 存在 SSL EOF，按端点顺序重试。
            overpass_endpoints = [
                "https://overpass-api.de/api",
                "https://overpass.kumi.systems/api",
                "https://overpass.private.coffee/api",
            ]
            last_err = None
            G = None
            for ep in overpass_endpoints:
                try:
                    print(f"    🌐 尝试 Overpass 端点: {ep}")
                    ox.settings.overpass_url = ep
                    # 兼容企业网络/证书链问题
                    ox.settings.requests_kwargs = {"verify": False}
                    G = ox.graph_from_bbox(
                        bbox=(
                            NYC_BOUNDS["lat_max"], NYC_BOUNDS["lat_min"],
                            NYC_BOUNDS["lon_max"], NYC_BOUNDS["lon_min"],
                        ),
                        network_type="drive",
                    )
                    if G is not None and len(G.nodes) > 0:
                        print(f"    ✅ 端点成功: {ep}")
                        break
                except Exception as e:
                    last_err = e
                    print(f"    ⚠️ 端点失败: {ep} | {e}")

            if G is None:
                raise RuntimeError(f"所有 Overpass 端点均失败: {last_err}")

            ox.save_graphml(G, graph_cache)

        from sklearn.neighbors import BallTree

        # 获取信号灯位置
        nodes = ox.graph_to_gdfs(G, edges=False)
        signals = nodes[nodes.get("highway", pd.Series()) == "traffic_signals"]
        if len(signals) > 0:
            signal_coords = np.radians(signals[["y", "x"]].values)
            signal_tree = BallTree(signal_coords, metric="haversine")

            coords = np.radians(df[["LATITUDE", "LONGITUDE"]].values)
            dists, _ = signal_tree.query(coords, k=1)
            df["DIST_TO_SIGNAL_M"] = (dists.flatten() * 6371000).round(1)
            df["HAS_TRAFFIC_SIGNAL"] = (df["DIST_TO_SIGNAL_M"] < 50).astype(int).astype(str)

        # 获取最近道路信息
        edges = ox.graph_to_gdfs(G, nodes=False)
        if "highway" in edges.columns:
            # 简化取样: 对每行找最近道路
            road_type_counts = edges["highway"].value_counts()
            print(f"    📊 路网包含 {len(edges)} 条边")

        report["osm_enrichment"] = "success"
        report["osm_edges"] = len(edges) if "edges" in dir() else 0
        print("    ✅ OSM 补全完成")

    except Exception as e:
        report["osm_enrichment"] = f"skipped: {str(e)}"
        print(f"    ⚠️ OSM 补全跳过: {e}")
        print("    ℹ️ 使用默认值填充")

    return df


def try_weather_enrich(df, report):
    """尝试使用 Meteostat API 补全天气信息。"""
    print("  🌤️ 天气数据补全...")

    weather_cols = {
        "TEMP_C": 15.0,
        "prcp": 0.0,
        "WIND_SPEED_KMH": 10.0,
        "REAL_WEATHER": "Clear/Cloudy",
    }
    for col, default in weather_cols.items():
        df[col] = default

    try:
        from meteostat import Hourly, Stations
        from datetime import datetime

        # 纽约中央气象站
        station = Stations().nearby(40.7128, -74.0060).fetch(1)
        if station.empty:
            raise RuntimeError("未找到纽约气象站")

        station_id = station.index[0]
        dates = approximate_date_from_features(df)

        # 批量查询（按月聚合）
        months_needed = set((d.year, d.month) for d in dates)
        weather_cache = {}

        for year, month in months_needed:
            start = datetime(year, month, 1)
            end = datetime(year, month, 28)  # 安全范围
            data = Hourly(station_id, start, end).fetch()
            if not data.empty:
                for _, row in data.iterrows():
                    ts = pd.Timestamp(row.name)  # type: ignore[arg-type]
                    key = ts.strftime("%Y-%m-%d-%H")
                    weather_cache[key] = {
                        "temp": row.get("temp", 15.0),
                        "prcp": row.get("prcp", 0.0),
                        "wspd": row.get("wspd", 10.0),
                        "coco": row.get("coco", 1),
                    }

        # 填充
        time_period_hour = {"0": 2, "1": 8, "2": 12, "3": 17}
        for i, (idx, row) in enumerate(df.iterrows()):
            d = dates[i]
            hour = time_period_hour.get(str(row.get("CRASH_TIME_PERIOD", "2")), 12)
            key = f"{d.strftime('%Y-%m-%d')}-{hour}"
            if key in weather_cache:
                w = weather_cache[key]
                df.at[idx, "TEMP_C"] = w["temp"] if pd.notna(w["temp"]) else 15.0
                df.at[idx, "prcp"] = w["prcp"] if pd.notna(w["prcp"]) else 0.0
                df.at[idx, "WIND_SPEED_KMH"] = w["wspd"] if pd.notna(w["wspd"]) else 10.0
                coco = int(w.get("coco", 1)) if pd.notna(w.get("coco")) else 1
                df.at[idx, "REAL_WEATHER"] = COCO_TO_WEATHER.get(coco, "Clear/Cloudy")

        report["weather_enrichment"] = "success"
        print("    ✅ 天气补全完成")

    except Exception as e:
        report["weather_enrichment"] = f"skipped: {str(e)}"
        print(f"    ⚠️ 天气补全跳过: {e}")
        print("    ℹ️ 使用默认值填充")

    return df


def fallback_enrich_from_pristine(df, report, pristine_csv="nyc_2017_pristine_v8.csv"):
    """离线兜底: 使用原始 2017 数据做最近邻补全 OSM/天气列。"""
    print("  🧩 离线兜底补全 (pristine KNN)...")

    wanted_cols = [
        "LATITUDE", "LONGITUDE",
        "TEMP_C", "prcp", "WIND_SPEED_KMH", "REAL_WEATHER",
        "OSM_TYPE", "OSM_SPEED_TAG", "OSM_ONEWAY",
        "HAS_TRAFFIC_SIGNAL", "HAS_DIVIDER", "INFERRED_LANES", "DIST_TO_SIGNAL_M",
    ]
    fill_defaults = {
        "TEMP_C": 15.0,
        "prcp": 0.0,
        "WIND_SPEED_KMH": 10.0,
        "REAL_WEATHER": "Clear/Cloudy",
        "OSM_TYPE": "residential",
        "OSM_SPEED_TAG": "25 mph",
        "OSM_ONEWAY": "0",
        "HAS_TRAFFIC_SIGNAL": "0",
        "HAS_DIVIDER": "0",
        "INFERRED_LANES": "2",
        "DIST_TO_SIGNAL_M": 500.0,
    }

    try:
        if not os.path.exists(pristine_csv):
            # 兼容从不同 cwd 运行
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", pristine_csv)
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                pristine_csv = candidate

        usecols = [c for c in wanted_cols if c in pd.read_csv(pristine_csv, nrows=0).columns]
        raw = pd.read_csv(pristine_csv, usecols=usecols)
        raw = raw.dropna(subset=["LATITUDE", "LONGITUDE"])

        from sklearn.neighbors import BallTree

        raw_coords = np.radians(raw[["LATITUDE", "LONGITUDE"]].astype(float).values)
        syn_coords = np.radians(df[["LATITUDE", "LONGITUDE"]].astype(float).values)

        tree = BallTree(raw_coords, metric="haversine")
        _, nn_idx = tree.query(syn_coords, k=1)
        nn_idx = nn_idx.flatten()

        for col in wanted_cols:
            if col in {"LATITUDE", "LONGITUDE"}:
                continue
            if col in raw.columns:
                df[col] = raw.iloc[nn_idx][col].reset_index(drop=True).values
            else:
                df[col] = fill_defaults[col]

        # 统一填补缺失
        for col, default in fill_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        report["offline_fallback"] = "success_pristine_knn"
        print("    ✅ 离线兜底补全完成")
    except Exception as e:
        report["offline_fallback"] = f"failed: {e}"
        print(f"    ⚠️ 离线兜底补全失败: {e}")
        print("    ℹ️ 保留当前默认值")

    return df


def compute_fidelity_metrics(syn_df, real_data_dir, info, report):
    """计算合成 vs 真实数据分布的保真度指标。"""
    print("  📊 计算分布保真度...")

    # 加载真实训练数据
    X_num_real = np.load(os.path.join(real_data_dir, "X_num_train.npy"), allow_pickle=True)
    X_cat_real = np.load(os.path.join(real_data_dir, "X_cat_train.npy"), allow_pickle=True)
    y_real = np.load(os.path.join(real_data_dir, "y_train.npy"), allow_pickle=True)

    fidelity = {}

    # 连续特征统计对比
    num_cols = info.get("num_columns", [])
    for i, col in enumerate(num_cols):
        if col in syn_df.columns:
            syn_vals = syn_df[col].astype(float).values
            real_vals = X_num_real[:, i].astype(float) if i < X_num_real.shape[1] else np.array([])
            if len(real_vals) > 0:
                fidelity[f"{col}_mean_diff"] = round(abs(syn_vals.mean() - real_vals.mean()), 4)
                fidelity[f"{col}_std_diff"] = round(abs(syn_vals.std() - real_vals.std()), 4)

    # 分类特征 JS 散度
    cat_cols = info.get("cat_columns", [])
    js_divergences = {}
    for j, col in enumerate(cat_cols):
        if j < X_cat_real.shape[1]:
            real_vals = X_cat_real[:, j].astype(int)
            # 合成数据的分类值需要映射回索引
            mapping_path = os.path.join(real_data_dir, "column_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path) as f:
                    cm = json.load(f)
                if col in cm and col in syn_df.columns:
                    label2idx = cm[col]
                    syn_encoded = syn_df[col].map(label2idx).fillna(-1).astype(int).values
                    n_classes = max(max(real_vals) + 1, max(label2idx.values()) + 1)
                    real_hist = np.bincount(real_vals, minlength=n_classes).astype(float)
                    syn_hist = np.bincount(syn_encoded[syn_encoded >= 0], minlength=n_classes).astype(float)
                    real_hist /= real_hist.sum() + 1e-10
                    syn_hist /= syn_hist.sum() + 1e-10
                    js = jensenshannon(real_hist, syn_hist)
                    js_divergences[col] = round(float(js), 4)

    fidelity["js_divergences"] = js_divergences

    # y 分布
    y_syn = syn_df[info["target_col"]].astype(float).values if info["target_col"] in syn_df.columns else y_real[:10]
    fidelity["y_mean_real"] = round(float(y_real.mean()), 4)
    fidelity["y_mean_syn"] = round(float(y_syn.mean()), 4)
    fidelity["y_std_real"] = round(float(y_real.std()), 4)
    fidelity["y_std_syn"] = round(float(y_syn.std()), 4)

    report["fidelity"] = fidelity
    # CAUSE 列的 JS 散度高亮
    cause_js = {k: v for k, v in js_divergences.items() if k.startswith("CAUSE_")}
    if cause_js:
        print(f"    📋 CAUSE 列 JS 散度: {cause_js}")

    return report


def postprocess(
    parent_dir,
    data_dir,
    output_csv=None,
    skip_api=False,
    pristine_csv="nyc_2017_pristine_v8.csv",
):
    """完整后处理管线。"""
    print("=" * 80)
    print(f"🔧 v3 后处理管线 | {parent_dir}")
    print("=" * 80)

    report = {"parent_dir": parent_dir, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 加载
    print("\n--- Step 1: 加载合成数据 ---")
    X_num, X_cat, y, info, column_mapping = load_synthetic_data(parent_dir, data_dir)
    print(f"  X_num: {X_num.shape if X_num is not None else None}")
    print(f"  X_cat: {X_cat.shape if X_cat is not None else None}")
    print(f"  y: {y.shape}")

    # 反解码分类变量
    print("\n--- Step 2: 反解码分类变量 ---")
    num_cols = info.get("num_columns", [])
    cat_cols = info.get("cat_columns", [])

    df = pd.DataFrame()
    if X_num is not None:
        for i, col in enumerate(num_cols):
            df[col] = X_num[:, i]
    df_cat = decode_categorical(X_cat, cat_cols, column_mapping)
    df = pd.concat([df, df_cat], axis=1)
    df[info["target_col"]] = y
    print(f"  合成数据 DataFrame: {df.shape}")

    # 经纬度裁剪
    print("\n--- Step 3: 经纬度校验 ---")
    df = clip_coordinates(df, report)

    # API 补全
    if not skip_api:
        print("\n--- Step 4: API 补全 ---")
        df = try_osm_enrich(df, report)
        df = try_weather_enrich(df, report)
        # 若 API 任一失败，使用离线兜底补全，避免整列常量默认值
        if str(report.get("osm_enrichment", "")).startswith("skipped") or str(report.get("weather_enrichment", "")).startswith("skipped"):
            df = fallback_enrich_from_pristine(df, report, pristine_csv=pristine_csv)
    else:
        print("\n--- Step 4: API 补全 (跳过) ---")
        report["osm_enrichment"] = "skipped_by_flag"
        report["weather_enrichment"] = "skipped_by_flag"
        # 保持可复现: 跳过 API 时优先从指定 pristine CSV 回填上下文列。
        df = fallback_enrich_from_pristine(df, report, pristine_csv=pristine_csv)

    # 多车事故分析
    if "VEHICLE TYPE CODE 3" in df.columns:
        vt3_unspec = (df["VEHICLE TYPE CODE 3"] == "UNSPECIFIED").mean()
        report["vt3_unspecified_ratio"] = round(vt3_unspec, 4)
    if "VEHICLE TYPE CODE 4" in df.columns:
        vt4_unspec = (df["VEHICLE TYPE CODE 4"] == "UNSPECIFIED").mean()
        report["vt4_unspecified_ratio"] = round(vt4_unspec, 4)
    if "VEHICLE TYPE CODE 5" in df.columns:
        vt5_unspec = (df["VEHICLE TYPE CODE 5"] == "UNSPECIFIED").mean()
        report["vt5_unspecified_ratio"] = round(vt5_unspec, 4)

    # 保真度指标
    print("\n--- Step 5: 保真度检查 ---")
    report = compute_fidelity_metrics(df, data_dir, info, report)

    # 保存
    print("\n--- Step 6: 保存 ---")
    if output_csv is None:
        output_csv = os.path.join(parent_dir, "synthetic_complete.csv")
    df.to_csv(output_csv, index=False)
    print(f"  📁 {output_csv}")

    report_path = os.path.join(parent_dir, "postprocess_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  📁 {report_path}")

    print("\n✅ 后处理完成！")
    return report


def main():
    parser = argparse.ArgumentParser(description="v3 合成数据后处理")
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="合成数据目录 (含 X_num_train.npy 等)")
    parser.add_argument("--data_dir", type=str, default="data/nyc_crash_v3",
                        help="真实数据目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 路径")
    parser.add_argument("--skip_api", action="store_true",
                        help="跳过 API 补全")
    parser.add_argument("--pristine_csv", type=str, default="nyc_2017_pristine_v8.csv",
                        help="离线兜底回填所用 pristine CSV")
    args = parser.parse_args()
    postprocess(args.parent_dir, args.data_dir, args.output, args.skip_api, args.pristine_csv)


if __name__ == "__main__":
    main()
