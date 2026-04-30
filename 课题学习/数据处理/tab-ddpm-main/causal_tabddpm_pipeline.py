import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder

# =========================================================
# 【修复核心】将因果关系边列表提升为全局变量，允许外部脚本导入
# =========================================================
fci_discovered_edges = [
    # ==== 1. 基础设施与路网地理系统 ====
    ("BOROUGH", "ZIP CODE"),
    ("ZIP CODE", "LATITUDE"), ("ZIP CODE", "LONGITUDE"),
    ("LATITUDE", "LONGITUDE"),
    ("OSM_SPEED_TAG", "REAL_SPEED_LIMIT"),
    ("OSM_LANES_TAG", "INFERRED_LANES"),
    ("HAS_DIVIDER", "REAL_SPEED_LIMIT"),
    ("BOROUGH", "IS_OVERSIZED_VEHICLE"),
    ("REAL_SPEED_LIMIT", "IS_SPEEDING"),

    # ==== 2. 天气系统 ====
    ("TEMP_C", "REAL_WEATHER"),
    ("prcp", "REAL_WEATHER"),
    ("WIND_SPEED_KMH", "REAL_WEATHER"),
    ("REAL_WEATHER", "WEATHER_CONDITION"),
    ("WEATHER_CONDITION", "IS_VISION_OBSCURED"),
    ("WEATHER_CONDITION", "IS_POOR_ROAD_CONDITION"),
    ("WEATHER_CONDITION", "IS_ANIMAL_RELATED"),

    # ==== 3. 时间与多重动态影响 ====
    ("CRASH TIME", "HAS_TRAFFIC_SIGNAL"),
    ("CRASH TIME", "IS_VISION_OBSCURED"),
    ("CRASH TIME", "TOTAL_VEHICLES"),
    ("CRASH TIME", "IS_AGGRESSIVE_DRIVING"),
    ("CRASH TIME", "IS_ALCOHOL_INVOLVED"),
    ("CRASH_WEEKDAY", "IS_ALCOHOL_INVOLVED"),

    # ==== 4. 车辆因素与状况 ====
    ("IS_OVERSIZED_VEHICLE", "IS_VISION_OBSCURED"),
    ("IS_VEHICLE_DEFECT", "IS_SPEEDING"),
    ("VEHICLE TYPE CODE 1", "IS_VEHICLE_DEFECT"),
    ("VEHICLE TYPE CODE 1", "IS_OVERSIZED_VEHICLE"),

    # ==== 5. 驾驶员异常状态诱发事故模式 ====
    ("IS_INEXPERIENCED_DRIVER", "IS_DISTRACTED"),
    ("IS_FATIGUED", "IS_DISTRACTED"),
    ("IS_ALCOHOL_INVOLVED", "IS_AGGRESSIVE_DRIVING"),
    ("IS_ALCOHOL_INVOLVED", "IS_SPEEDING"),

    # ==== 6. 危险驾驶行为传导 ====
    ("IS_AGGRESSIVE_DRIVING", "IS_SPEEDING"),
    ("IS_AGGRESSIVE_DRIVING", "IS_FOLLOWING_TOO_CLOSE"),
    ("IS_AGGRESSIVE_DRIVING", "IS_IMPROPER_LANE_USE"),
    ("IS_DISTRACTED", "IS_FAILURE_TO_YIELD"),
    ("IS_DISTRACTED", "IS_TRAFFIC_SIGNAL_VIOLATION"),
    ("IS_SPEEDING", "IS_IMPROPER_TURNING"),
    ("HAS_TRAFFIC_SIGNAL", "IS_TRAFFIC_SIGNAL_VIOLATION"),
    ("INFERRED_LANES", "IS_IMPROPER_LANE_USE"),
    ("IS_TRAFFIC_SIGNAL_VIOLATION", "IS_MULTI_VEHICLE"),

    # ==== 7. 行人与非机动车 ====
    ("BOROUGH", "IS_PEDESTRIAN_CYCLIST_ERROR"),
    ("IS_VISION_OBSCURED", "IS_PEDESTRIAN_CYCLIST_ERROR"),

    # ==== 8. 事故形态影响最终伤亡结果 ====
    ("TOTAL_VEHICLES", "IS_MULTI_VEHICLE"),
    ("IS_SPEEDING", "NUMBER OF PERSONS INJURED"),
    ("IS_ALCOHOL_INVOLVED", "NUMBER OF PERSONS INJURED"),
    ("TOTAL_VEHICLES", "NUMBER OF PERSONS INJURED"),
    ("IS_PEDESTRIAN_CYCLIST_ERROR", "NUMBER OF PERSONS INJURED"),
    ("IS_MULTI_VEHICLE", "NUMBER OF PERSONS INJURED"),
    ("IS_FAILURE_TO_YIELD", "NUMBER OF PERSONS INJURED"),
    ("IS_VEHICLE_DEFECT", "NUMBER OF PERSONS INJURED")
]

class CausalTabularDataset(Dataset):
    def __init__(self, csv_path, causal_edges):
        """
        csv_path: 你的纯净版数据路径 (nyc_2017_pristine_v8.csv)
        causal_edges: FCI 跑出来的因果边列表，格式为 [(原因, 结果), (原因, 结果)...]
        """
        print(f"1. 正在加载纯净数据: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # =========================================================
        # 修复 1：剔除无法参与深度学习计算的纯文本ID和时间列
        # =========================================================
        cols_to_drop = ['COLLISION_ID', 'LOCATION', 'CRASH DATE', 'CRASH TIME', 'CRASH_FULL_TIME', 'coco']
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns])
        
        # =========================================================
        # 修复 2：将文本/分类特征转换为数值 (Ordinal Encoding)
        # =========================================================
        print("-> 正在将分类/文本特征转换为神经网络可读取的数值...")
        cat_cols = self.df.select_dtypes(exclude=['float64', 'int64', 'int32']).columns
        
        self.encoder = OrdinalEncoder()
        if len(cat_cols) > 0:
            self.df[cat_cols] = self.df[cat_cols].astype(str)
            self.df[cat_cols] = self.encoder.fit_transform(self.df[cat_cols])
            
        # 强制整个 DataFrame 变为 float32 (PyTorch 的标准类型)
        self.df = self.df.astype(np.float32)

        # =========================================================
        # 2. 构建有向无环图 (DAG)
        # =========================================================
        print("2. 构建因果拓扑图...")
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.df.columns)
        valid_edges = [
            (s, t) for s, t in causal_edges
            if s in self.df.columns and t in self.df.columns
        ]
        self.G.add_edges_from(valid_edges)
        
        # 3. 检查并强制转换为有向无环图 (打破可能存在的循环)
        if not nx.is_directed_acyclic_graph(self.G):
            print("⚠️ 警告: 因果图中存在环！正在尝试打破循环以保证严格的因果顺序...")
            cycles = list(nx.simple_cycles(self.G))
            for cycle in cycles:
                if self.G.has_edge(cycle[-1], cycle[0]):
                    self.G.remove_edge(cycle[-1], cycle[0])
            print("-> 循环已打破，DAG 构建成功。")
            
        # 4. 获取因果拓扑排序 (Topological Sort)
        self.topological_order = list(nx.topological_sort(self.G))
        print("\n【严格因果生成顺序】(神经网络将按照此顺序从左到右预测数据):")
        print(" -> ".join(self.topological_order))
        
        # 5. 按照因果顺序重排 DataFrame 列
        self.df_ordered = self.df[self.topological_order]
        
        # =========================================================
        # 6. Min-Max 归一化处理 (-1 到 1 之间，DDPM 的最爱)
        # =========================================================
        self.data_min = self.df_ordered.min()
        self.data_max = self.df_ordered.max()
        denominator = (self.data_max - self.data_min).replace(0, 1e-5) 
        
        # 将数据映射到 [0, 1]，然后再映射到 [-1, 1]
        self.df_normalized = 2 * ((self.df_ordered - self.data_min) / denominator) - 1
        
        # 转换为 PyTorch 张量
        self.tensor_data = torch.FloatTensor(self.df_normalized.values)
        print(f"\n✅ 数据准备就绪！特征维度: {self.tensor_data.shape[1]}, 样本量: {self.tensor_data.shape[0]}")

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx]

if __name__ == "__main__":
    # 实例化数据集 (直接使用全局变量 fci_discovered_edges)
    dataset = CausalTabularDataset("nyc_2017_pristine_v8.csv", fci_discovered_edges)
    
    # 测试 DataLoader
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    batch_data = next(iter(dataloader))
    print(f"\n[测试成功] 抽取了一个 Batch，形状: {batch_data.shape} (Batch_size, 因果特征数)")