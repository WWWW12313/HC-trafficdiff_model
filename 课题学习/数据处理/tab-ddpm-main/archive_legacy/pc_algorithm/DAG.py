import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import LabelEncoder
# 引入 causal-learn 的核心模块
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

# ==========================================
# 1. 数据准备与特征工程
# ==========================================
print("--- 1. 数据加载与预处理 ---")

# 读取上一环节生成的全量清洗数据
# 请确保此文件在当前目录下
df = pd.read_csv("NYC_Full_Processed_2017.csv")

# A. 特征选择
# 我们精选了三个层级的关键变量：
# 1. 外生层 (Environment/Context): 时间、区域、天气、路面
# 2. 物理/中介层 (Physical/Mechanism): 车辆类型(重型/弱势群体)、核心致因
# 3. 结果层 (Consequence): 伤亡情况
features = [
    "HOUR", "BOROUGH", 
    "WEATHER_TYPE", "ROAD_SURFACE", 
    "HAS_HEAVY_VEHICLE", "HAS_VRU", 
    "GLOBAL_PRIMARY_CAUSE", 
    "NUMBER OF PERSONS INJURED", 
    "NUMBER OF PERSONS KILLED"
]

data = df[features].copy()

# B. 关键调整：结果二值化 (Binarization)
# 原因：伤亡数据是极度长尾的 (Zero-inflated)，直接作为连续变量跑 PC 算法会导致统计检验失效。
# 转换为 0/1 (无/有) 能显著提高因果方向识别的准确率。
data["INJURY_BIN"] = (data["NUMBER OF PERSONS INJURED"] > 0).astype(int)
data["FATALITY_BIN"] = (data["NUMBER OF PERSONS KILLED"] > 0).astype(int)

# C. 时间分段 (Discretization)
# 将连续的小时转换为具有社会学意义的时段
def bin_hour(h):
    if 6 <= h < 10: return "Morning_Rush"   # 早高峰
    elif 16 <= h < 20: return "Evening_Rush" # 晚高峰
    elif 0 <= h < 6: return "Night"          # 深夜
    else: return "Day_Normal"                # 平峰
data["TIME_PERIOD"] = data["HOUR"].apply(bin_hour)

# ==========================================
# 2. 数据编码 (Label Encoding)
# ==========================================
# PC 算法基于卡方检验 (Chi-square)，要求输入为离散的整数编码
node_names = [
    "TIME_PERIOD", "BOROUGH",           # 外生
    "WEATHER_TYPE", "ROAD_SURFACE",     # 环境
    "HAS_HEAVY_VEHICLE", "HAS_VRU",     # 物理约束
    "GLOBAL_PRIMARY_CAUSE",             # 核心致因
    "INJURY_BIN", "FATALITY_BIN"        # 结果
]

data_encoded = data[node_names].copy()
le_dict = {}

for col in data_encoded.columns:
    le = LabelEncoder()
    # 转换为字符串再编码，防止混合类型报错
    data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
    le_dict[col] = le

# 转换为 numpy 数组供算法输入
data_matrix = data_encoded.to_numpy()

# ==========================================
# 3. 运行 PC 算法 (核心步骤)
# ==========================================
print("--- 2. 正在运行 PC 算法挖掘因果结构 ---")
print("注意：数据量较大时可能需要几分钟...")

# 参数说明：
# alpha=0.01: 显著性水平。值越小，条件独立性测试越严格，边的数量越少（保留最强的因果边）。
# indep_test=chisq: 使用卡方检验作为独立性测试方法（适用于离散数据）。
cg = pc(data_matrix, alpha=0.01, indep_test=chisq, node_names=node_names)

# ==========================================
# 4. 因果图可视化 (分层着色)
# ==========================================
print("--- 3. 正在绘制分层因果图 ---")

# 从 PC 算法结果提取邻接矩阵
# matrix[i,j] = 1, matrix[j,i] = -1  => i -> j (有向边)
# matrix[i,j] = -1, matrix[j,i] = -1 => i -- j (无向边/未定)
adj_matrix = cg.G.graph 

G = nx.DiGraph()

# 添加节点
for name in node_names:
    G.add_node(name)

# 添加边
for i in range(len(node_names)):
    for j in range(len(node_names)):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1: # 有向边 i->j
            G.add_edge(node_names[i], node_names[j])
        elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1: # 无向边 i--j
            if i < j: # 避免重复添加
                G.add_edge(node_names[i], node_names[j], style='dashed') # 虚线表示方向未定

# --- 可视化配置 ---
plt.figure(figsize=(16, 10))

# 布局算法 (Spring Layout)
pos = nx.spring_layout(G, k=0.6, iterations=60, seed=42)

# 颜色映射 (Color Coding) - 对应 PPT 的三层逻辑
color_map = []
for node in G.nodes():
    # 1. 外生/环境层 (蓝色系)
    if node in ["TIME_PERIOD", "BOROUGH", "WEATHER_TYPE", "ROAD_SURFACE"]:
        color_map.append('#87CEFA') # Light Sky Blue
    # 2. 物理/致因层 (金色/橙色系)
    elif node in ["HAS_HEAVY_VEHICLE", "HAS_VRU", "GLOBAL_PRIMARY_CAUSE"]:
        color_map.append('#FFD700') # Gold
    # 3. 结果层 (红色系)
    else: # INJURY_BIN, FATALITY_BIN
        color_map.append('#FF6347') # Tomato Red

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2500, alpha=0.9, edgecolors='gray')
nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold')

# 绘制边 (区分实线和虚线)
solid_edges = [e for e in G.edges(data=True) if 'style' not in e[2]]
dashed_edges = [e for e in G.edges(data=True) if 'style' in e[2]]

nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2.0, edge_color='#555555', arrowsize=20, arrowstyle='->')
nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=2.0, edge_color='#AAAAAA', style='dashed')

# 添加图例和标题
plt.title("Causal Graph (PC Algorithm): NYC Traffic Accidents 2017", fontsize=18, fontweight='bold', pad=20)

# 手动添加图例 (Legend)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Environment (Exogenous)', markerfacecolor='#87CEFA', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Mechanism (Physical/Cause)', markerfacecolor='#FFD700', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Consequence (Severity)', markerfacecolor='#FF6347', markersize=15),
    Line2D([0], [0], color='#555555', lw=2, label='Directed Causal Edge (->)'),
    Line2D([0], [0], color='#AAAAAA', lw=2, linestyle='--', label='Undirected Association (--)')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.axis('off')
plt.tight_layout()

# 保存高清图片用于 PPT
output_file = "Causal_Graph_PC_Algorithm.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 因果图已保存至: {output_file}")
plt.show()

# ==========================================
# 5. 输出关键因果路径 (用于 PPT 解说)
# ==========================================
print("\n--- 关键因果发现 (Interpretation) ---")
print("如果算法运行理想，您应该能在图中观察到以下逻辑链 (PPT重点)：")

def check_edge(u, v):
    if G.has_edge(u, v):
        print(f"✅ 发现: {u} -> {v}")
    elif G.has_edge(v, u):
        print(f"⚠️ 反向: {v} -> {u} (可能需专家知识修正)")
    else:
        print(f"❌ 未发现直接连接: {u} 与 {v}")

print("\n1. 物理致死性验证:")
check_edge("HAS_HEAVY_VEHICLE", "FATALITY_BIN")

print("\n2. 环境对路面的影响:")
check_edge("WEATHER_TYPE", "ROAD_SURFACE")

print("\n3. 弱势群体对受伤的影响:")
check_edge("HAS_VRU", "INJURY_BIN")

print("\n4. 核心致因路径:")
check_edge("ROAD_SURFACE", "GLOBAL_PRIMARY_CAUSE")
check_edge("GLOBAL_PRIMARY_CAUSE", "INJURY_BIN")