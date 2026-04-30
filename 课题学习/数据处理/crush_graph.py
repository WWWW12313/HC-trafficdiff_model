import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置绘图风格
plt.style.use('default') 

print("--- 构建基于 Haddon Matrix 与 RAD 框架的改进因果图 ---")

# 1. 定义节点与层级
nodes_layer_1 = ['MONTH', 'HOUR', 'DAY_OF_WEEK', 'BOROUGH'] # 时空背景
nodes_layer_2 = ['WEATHER_TYPE', 'temp', 'PRECIPITATION']   # 气象环境
nodes_layer_3 = ['ROAD_TYPE', 'LANES', 'SPEED_LIMIT', 'MEDIAN'] # 道路静态属性
nodes_layer_4 = ['ROAD_SURFACE'] # 物理中介 (环境+道路共同作用)
nodes_layer_5 = ['PRIMARY_CAUSE'] # 行为致因 (Crash Event)
nodes_layer_6 = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'] # 后果 (Outcome)

all_nodes = nodes_layer_1 + nodes_layer_2 + nodes_layer_3 + nodes_layer_4 + nodes_layer_5 + nodes_layer_6

# 2. 定义边 (基于文献的强因果)
edges = [
    # --- 环境生成链 (Meteorological Chain) ---
    ('MONTH', 'temp'),           # 季节决定气温
    ('MONTH', 'WEATHER_TYPE'),   # 季节决定天气分布
    ('temp', 'PRECIPITATION'),   # 温度影响降水形态(雪/雨)
    
    # --- 物理中介链 (Physical Surface Chain) ---
    # 文献: Elvik (2002) - Road safety effects of winter maintenance
    ('PRECIPITATION', 'ROAD_SURFACE'), 
    ('temp', 'ROAD_SURFACE'), 
    ('WEATHER_TYPE', 'ROAD_SURFACE'),
    
    # --- 道路环境链 (Infrastructure Context) ---
    # 空间决定道路类型 (曼哈顿街道 vs 皇后区高速)
    ('BOROUGH', 'SPEED_LIMIT'),
    ('BOROUGH', 'LANES'),
    
    # --- 行为致因链 (Behavioral Mechanism) ---
    # 文献: Risk Homeostasis Theory
    ('ROAD_SURFACE', 'PRIMARY_CAUSE'),  # 路滑 -> 失控/刹车不及
    ('HOUR', 'PRIMARY_CAUSE'),          # 深夜 -> 疲劳/酒驾
    ('SPEED_LIMIT', 'PRIMARY_CAUSE'),   # 限速高 -> 容易超速/追尾
    ('LANES', 'PRIMARY_CAUSE'),         # 多车道 -> 变道事故
    
    # --- 后果链 (Severity Outcome) ---
    # 文献: Kinetic Energy Theory (E=1/2mv^2)
    ('SPEED_LIMIT', 'NUMBER OF PERSONS INJURED'), 
    ('SPEED_LIMIT', 'NUMBER OF PERSONS KILLED'),
    ('PRIMARY_CAUSE', 'NUMBER OF PERSONS INJURED'), # 事故类型决定伤害
    ('PRIMARY_CAUSE', 'NUMBER OF PERSONS KILLED')
]

# 3. 初始化图
G = nx.DiGraph()
G.add_nodes_from(all_nodes)
G.add_edges_from(edges)

# 4. 专业的层级布局 (手动精调坐标以符合论文排版)
pos = {
    # Layer 1: Context (Top)
    'MONTH': (0, 10), 'BOROUGH': (2, 10), 'HOUR': (4, 10), 'DAY_OF_WEEK': (6, 10),
    
    # Layer 2: Environment & Road (Upper Middle)
    'temp': (0, 7), 'WEATHER_TYPE': (1, 8), 'PRECIPITATION': (2, 7),
    'ROAD_TYPE': (3, 7), 'LANES': (4, 7), 'SPEED_LIMIT': (5, 7), 'MEDIAN': (6, 7),
    
    # Layer 3: Physical Mediator (Middle)
    'ROAD_SURFACE': (1.5, 4), 
    
    # Layer 4: Mechanism (Lower Middle)
    'PRIMARY_CAUSE': (3.5, 4),
    
    # Layer 5: Outcome (Bottom)
    'NUMBER OF PERSONS INJURED': (2, 1),
    'NUMBER OF PERSONS KILLED': (5, 1)
}

# 5. 绘图
plt.figure(figsize=(12, 9))
ax = plt.gca()

# 绘制层级背景框 (Highlighting Layers)
# Pre-Crash Phase (Context)
rect1 = patches.Rectangle((-0.5, 6), 7, 5, linewidth=1, edgecolor='none', facecolor='#e1f5fe', alpha=0.3)
ax.add_patch(rect1)
plt.text(-0.3, 10.5, "Phase I: Pre-Crash Context\n(Exogenous Variables)", fontsize=10, fontweight='bold', color='#01579b')

# Crash Phase (Mechanism)
rect2 = patches.Rectangle((-0.5, 3), 7, 2.5, linewidth=1, edgecolor='none', facecolor='#fff9c4', alpha=0.3)
ax.add_patch(rect2)
plt.text(-0.3, 5.2, "Phase II: Crash Mechanism\n(Physical & Behavioral)", fontsize=10, fontweight='bold', color='#f57f17')

# Post-Crash Phase (Outcome)
rect3 = patches.Rectangle((-0.5, 0), 7, 2.5, linewidth=1, edgecolor='none', facecolor='#e8f5e9', alpha=0.3)
ax.add_patch(rect3)
plt.text(-0.3, 2.2, "Phase III: Post-Crash Outcome\n(Severity)", fontsize=10, fontweight='bold', color='#1b5e20')

# 绘制节点与边
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='white', edgecolors='black', linewidths=1.5)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

# 绘制边 (区分不同类型的边)
# 普通因果
nx.draw_networkx_edges(G, pos, edge_color='#455a64', arrows=True, arrowsize=15, width=1.5)
# 关键物理因果 (加粗高亮)
critical_edges = [('PRECIPITATION', 'ROAD_SURFACE'), ('temp', 'ROAD_SURFACE'), ('SPEED_LIMIT', 'NUMBER OF PERSONS KILLED')]
nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color='#d32f2f', arrows=True, arrowsize=20, width=2.5)

plt.title("Hierarchical Causal Graph for Crash Data Generation\n(Based on Haddon Matrix & RAD Framework)", fontsize=14, pad=20)
plt.axis('off')

# 保存
output_file = 'improved_causal_graph.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 改进版因果图已保存至: {output_file}")