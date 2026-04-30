import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import platform

# ================= 1. 中文字体配置 (防乱码核心) =================
print("正在配置中文字体...")

# 根据操作系统自动选择字体
system_name = platform.system()
if system_name == "Windows":
    font_path = "C:/Windows/Fonts/simhei.ttf" # 黑体
    # 如果没有simhei，尝试微软雅黑
    if not font_manager.os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/msyh.ttf"
elif system_name == "Darwin": # Mac
    font_path = "/System/Library/Fonts/PingFang.ttc"
else: # Linux
    font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

# 加载字体
try:
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 字体加载成功: {font_path}")
except:
    print("⚠️ 警告: 未找到指定中文字体，将使用默认字体(可能乱码)")

# ================= 2. 定义节点与层级 (中英对照) =================
# 节点名称映射 (用于图上显示中文)
node_labels = {
    'MONTH': '月份\n(Month)', 
    'HOUR': '小时\n(Hour)', 
    'DAY_OF_WEEK': '星期\n(Day)', 
    'BOROUGH': '行政区\n(Borough)',
    
    'WEATHER_TYPE': '天气类型\n(Weather)', 
    'temp': '气温\n(Temp)', 
    'PRECIPITATION': '降水\n(Precip)',
    
    'ROAD_TYPE': '道路等级\n(Type)', 
    'LANES': '车道数\n(Lanes)', 
    'SPEED_LIMIT': '限速\n(Speed)', 
    'MEDIAN': '隔离带\n(Median)',
    
    'ROAD_SURFACE': '路面状况\n(Surface)',
    
    'PRIMARY_CAUSE': '事故致因\n(Cause)',
    
    'NUMBER OF PERSONS INJURED': '受伤人数\n(Injured)', 
    'NUMBER OF PERSONS KILLED': '死亡人数\n(Killed)'
}

all_nodes = list(node_labels.keys())

# ================= 3. 定义边 (基于 Haddon Matrix & RAD) =================
edges = [
    # --- 环境生成链 ---
    ('MONTH', 'temp'),
    ('MONTH', 'WEATHER_TYPE'),
    ('temp', 'PRECIPITATION'),
    
    # --- 物理中介链 ---
    ('PRECIPITATION', 'ROAD_SURFACE'), 
    ('temp', 'ROAD_SURFACE'), 
    ('WEATHER_TYPE', 'ROAD_SURFACE'),
    
    # --- 道路环境链 ---
    ('BOROUGH', 'SPEED_LIMIT'),
    ('BOROUGH', 'LANES'),
    
    # --- 行为致因链 ---
    ('ROAD_SURFACE', 'PRIMARY_CAUSE'),
    ('HOUR', 'PRIMARY_CAUSE'),
    ('SPEED_LIMIT', 'PRIMARY_CAUSE'),
    ('LANES', 'PRIMARY_CAUSE'),
    
    # --- 后果链 ---
    ('SPEED_LIMIT', 'NUMBER OF PERSONS INJURED'), 
    ('SPEED_LIMIT', 'NUMBER OF PERSONS KILLED'),
    ('PRIMARY_CAUSE', 'NUMBER OF PERSONS INJURED'),
    ('PRIMARY_CAUSE', 'NUMBER OF PERSONS KILLED')
]

# 初始化图
G = nx.DiGraph()
G.add_nodes_from(all_nodes)
G.add_edges_from(edges)

# ================= 4. 专业的层级布局 =================
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

# ================= 5. 绘图 =================
plt.figure(figsize=(12, 9))
ax = plt.gca()

# --- 绘制阶段背景框 (中文) ---
# Phase I
rect1 = patches.Rectangle((-0.5, 6), 7, 5, linewidth=1, edgecolor='none', facecolor='#e1f5fe', alpha=0.3)
ax.add_patch(rect1)
plt.text(-0.3, 10.5, "阶段一：事故前置背景 (Pre-Crash Phase)\n[外生变量与环境]", fontsize=11, fontweight='bold', color='#01579b', fontproperties=prop)

# Phase II
rect2 = patches.Rectangle((-0.5, 3), 7, 2.5, linewidth=1, edgecolor='none', facecolor='#fff9c4', alpha=0.3)
ax.add_patch(rect2)
plt.text(-0.3, 5.2, "阶段二：事故发生机理 (Crash Phase)\n[物理中介与行为致因]", fontsize=11, fontweight='bold', color='#f57f17', fontproperties=prop)

# Phase III
rect3 = patches.Rectangle((-0.5, 0), 7, 2.5, linewidth=1, edgecolor='none', facecolor='#e8f5e9', alpha=0.3)
ax.add_patch(rect3)
plt.text(-0.3, 2.2, "阶段三：事故后果 (Post-Crash Phase)\n[严重程度]", fontsize=11, fontweight='bold', color='#1b5e20', fontproperties=prop)

# --- 绘制节点 ---
nx.draw_networkx_nodes(G, pos, node_size=2200, node_color='white', edgecolors='black', linewidths=1.5)

# 绘制标签 (使用中文映射)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold', font_family=prop.get_name())

# --- 绘制边 ---
# 普通因果
nx.draw_networkx_edges(G, pos, edge_color='#455a64', arrows=True, arrowsize=15, width=1.5)
# 关键物理因果 (红色加粗)
critical_edges = [('PRECIPITATION', 'ROAD_SURFACE'), ('temp', 'ROAD_SURFACE'), ('SPEED_LIMIT', 'NUMBER OF PERSONS KILLED')]
nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color='#d32f2f', arrows=True, arrowsize=20, width=2.5)

# --- 标题 ---
plt.title("基于哈顿矩阵(Haddon Matrix)与RAD框架的交通事故生成因果模型", fontsize=15, pad=20, fontproperties=prop)
plt.axis('off')

# 保存
output_file = 'causal_graph_cn.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 中文版因果图已保存至: {output_file}")