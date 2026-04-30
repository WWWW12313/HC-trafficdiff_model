# 定义映射字典
CAUSE_MAPPING = {
    # --- 1. Driver Distraction (分心) ---
    "Driver Inattention/Distraction": "Driver_Distraction",
    "Cell Phone (hand-Held)": "Driver_Distraction",
    "Cell Phone (hands-free)": "Driver_Distraction",
    "Eating or Drinking": "Driver_Distraction",
    "Using On Board Navigate Device": "Driver_Distraction",
    "Texting": "Driver_Distraction",
    "Passenger Distraction": "Driver_Distraction",
    "Outside Car Distraction": "Driver_Distraction",

    # --- 2. Driver State (状态/内因) ---
    "Alcohol Involvement": "Driver_State",
    "Drugs (illegal)": "Driver_State",
    "Prescription Medication": "Driver_State",
    "Fatigued/Drowsy": "Driver_State",
    "Fell Asleep": "Driver_State",
    "Illnes": "Driver_State",  # 包含原始拼写错误
    "Illness": "Driver_State",
    "Lost Consciousness": "Driver_State",
    "Physical Disability": "Driver_State",
    "Driver Inexperience": "Driver_State", # 也可以归为 Human Error，但此处作为内在属性

    # --- 3. Violation Rules (违规) ---
    "Unsafe Speed": "Violation_Rules",
    "Aggressive Driving/Road Rage": "Violation_Rules",
    "Failure to Yield Right-of-Way": "Violation_Rules",
    "Failure to Keep Right": "Violation_Rules",
    "Traffic Control Disregarded": "Violation_Rules",
    "Backing Unsafely": "Violation_Rules",
    "Turning Improperly": "Violation_Rules",
    "Driving Wrong Way": "Violation_Rules",
    "Passing or Lane Usage Improper": "Violation_Rules",

    # --- 4. Traffic Interaction (交互) ---
    "Following Too Closely": "Traffic_Interaction",
    "Unsafe Lane Changing": "Traffic_Interaction",
    "Passing Too Closely": "Traffic_Interaction",
    "Driverless/Runaway Vehicle": "Traffic_Interaction", # 通常指溜车
    "Reaction to Uninvolved Vehicle": "Traffic_Interaction",

    # --- 5. Vehicle Defect (车况) ---
    "Brakes Defective": "Vehicle_Defect",
    "Steering Failure": "Vehicle_Defect",
    "Tire Failure/Inadequate": "Vehicle_Defect",
    "Accelerator Defective": "Vehicle_Defect",
    "Headlights Defective": "Vehicle_Defect",
    "Tinted Windows": "Vehicle_Defect",
    "Oversized Vehicle": "Vehicle_Defect",

    # --- 6. Environment (环境/设施) ---
    "Pavement Slippery": "Environment",
    "Glare": "Environment",
    "View Obstructed/Limited": "Environment",
    "Animals Action": "Environment",
    "Lane Marking Improper/Inadequate": "Environment", # 设施问题归为环境
    "Traffic Control Device Improper/Non-Working": "Environment",
    "Pavement Defective": "Environment",
    "Obstruction/Debris": "Environment",
    "Shoulders Defective/Improper": "Environment",

    # --- 7. Ambiguous (模糊/空白) ---
    "Unspecified": "Unspecified",
    "Unknown": "Unspecified",
    "Other Vehicular": "Unspecified" # 信息量太少，归为模糊
}
from graphviz import Digraph

# 1. 您的映射字典
CAUSE_MAPPING = {
    # --- 1. Driver Distraction (分心) ---
    "Driver Inattention/Distraction": "Driver_Distraction",
    "Cell Phone (hand-Held)": "Driver_Distraction",
    "Cell Phone (hands-free)": "Driver_Distraction",
    "Eating or Drinking": "Driver_Distraction",
    "Using On Board Navigate Device": "Driver_Distraction",
    "Texting": "Driver_Distraction",
    "Passenger Distraction": "Driver_Distraction",
    "Outside Car Distraction": "Driver_Distraction",

    # --- 2. Driver State (状态/内因) ---
    "Alcohol Involvement": "Driver_State",
    "Drugs (illegal)": "Driver_State",
    "Prescription Medication": "Driver_State",
    "Fatigued/Drowsy": "Driver_State",
    "Fell Asleep": "Driver_State",
    "Illnes": "Driver_State",
    "Illness": "Driver_State",
    "Lost Consciousness": "Driver_State",
    "Physical Disability": "Driver_State",
    "Driver Inexperience": "Driver_State",

    # --- 3. Violation Rules (违规) ---
    "Unsafe Speed": "Violation_Rules",
    "Aggressive Driving/Road Rage": "Violation_Rules",
    "Failure to Yield Right-of-Way": "Violation_Rules",
    "Failure to Keep Right": "Violation_Rules",
    "Traffic Control Disregarded": "Violation_Rules",
    "Backing Unsafely": "Violation_Rules",
    "Turning Improperly": "Violation_Rules",
    "Driving Wrong Way": "Violation_Rules",
    "Passing or Lane Usage Improper": "Violation_Rules",

    # --- 4. Traffic Interaction (交互) ---
    "Following Too Closely": "Traffic_Interaction",
    "Unsafe Lane Changing": "Traffic_Interaction",
    "Passing Too Closely": "Traffic_Interaction",
    "Driverless/Runaway Vehicle": "Traffic_Interaction",
    "Reaction to Uninvolved Vehicle": "Traffic_Interaction",

    # --- 5. Vehicle Defect (车况) ---
    "Brakes Defective": "Vehicle_Defect",
    "Steering Failure": "Vehicle_Defect",
    "Tire Failure/Inadequate": "Vehicle_Defect",
    "Accelerator Defective": "Vehicle_Defect",
    "Headlights Defective": "Vehicle_Defect",
    "Tinted Windows": "Vehicle_Defect",
    "Oversized Vehicle": "Vehicle_Defect",

    # --- 6. Environment (环境/设施) ---
    "Pavement Slippery": "Environment",
    "Glare": "Environment",
    "View Obstructed/Limited": "Environment",
    "Animals Action": "Environment",
    "Lane Marking Improper/Inadequate": "Environment",
    "Traffic Control Device Improper/Non-Working": "Environment",
    "Pavement Defective": "Environment",
    "Obstruction/Debris": "Environment",
    "Shoulders Defective/Improper": "Environment",

    # --- 7. Ambiguous (模糊/空白) ---
    "Unspecified": "Unspecified",
    "Unknown": "Unspecified",
    "Other Vehicular": "Unspecified"
}

# 2. 反转字典逻辑：将 映射 -> {原始值列表}
category_tree = {}
for original, category in CAUSE_MAPPING.items():
    if category not in category_tree:
        category_tree[category] = []
    category_tree[category].append(original)

# 3. 初始化 Graphviz 对象
dot = Digraph(comment='Traffic Cause Taxonomy')
# 设置图的方向为从左到右 (LR)，这样更适合长文本
dot.attr(rankdir='LR', size='12,12', dpi='300')
dot.attr('node', shape='box', style='filled', fontname='Helvetica')

# 4. 定义颜色方案 (PPT 风格)
colors = {
    "Driver_Distraction": "#FFB3BA", # 浅红 (分心)
    "Driver_State": "#FFDFBA",       # 浅橙 (状态)
    "Violation_Rules": "#FFFFBA",    # 浅黄 (违规)
    "Traffic_Interaction": "#BAFFC9",# 浅绿 (交互)
    "Vehicle_Defect": "#BAE1FF",     # 浅蓝 (车辆)
    "Environment": "#E2F0CB",        # 灰绿 (环境)
    "Unspecified": "#E0E0E0"         # 灰色 (模糊)
}

# 5. 构建节点和边
root_name = "Global Primary Cause\n(7 Core Categories)"
dot.node('ROOT', root_name, shape='doubleoctagon', fillcolor='white', fontsize='16', fontweight='bold')

for category, items in category_tree.items():
    # 创建大类节点
    color = colors.get(category, "white")
    dot.node(category, category, fillcolor=color, fontsize='14', fontweight='bold')
    dot.edge('ROOT', category, penwidth='2.0')
    
    # 创建子类节点（原始值）
    for item in items:
        # 清理 item 名字中的特殊字符
        safe_item_id = str(hash(item)) 
        dot.node(safe_item_id, item, shape='note', fontsize='10', fillcolor='white', color=color)
        dot.edge(category, safe_item_id, color='gray')

# 6. 保存并渲染
try:
    # 渲染为 PDF 和 PNG
    output_path = dot.render('cause_taxonomy_tree', format='png', view=True)
    print(f"✅ 树状图已生成: {output_path}")
except Exception as e:
    print(f"❌ 生成失败，请确保安装了 Graphviz 软件。\n错误信息: {e}")