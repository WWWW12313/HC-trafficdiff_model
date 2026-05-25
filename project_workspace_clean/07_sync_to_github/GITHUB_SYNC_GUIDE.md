# GitHub 同步方案

> 目标仓库：`git@github.com:WWWW12313/HC-trafficdiff_model.git`  
> 当前已同步副本：`C:\Users\Admin\Desktop\hujunzhe_release_upload\`（commit `a50e201`）  
> 本文档说明后续如何持续同步整理后的结构

---

## 一、内容分类：适合 / 不适合同步 GitHub

### ✅ 适合同步的内容

| 类型 | 具体内容 | 目录 |
|------|---------|------|
| 核心代码 | `src/*.py`、`pipeline/*.py` | 01~04 各链路 |
| 实验配置 | `configs/experiments/*.yaml` | 05_experiment_configs |
| 因果矩阵（小文件） | `*.csv`、`*.json`（非 `.npy`） | 04_causal_graph |
| 实验日志 | `Experiment_Logs/*.md` | 06_logs_and_reports |
| 说明文档 | `README.md`、`MAINLINE.md`、`LOG_INDEX.md` | 各目录 |
| 脚本 | `Experimental script/*.py`、诊断脚本 | scripts/ |
| 框架配置 | `CausalDiffTab.yaml`、`.gitignore` | 项目根 |

### ❌ 不适合同步的内容

| 类型 | 原因 | 建议处理方式 |
|------|------|------------|
| `.npy` 训练数据 | 单文件 50MB-360MB，总量数 GB | 用 `.gitignore` 排除，路径记录在 `snapshots/LARGE_FILES_REFERENCE.md` |
| Checkpoint（`.pt`、`.bin`） | 单文件数百 MB | 同上 |
| 大型原始 CSV（> 10MB） | 历史数据文件 | 同上 |
| `__pycache__/`、`*.pyc` | 编译缓存 | gitignore |
| `swanlog/`、`runs/` | 训练日志（大量小文件） | gitignore，或只同步最新一次 |
| `results/` 大结果文件 | 合成 CSV 通常 > 50MB | gitignore，只保留摘要 |
| `cache/` | API 缓存 | gitignore |
| `ckpt/` | Checkpoint 目录 | gitignore |

---

## 二、建议 `.gitignore`

```gitignore
# ===== 数据文件 =====
*.npy
*.npy.gz
*.h5
*.hdf5
*.parquet

# ===== 大型 CSV（超过 5MB 的原始数据）=====
# 小型示例 CSV（< 1MB）可以保留
nyc_accidents_*.csv
nyc_2017_*.csv
nyc_2024_*.csv
nyc_2025_*.csv
*_pristine_*.csv
*_speedfix.csv
*_backup_*.csv

# ===== Checkpoint / 模型权重 =====
*.pt
*.pth
*.bin
*.pkl
*.joblib
ckpt/
tuned_models/

# ===== 大型结果文件 =====
results/synth_*.csv
results/synthetic/
synthetic/

# ===== 训练 / 实验日志大文件 =====
swanlog/
runs/
*.log
build_2025_like_2017.log
train_*.log
prepare_*.log
run_v2_*.log
v2_*.log
transfer_eval.log
postcovid_transfer_*.log

# ===== 缓存 =====
__pycache__/
*.pyc
*.pyo
.pytest_cache/
cache/
.cache/
catboost_info/

# ===== IDE / OS =====
.vscode/settings.json
.DS_Store
Thumbs.db
*.swp

# ===== Python 环境 =====
*.egg-info/
dist/
build/
.eggs/
*.egg

# ===== 临时文件 =====
*.tmp
*.bak
_diag_*.py
```

---

## 三、逐步同步到 GitHub 的推荐步骤

### 第一步：更新 release 副本（推荐方式）

当前已有轻量 release 目录：`C:\Users\Admin\Desktop\hujunzhe_release_upload\`

```powershell
cd C:\Users\Admin\Desktop\hujunzhe_release_upload

# 1. 把整理好的 project_workspace_clean 内容同步进来
# （选择性复制，跳过大文件）
$src = "C:\Users\Admin\Desktop\hujunzhe\project_workspace_clean"

# 同步核心内容（排除 .npy 文件）
robocopy $src . /E /XF *.npy *.pt *.pkl /XD __pycache__ .git

# 2. 提交
git add .
git commit -m "refactor: add project_workspace_clean structure with mainline docs"

# 3. 推送（SSH 已配置）
git push origin master
```

### 第二步：每次实验后同步

```powershell
cd C:\Users\Admin\Desktop\hujunzhe_release_upload

# 只同步新增/修改的文档和脚本
git add 06_logs_and_reports/Experiment_Logs/
git add 05_experiment_configs/
git add src/  # 如有代码修改
git commit -m "exp: 2026-MM-DD <本次实验摘要>"
git push origin master
```

### 第三步：保证日志体系可持续同步

每次追加实验日志后：
1. 将新的 `YYYY-MM-DD.md` 复制到 `06_logs_and_reports/Experiment_Logs/`
2. 更新 `06_logs_and_reports/LOG_INDEX.md`（在对应分类下追加一行）
3. 按上面第二步提交推送

---

## 四、不推荐的同步方式

- ❌ **不要直接 push 原始 `hujunzhe` 目录**：含有 ~15GB 数据文件，会卡死
- ❌ **不要 `git add -A` 后强制推送**：可能带入 .npy / checkpoint
- ❌ **不要把 `swanlog/` 或 `runs/` 推上去**：大量小文件会拖慢推送
- ❌ **不要在 release 副本里修改代码然后再手动同步回原始项目**：会造成双向不一致，应始终在原始项目修改，然后单向复制到 release 副本

---

## 五、SSH 连接信息

```
SSH Key：~/.ssh/id_ed25519（ED25519，comment: hujunzhe-crashgen）
已添加到 GitHub 账户 WWWW12313
远端：git@github.com:WWWW12313/HC-trafficdiff_model.git
```

如 SSH 超时，检查：`ssh -T git@github.com`
