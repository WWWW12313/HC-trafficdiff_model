# 数据链路说明

> 源目录：`课题学习\数据处理\CausalDiffTab-main\CausalDiffTab-main\`  
> 当前口径：**2024 训练 / 2025 迁移测试**

---

## 链路总览

```
NYC OpenData Motor Vehicle Collisions (原始 CSV)
        ↓ download_dataset.py / 手动下载
raw_data/  (nyc_accidents_2024.csv, nyc_accidents_2025.csv …)
        ↓ build_2025_like_2017.py
        ↓ data_processor.py
features: 时间分箱 + 事故类型 0/1 + 车辆类型 0/1 + 经纬度保留
        ↓ road_snap.py
        ↓ (OSM 道路特征补全：speed_limit / road_type / num_lanes)
        ↓ weather match (OpenMeteo API)
enriched_csv  (nyc_2024_final.csv / nyc_2025_test.csv)
        ↓ prepare_2025_data.py
        ↓ prepare_dataset.py
data/nyc_crash/  → X_num_train.npy / X_cat_train.npy / y_train.npy …
```

---

## 各脚本职责

| 脚本 | 职责 | 注意事项 |
|------|------|---------|
| `build_2025_like_2017.py` | 将 2025 年原始数据按照 2017 schema 口径对齐，构建融合表 | **2024→2025 主线入口**；生成 `is_other_vehicle`（2026-04-29 新增）|
| `data_processor.py` | 核心特征工程：时间拆分、车辆类型合并、事故类型 0/1 化 | 字段定义的唯一权威来源；修改字段时这里是第一入口 |
| `prepare_dataset.py` | 从富化 CSV 生成 .npy 训练数据；domain rules 和 causal mask 注入点 | 修改 mask 策略时需同步此文件 |
| `prepare_2025_data.py` | 专门为 2025 年数据生成与训练集一致口径的测试集 | 必须与 2024 走相同特征工程路径 |
| `road_snap.py` | OSM 路网特征匹配（速限、路型、车道数）；匹配失败时 fallback | 需提前运行 `download_osm_cache.py` |
| `postprocess_samples.py` | 采样后后处理：坐标反归一化、弱语义一致性检查 | `--semantic_repair` 模式已验证对 TSTR 有负面影响，慎用 |
| `prepare_transfer_data.py` | 通用迁移数据准备（可用于其他年份组合） | 2024→2025 主线请用 `prepare_2025_data.py` |
| `prepare_postcovid_data.py` | ⚠️ **旧路线**：postcovid 2017→2025 口径 | 已归档，不在当前主线 |

---

## 输出数据位置（原始项目）

```
CausalDiffTab-main/data/
  nyc_crash/          ← 2024 训练集（当前主线）
    X_num_train.npy
    X_cat_train.npy
    y_train.npy
    X_num_val.npy / X_cat_val.npy / y_val.npy
    X_num_test.npy / X_cat_test.npy / y_test.npy
    train.csv / val.csv / test.csv
  nyc_crash_2025/     ← 2025 迁移测试集
    …
  nyc_crash_2024/     ← 与 nyc_crash/ 同义，部分脚本引用此路径
  nyc_stage1_2024/    ← Stage1（时空锚点）单独数据集
  nyc_stage2_2024/    ← Stage2（天气/OSM）单独数据集
```

> ⚠️ `.npy` 文件不进入 GitHub（体积过大），路径参照见 `snapshots/LARGE_FILES_REFERENCE.md`

---

## 关键注意事项

1. **schema 一致性**：2024 和 2025 必须走相同的 `build_2025_like_2017.py` + `data_processor.py` 路径，绝不能分别独立处理。
2. **`is_other_vehicle` 引入**（2026-04-29）：修改了 vehicle schema，导致旧 checkpoint 失效，需重新生成数据并重训。
3. **OSM 补全**：在无 OSM 缓存的新机器上首次运行需要较长时间下载，建议先运行 `download_osm_cache.py`。
4. **不使用 H3**：`apply_h3_roadcell_projection.py` 已移入 `scripts/` 作为消融脚本，数据链路不调用它。
