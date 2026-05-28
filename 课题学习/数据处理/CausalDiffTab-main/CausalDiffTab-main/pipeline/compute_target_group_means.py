"""预计算训练集的 group-wise injury mean，用于 Inference-time Macro Guidance."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

CDT_ROOT = Path(__file__).resolve().parent.parent

def main():
    train_csv = CDT_ROOT / "data" / "nyc_crash_2024_v2" / "train.csv"
    df = pd.read_csv(train_csv)
    
    target_col = "NUMBER OF PERSONS INJURED"
    group_cols = ["SEASON", "WEATHER_CONDITION", "OSM_TYPE"]
    
    # 确保列存在
    available_group_cols = [c for c in group_cols if c in df.columns]
    print(f"Group columns: {available_group_cols}")
    
    # 计算 group mean
    group_means = df.groupby(available_group_cols)[target_col].mean().reset_index()
    group_counts = df.groupby(available_group_cols)[target_col].count().reset_index(name="count")
    
    # 合并
    stats = group_means.merge(group_counts, on=available_group_cols)
    
    # 转为 dict
    records = stats.to_dict(orient="records")
    
    output = {
        "target_col": target_col,
        "group_cols": available_group_cols,
        "global_mean": float(df[target_col].mean()),
        "global_std": float(df[target_col].std()),
        "group_means": [
            {
                "group_key": "|".join(str(r[c]) for c in available_group_cols),
                "mean": float(r[target_col]),
                "count": int(r["count"]),
            }
            for r in records
        ]
    }
    
    out_path = CDT_ROOT / "data" / "processed" / "target_group_means.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(records)} group means to {out_path}")
    print(f"Global mean: {output['global_mean']:.4f}, std: {output['global_std']:.4f}")

if __name__ == "__main__":
    main()
