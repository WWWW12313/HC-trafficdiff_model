import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

# Find the latest runs
runs_dir = "runs"
subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
latest_run = max(subdirs, key=os.path.getmtime)
csv_path = os.path.join(latest_run, "training_loss_log.csv")

if not os.path.exists(csv_path):
    print(f"Log not found in {latest_run}")
    sys.exit(1)

df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label='Total Loss', color='#1f77b4', linewidth=2)

plt.title('Causal TabDDPM Training Loss Curve (Effective Batch Size = 64)', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

out_path = os.path.join(latest_run, "loss_curve.png")
plt.savefig(out_path, dpi=300)
print(f"Plot saved to: {out_path}")
