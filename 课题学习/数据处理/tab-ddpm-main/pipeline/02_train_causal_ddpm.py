from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 02: train causal ddpm")
    parser.add_argument("--config", type=str, default="configs/v9_experiment.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print("[pipeline/02] placeholder")
    print(f"config={repo_root / args.config}")
    print("TODO: wire src/data_loader + src/diffusion_core + src/causal_module")


if __name__ == "__main__":
    main()
