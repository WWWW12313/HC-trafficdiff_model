from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline 04: evaluate metrics")
    parser.add_argument("--config", type=str, default="configs/v9_experiment.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print("[pipeline/04] placeholder")
    print(f"config={repo_root / args.config}")
    print("TODO: evaluate JS, logic violation, TSTR, and privacy metrics")


if __name__ == "__main__":
    main()
