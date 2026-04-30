"""Unified entrypoint for causal TabDDPM training.

This file keeps backward compatibility with scripts expecting `train_causal.py`.
Core implementation lives in `train_causal_yandex.py`.
"""

from train_causal_yandex import (  # noqa: F401
    NEW_CAUSAL_EDGES,
    build_tensor_penalty_mask,
    train_yandex_causal,
)


if __name__ == "__main__":
    train_yandex_causal()
