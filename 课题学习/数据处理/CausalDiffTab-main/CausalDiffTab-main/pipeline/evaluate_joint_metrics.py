"""
Module 1: Joint Distribution and Causal Structure Metrics.

This module provides:
1) Conditional mutual information (CMI) error between real/synthetic data.
2) Structural Hamming Distance (SHD) between two causal adjacency matrices.

Notes:
- sklearn does not expose a direct CMI estimator. We use a practical proxy:
  CMI(X;Y|Z) ~= MI(residual(X|Z), residual(Y|Z)).
- Residuals are produced with RandomForestRegressor to capture nonlinearity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


@dataclass
class CMIPairSpec:
    """Describe one CMI measurement tuple: I(x; y | cond).

    Attributes:
        x_col: Target variable X.
        y_col: Target variable Y.
        cond_cols: Conditioning variables Z.
    """

    x_col: str
    y_col: str
    cond_cols: Sequence[str]


class ConditionalMutualInformationEvaluator:
    """Evaluate CMI proxy and CMI error between real/synthetic DataFrames."""

    def __init__(self, random_state: int = 42, n_estimators: int = 200) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators

    @staticmethod
    def _to_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        out = df.loc[:, list(columns)].copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _residualize(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return residual y - f(z), with a robust fallback for empty z."""
        if z.size == 0:
            centered = y - np.nanmean(y)
            return np.where(np.isfinite(centered), centered, 0.0)

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(z, y)
        pred = model.predict(z)
        residual = y - pred
        return np.where(np.isfinite(residual), residual, 0.0)

    def estimate_cmi_proxy(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        cond_cols: Sequence[str],
    ) -> float:
        """Estimate CMI proxy I(X;Y|Z) via MI on residualized variables.

        Returns NaN if required columns are missing or too few valid rows exist.
        """
        needed = [x_col, y_col] + list(cond_cols)
        if any(c not in df.columns for c in needed):
            return float("nan")

        sub = self._to_numeric_frame(df, needed).dropna()
        if len(sub) < 32:
            return float("nan")

        x = sub[x_col].to_numpy(dtype=np.float64)
        y = sub[y_col].to_numpy(dtype=np.float64)
        z = sub[list(cond_cols)].to_numpy(dtype=np.float64) if cond_cols else np.empty((len(sub), 0))

        x_res = self._residualize(x, z)
        y_res = self._residualize(y, z)

        # MI(Y_res ; X_res), treat X_res as one-feature input.
        mi = mutual_info_regression(
            X=x_res.reshape(-1, 1),
            y=y_res,
            random_state=self.random_state,
        )
        return float(mi[0])

    def evaluate_cmi_error(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        specs: Sequence[CMIPairSpec],
        eps: float = 1e-8,
    ) -> pd.DataFrame:
        """Compute per-spec CMI on real/synthetic data and return error table."""
        rows: List[Dict[str, Any]] = []
        for s in specs:
            real_cmi = self.estimate_cmi_proxy(real_df, s.x_col, s.y_col, s.cond_cols)
            syn_cmi = self.estimate_cmi_proxy(syn_df, s.x_col, s.y_col, s.cond_cols)
            abs_err = float(np.nan) if (np.isnan(real_cmi) or np.isnan(syn_cmi)) else float(abs(real_cmi - syn_cmi))
            rel_err = (
                float(np.nan)
                if (np.isnan(abs_err) or np.isnan(real_cmi))
                else float(abs_err / (abs(real_cmi) + eps))
            )
            rows.append(
                {
                    "x_col": s.x_col,
                    "y_col": s.y_col,
                    "cond_cols": ",".join(s.cond_cols),
                    "real_cmi": real_cmi,
                    "syn_cmi": syn_cmi,
                    "cmi_abs_error": abs_err,
                    "cmi_rel_error": rel_err,
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["real_cmi"] = out["real_cmi"].round(6)
            out["syn_cmi"] = out["syn_cmi"].round(6)
            out["cmi_abs_error"] = out["cmi_abs_error"].round(6)
            out["cmi_rel_error"] = out["cmi_rel_error"].round(6)
        return out


def _to_binary_adjacency(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be square 2D array, got shape={a.shape}")
    b = (np.abs(a) > 0).astype(np.int32)
    np.fill_diagonal(b, 0)
    return b


def structural_hamming_distance(real_adj: np.ndarray, syn_adj: np.ndarray) -> int:
    """Compute directed SHD between two adjacency matrices.

    SHD counts edge additions, deletions, and reversals.
    For a reversed edge i->j vs j->i, this implementation counts 1 (not 2).
    """
    b_true = _to_binary_adjacency(real_adj)
    b_pred = _to_binary_adjacency(syn_adj)

    if b_true.shape != b_pred.shape:
        raise ValueError(f"Adjacency shape mismatch: true={b_true.shape}, pred={b_pred.shape}")

    d = b_true.shape[0]
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t_ij, t_ji = b_true[i, j], b_true[j, i]
            p_ij, p_ji = b_pred[i, j], b_pred[j, i]

            # same unordered pair pattern -> no penalty.
            if t_ij == p_ij and t_ji == p_ji:
                continue

            # reversal case: one direction in each graph but opposite.
            if (t_ij == 1 and t_ji == 0 and p_ij == 0 and p_ji == 1) or (
                t_ij == 0 and t_ji == 1 and p_ij == 1 and p_ji == 0
            ):
                shd += 1
                continue

            # otherwise, count by directed mismatch for this unordered pair.
            shd += abs(t_ij - p_ij) + abs(t_ji - p_ji)

    return int(shd)


def load_adjacency(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Adjacency file not found: {p}")
    if p.suffix.lower() == ".npy":
        return np.load(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, header=None).values
    raise ValueError(f"Unsupported adjacency format: {p.suffix}, use .npy or .csv")


def evaluate_shd_from_files(real_adj_path: str | Path, syn_adj_path: str | Path) -> Dict[str, Any]:
    """Convenience helper for direct SHD evaluation from files."""
    a_real = load_adjacency(real_adj_path)
    a_syn = load_adjacency(syn_adj_path)
    shd = structural_hamming_distance(a_real, a_syn)
    d = int(np.asarray(a_real).shape[0])
    max_edges = d * (d - 1)
    shd_norm = float(shd / max(max_edges, 1))
    return {
        "real_adj_path": str(real_adj_path),
        "syn_adj_path": str(syn_adj_path),
        "n_nodes": d,
        "shd": shd,
        "shd_normalized": round(shd_norm, 6),
    }
