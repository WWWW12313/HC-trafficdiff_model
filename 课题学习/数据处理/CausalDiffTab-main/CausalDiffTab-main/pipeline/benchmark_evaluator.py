from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    model: str
    status: str
    avg_score: float
    metrics: Dict[str, float]
    error: str = ""


class BenchmarkEvaluator:
    """Run a standardized benchmark on tabular downstream tasks.

    Supported task types:
    - regression: R2, MSE, MAE
    - classification: F1-macro, F1-micro, Accuracy
    """

    def __init__(self, task_type: str = "regression", random_state: int = 42) -> None:
        task_type = str(task_type).lower().strip()
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")
        self.task_type = task_type
        self.random_state = int(random_state)

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

        X_train_arr = np.asarray(X_train.values, dtype=np.float32)
        X_test_arr = np.asarray(X_test.values, dtype=np.float32)

        models = self._build_models()
        rows: List[BenchmarkResult] = []

        if self.task_type == "regression":
            y_train_arr = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=np.float32)
            y_test_arr = pd.to_numeric(pd.Series(y_test), errors="coerce").to_numpy(dtype=np.float32)
            tr_mask = np.isfinite(y_train_arr)
            te_mask = np.isfinite(y_test_arr)
            if not np.any(tr_mask) or not np.any(te_mask):
                raise ValueError("No valid labels for regression benchmark evaluation")
            Xtr = X_train_arr[tr_mask]
            ytr = y_train_arr[tr_mask]
            Xte = X_test_arr[te_mask]
            yte = y_test_arr[te_mask]
        else:
            # For classification, if labels look continuous, auto-discretize by quantile bins.
            ytr_raw = pd.Series(y_train)
            yte_raw = pd.Series(y_test)
            ytr, yte = self._prepare_classification_labels(ytr_raw, yte_raw)
            Xtr = X_train_arr
            Xte = X_test_arr
            ytr = np.asarray(ytr)
            yte = np.asarray(yte)

        for name, model in models:
            print(f"[BenchmarkEvaluator] Running model: {name} ({self.task_type})")
            try:
                Xtr_local = Xtr
                ytr_local = ytr
                Xte_local = Xte
                yte_local = yte

                if self.task_type == "classification" and name == "xgboost":
                    # XGBoost classifier requires contiguous labels 0..K-1 in training split.
                    seen = np.sort(np.unique(np.asarray(ytr_local, dtype=np.int64)))
                    remap = {int(v): i for i, v in enumerate(seen)}
                    ytr_local = np.asarray([remap[int(v)] for v in np.asarray(ytr_local, dtype=np.int64)], dtype=np.int32)

                    yte_raw = np.asarray(yte_local, dtype=np.int64)
                    te_mask = np.isin(yte_raw, seen)
                    if not np.any(te_mask):
                        raise ValueError("No overlapping classes between train and test for xgboost classification")
                    Xte_local = Xte_local[te_mask]
                    yte_local = np.asarray([remap[int(v)] for v in yte_raw[te_mask]], dtype=np.int32)

                if len(Xtr_local) == 0 or len(Xte_local) == 0:
                    raise ValueError("Empty train/test split after label cleaning")
                model.fit(Xtr_local, ytr_local)
                pred = model.predict(Xte_local)

                if self.task_type == "regression":
                    r2 = float(r2_score(yte_local, pred))
                    mse = float(mean_squared_error(yte_local, pred))
                    mae = float(mean_absolute_error(yte_local, pred))
                    avg_score = float(np.mean([self._clip01(r2), 1.0 / (1.0 + mse), 1.0 / (1.0 + mae)]))
                    rows.append(
                        BenchmarkResult(
                            model=name,
                            status="ok",
                            avg_score=avg_score,
                            metrics={"r2": r2, "mse": mse, "mae": mae},
                        )
                    )
                else:
                    f1_macro = float(f1_score(yte_local, pred, average="macro", zero_division=0))
                    f1_micro = float(f1_score(yte_local, pred, average="micro", zero_division=0))
                    acc = float(accuracy_score(yte_local, pred))
                    # Compute AUROC when feasible (binary or OvR for multiclass)
                    auroc: float = float("nan")
                    try:
                        from sklearn.metrics import roc_auc_score

                        n_cls = len(np.unique(yte_local))
                        if n_cls == 2:
                            if hasattr(model, "predict_proba"):
                                prob = model.predict_proba(Xte_local)[:, 1]
                                auroc = float(roc_auc_score(yte_local, prob))
                        elif n_cls > 2:
                            if hasattr(model, "predict_proba"):
                                prob = model.predict_proba(Xte_local)
                                auroc = float(
                                    roc_auc_score(
                                        yte_local,
                                        prob,
                                        multi_class="ovr",
                                        average="macro",
                                    )
                                )
                    except Exception:
                        pass
                    avg_score = float(np.mean([self._clip01(f1_macro), self._clip01(f1_micro), self._clip01(acc)]))
                    metrics: Dict[str, float] = {
                        "f1_macro": f1_macro,
                        "f1_micro": f1_micro,
                        "accuracy": acc,
                    }
                    if np.isfinite(auroc):
                        metrics["auroc"] = auroc
                    rows.append(
                        BenchmarkResult(
                            model=name,
                            status="ok",
                            avg_score=avg_score,
                            metrics=metrics,
                        )
                    )
            except Exception as exc:
                rows.append(
                    BenchmarkResult(
                        model=name,
                        status="error",
                        avg_score=float("nan"),
                        metrics={},
                        error=str(exc),
                    )
                )

        detail_df = self._to_dataframe(rows)
        summary = self._build_summary(detail_df)
        return detail_df, summary

    def _build_models(self) -> List[Tuple[str, Any]]:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise SystemExit("BenchmarkEvaluator requires xgboost. Install with: python -m pip install xgboost") from exc

        if self.task_type == "regression":
            return [
                (
                    "xgboost",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=self.random_state,
                        n_jobs=-1,
                        objective="reg:squarederror",
                        verbosity=0,
                    ),
                ),
                (
                    "random_forest",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        random_state=self.random_state,
                    ),
                ),
            ]

        return [
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                    verbosity=0,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    random_state=self.random_state,
                ),
            ),
        ]

    def _prepare_classification_labels(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        max_bins: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.preprocessing import LabelEncoder

        y_train_num = pd.to_numeric(y_train, errors="coerce")
        y_test_num = pd.to_numeric(y_test, errors="coerce")

        train_is_num = np.isfinite(y_train_num).all()
        test_is_num = np.isfinite(y_test_num).all()

        if train_is_num and test_is_num:
            tr_unique = int(pd.Series(y_train_num).nunique(dropna=True))
            looks_discrete = np.allclose(y_train_num, np.round(y_train_num), atol=1e-9)
            if looks_discrete and tr_unique <= 100:
                y_tr = pd.Series(np.round(y_train_num).astype(np.int64)).astype(str)
                y_te = pd.Series(np.round(y_test_num).astype(np.int64)).astype(str)
            else:
                if tr_unique < 2:
                    raise ValueError("Classification labels have less than 2 unique values")
                bins = min(max_bins, tr_unique)
                print(
                    "[BenchmarkEvaluator] Classification labels appear continuous; "
                    f"auto-discretizing by quantile bins (q={bins})."
                )
                y_tr = pd.qcut(y_train_num, q=bins, duplicates="drop").astype(str)
                bin_edges = np.unique(np.quantile(y_train_num, np.linspace(0.0, 1.0, bins + 1)))
                y_te = pd.cut(
                    y_test_num,
                    bins=bin_edges.tolist(),
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str)
        else:
            y_tr = y_train.astype(str)
            y_te = y_test.astype(str)

        le = LabelEncoder()
        y_tr_arr = np.asarray(y_tr.astype(str), dtype=object)
        y_te_arr = np.asarray(y_te.astype(str), dtype=object)
        merged_labels = np.concatenate([y_tr_arr, y_te_arr], axis=0)
        le.fit(np.unique(merged_labels))
        return np.asarray(le.transform(y_tr_arr)), np.asarray(le.transform(y_te_arr))

    @staticmethod
    def _clip01(v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(min(1.0, max(0.0, v)))

    def _to_dataframe(self, rows: List[BenchmarkResult]) -> pd.DataFrame:
        out: List[Dict[str, Any]] = []
        for row in rows:
            rec: Dict[str, Any] = {
                "model": row.model,
                "status": row.status,
                "avg_score": row.avg_score,
                "error": row.error,
            }
            rec.update(row.metrics)
            out.append(rec)

        if self.task_type == "regression":
            cols = ["model", "status", "avg_score", "r2", "mse", "mae", "error"]
        else:
            cols = ["model", "status", "avg_score", "f1_macro", "f1_micro", "accuracy", "auroc", "error"]

        df = pd.DataFrame(out)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan if c != "error" else ""
        return df[cols]

    def _build_summary(self, detail_df: pd.DataFrame) -> Dict[str, Any]:
        ok = detail_df[detail_df["status"] == "ok"].copy()
        if ok.empty:
            return {
                "task_type": self.task_type,
                "n_models_ok": 0,
                "avg_score_mean": float("nan"),
                "best_model": "",
                "best_model_avg_score": float("nan"),
            }

        best_idx = ok["avg_score"].astype(float).idxmax()
        best_model = str(ok.loc[best_idx, "model"])
        best_score = float(ok.loc[best_idx, "avg_score"])

        summary: Dict[str, Any] = {
            "task_type": self.task_type,
            "n_models_ok": int(len(ok)),
            "avg_score_mean": float(ok["avg_score"].astype(float).mean()),
            "best_model": best_model,
            "best_model_avg_score": best_score,
        }

        if self.task_type == "regression":
            summary.update(
                {
                    "mean_r2": float(ok["r2"].astype(float).mean()),
                    "mean_mse": float(ok["mse"].astype(float).mean()),
                    "mean_mae": float(ok["mae"].astype(float).mean()),
                }
            )
        else:
            summary.update(
                {
                    "mean_f1_macro": float(ok["f1_macro"].astype(float).mean()),
                    "mean_f1_micro": float(ok["f1_micro"].astype(float).mean()),
                    "mean_accuracy": float(ok["accuracy"].astype(float).mean()),
                }
            )
            # Add mean AUROC if present
            if "auroc" in ok.columns:
                auroc_vals = pd.to_numeric(ok["auroc"], errors="coerce")
                if auroc_vals.notna().any():
                    summary["mean_auroc"] = float(auroc_vals.mean())

        return summary
