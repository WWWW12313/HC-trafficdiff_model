"""
plot_v7_experiment.py

Generate v7 experiment figures and a markdown experiment log.
Outputs:
- loss curve
- sparse target metric heatmap
- confusion matrices (TSTR: train on synthetic, test on real)
- markdown log referencing generated figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ATOMIC_TARGETS = [
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
]


def _feature_columns(df: pd.DataFrame, target: str, max_cat_card: int = 500) -> List[str]:
    drop_cols = set(ATOMIC_TARGETS + ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", target])
    cols: List[str] = []
    for c in df.columns:
        if c in drop_cols:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
            continue
        nunique = s.fillna("<NA>").astype(str).nunique(dropna=False)
        if nunique <= max_cat_card:
            cols.append(c)
    return cols


def _encode_ordinal(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_train = np.zeros((len(train_df), len(cols)), dtype=np.float32)
    x_test = np.zeros((len(test_df), len(cols)), dtype=np.float32)

    for j, c in enumerate(cols):
        tr = train_df[c]
        te = test_df[c]
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            x_train[:, j] = pd.to_numeric(tr, errors="coerce").fillna(0.0).astype(float).to_numpy()
            x_test[:, j] = pd.to_numeric(te, errors="coerce").fillna(0.0).astype(float).to_numpy()
        else:
            trs = tr.fillna("<NA>").astype(str)
            tes = te.fillna("<NA>").astype(str)
            cats = pd.Index(pd.concat([trs, tes], axis=0).unique())
            c2i = {k: i for i, k in enumerate(cats)}
            x_train[:, j] = trs.map(c2i).astype(float).to_numpy()
            x_test[:, j] = tes.map(c2i).astype(float).to_numpy()

    return x_train, x_test


def _safe_train_test(real_sub: pd.DataFrame, target: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    cols = _feature_columns(real_sub, target)
    y_real = pd.to_numeric(real_sub[target], errors="coerce").fillna(0).astype(int)
    vc = y_real.value_counts()
    use_stratify = (len(vc) > 1) and (vc.min() >= 2)

    X_train_real_df, X_test_real_df, y_train_real, y_test_real = train_test_split(
        real_sub[cols],
        y_real,
        test_size=0.3,
        random_state=seed,
        stratify=y_real if use_stratify else None,
    )
    return X_train_real_df, X_test_real_df, y_train_real, y_test_real, cols


def plot_loss(loss_csv: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(loss_csv)
    fig, ax = plt.subplots(figsize=(9, 5))
    if "loss" in df.columns:
        ax.plot(df["step"], df["loss"], label="total", linewidth=2)
    if "loss_multi" in df.columns:
        ax.plot(df["step"], df["loss_multi"], label="multinomial", linewidth=1.6)
    if "loss_gauss" in df.columns:
        ax.plot(df["step"], df["loss_gauss"], label="gaussian", linewidth=1.6)
    ax.set_title("V7 Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_tstr_heatmap(eval_json: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    names = []
    for item in eval_json.get("sparse_tstr", []):
        if "error" in item:
            continue
        rows.append([
            float(item.get("tstr_macro_f1", 0.0)),
            float(item.get("real_macro_f1", 0.0)),
            float(item.get("fidelity_ratio", 0.0)),
            float(item.get("tstr_accuracy", 0.0)),
        ])
        names.append(str(item.get("target", "unknown")).replace("NUMBER OF ", "").replace(" INJURED", "_INJ").replace(" KILLED", "_KIL"))

    if not rows:
        return

    mat = np.array(rows, dtype=np.float32)
    col_names = ["TSTR_F1", "REAL_F1", "Fidelity", "TSTR_Acc"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(col_names)), labels=col_names)
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_title("Sparse Target Metrics Heatmap")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(real_df: pd.DataFrame, syn_df: pd.DataFrame, out_png: Path, seed: int = 42) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_sub = real_df.sample(n=min(len(real_df), 50000), random_state=seed).copy()
    syn_sub = syn_df.sample(n=min(len(syn_df), 50000), random_state=seed).copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, target in zip(axes, ATOMIC_TARGETS):
        if target not in real_sub.columns or target not in syn_sub.columns:
            ax.set_title(f"{target}\nmissing")
            ax.axis("off")
            continue

        try:
            _, x_test_real_df, _, y_test_real, cols = _safe_train_test(real_sub, target, seed)
            y_syn = pd.to_numeric(syn_sub[target], errors="coerce").fillna(0).astype(int)
            x_syn, x_test = _encode_ordinal(syn_sub[cols], x_test_real_df, cols)

            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
            clf.fit(x_syn, y_syn)
            pred = clf.predict(x_test)

            labels = np.unique(np.concatenate([y_test_real.to_numpy(), pred]))
            cm = confusion_matrix(y_test_real, pred, labels=labels)
            cm = cm.astype(np.float64)
            row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
            cm_norm = cm / row_sum

            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_title(target.replace("NUMBER OF ", "").replace(" INJURED", "_INJ").replace(" KILLED", "_KIL"), fontsize=9)
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")

            tick_labels = [str(int(v)) for v in labels]
            if len(tick_labels) <= 8:
                ax.set_xticks(np.arange(len(labels)), labels=tick_labels, fontsize=7)
                ax.set_yticks(np.arange(len(labels)), labels=tick_labels, fontsize=7)
            else:
                idx = np.linspace(0, len(labels) - 1, num=8, dtype=int)
                ax.set_xticks(idx, labels=[tick_labels[i] for i in idx], fontsize=7)
                ax.set_yticks(idx, labels=[tick_labels[i] for i in idx], fontsize=7)

            # annotate only for small matrices for readability
            if cm_norm.shape[0] <= 6:
                for i in range(cm_norm.shape[0]):
                    for j in range(cm_norm.shape[1]):
                        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=6)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        except Exception as e:
            ax.set_title(f"{target}\nerror")
            ax.text(0.5, 0.5, str(e)[:90], ha="center", va="center", fontsize=7, wrap=True)
            ax.axis("off")

    fig.suptitle("TSTR Confusion Matrices (train=syn, test=real)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_experiment_log(
    out_md: Path,
    model_dir: Path,
    eval_json: dict,
    fig_dir: Path,
) -> None:
    sparse = [x for x in eval_json.get("sparse_tstr", []) if "error" not in x]
    lines: List[str] = []

    lines.append("# EXPERIMENT LOG - v7 Two-Stage CausalDDPM")
    lines.append("")
    lines.append("## Phase A: Code Fixes")
    lines.append("- Fixed static type issues in `scripts/evaluate_v7.py` by explicit numpy conversion before arithmetic.")
    lines.append("- Fixed static type issues in `scripts/sample_two_stage_v7.py` for optional pandas series, context encoding ndarray conversion, and diffusion internal API typing.")
    lines.append("")

    lines.append("## Phase B: Main Artifacts")
    lines.append(f"- Model dir: `{model_dir.as_posix()}`")
    lines.append(f"- Synthetic csv: `{(model_dir / 'synthetic_2017_v7.csv').as_posix()}`")
    lines.append(f"- Eval json: `{(model_dir / 'eval_v7.json').as_posix()}`")
    lines.append(f"- Figure dir: `{fig_dir.as_posix()}`")
    lines.append("")

    lines.append("## Phase C: Key Metrics")
    lines.append(f"- logic_violation_rate: `{eval_json.get('logic_violation_rate', 'NA')}`")
    marg = eval_json.get("marginal_similarity", {})
    lines.append(f"- numeric_drift_avg: `{marg.get('numeric_drift_avg', 'NA')}`")
    lines.append(f"- categorical_tv_avg: `{marg.get('categorical_tv_avg', 'NA')}`")
    lines.append("")

    lines.append("### Sparse TSTR Summary")
    if sparse:
        lines.append("| target | tstr_macro_f1 | real_macro_f1 | fidelity_ratio | tstr_accuracy |")
        lines.append("|---|---:|---:|---:|---:|")
        for s in sparse:
            lines.append(
                f"| {s.get('target', 'NA')} | {float(s.get('tstr_macro_f1', 0.0)):.4f} | {float(s.get('real_macro_f1', 0.0)):.4f} | {float(s.get('fidelity_ratio', 0.0)):.4f} | {float(s.get('tstr_accuracy', 0.0)):.4f} |"
            )
    else:
        lines.append("- No valid sparse_tstr rows found.")
    lines.append("")

    lines.append("## Phase D: Figures")
    lines.append(f"- Loss curve: `{(fig_dir / 'loss_curve_v7.png').as_posix()}`")
    lines.append(f"- Metric heatmap: `{(fig_dir / 'sparse_tstr_heatmap_v7.png').as_posix()}`")
    lines.append(f"- Confusion matrices: `{(fig_dir / 'confusion_matrix_tstr_v7.png').as_posix()}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Figures are saved separately in `figures/` as required.")
    lines.append("- Log follows the same phased style as the v6 experiment log.")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v7 figures and markdown experiment log")
    parser.add_argument("--model_dir", type=str, default="exp/nyc_crash_v7/causal_m4_v7")
    parser.add_argument("--real_csv", type=str, default="nyc_2017_pristine_v8.csv")
    parser.add_argument("--syn_csv", type=str, default="exp/nyc_crash_v7/causal_m4_v7/synthetic_2017_v7.csv")
    parser.add_argument("--eval_json", type=str, default="exp/nyc_crash_v7/causal_m4_v7/eval_v7.json")
    parser.add_argument("--out_log", type=str, default="EXPERIMENT_LOG_v7.md")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fig_dir = model_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_path = Path(args.eval_json)
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))

    loss_csv = model_dir / "loss_v7.csv"
    if loss_csv.exists():
        plot_loss(loss_csv, fig_dir / "loss_curve_v7.png")

    plot_tstr_heatmap(eval_data, fig_dir / "sparse_tstr_heatmap_v7.png")

    real_df = pd.read_csv(args.real_csv)
    syn_df = pd.read_csv(args.syn_csv)
    plot_confusion_matrices(real_df, syn_df, fig_dir / "confusion_matrix_tstr_v7.png")

    out_md = Path(args.out_log)
    write_experiment_log(out_md, model_dir, eval_data, fig_dir)

    print("=== v7 plotting/report done ===")
    print(f"figures: {fig_dir.as_posix()}")
    print(f"log: {out_md.as_posix()}")


if __name__ == "__main__":
    main()
