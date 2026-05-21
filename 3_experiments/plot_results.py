"""
plot_results.py — Generate all thesis figures and tables

Place this file in:
    3_experiments/

Reads:
    2_models_trainings/model_A/outputs/history.json
    2_models_trainings/model_B/outputs/history.json
    2_models_trainings/model_C/outputs/history.json
    3_experiments/model_A/evaluation_xbd.json
    3_experiments/model_B/evaluation_xbd.json
    3_experiments/model_C/evaluation_xbd.json

Generates in 3_experiments/figures/:
    fig1_loss_curves.png       Training & validation loss per model
    fig2_val_f1.png            Validation F1 over epochs + xBD test F1
    fig3_val_auc.png           Validation AUC over epochs + xBD test AUC
    fig4_summary_table.png     Val vs xBD test results table
    fig5_per_disaster.png      Per-disaster F1 and Recall bar charts
    fig6_roc_curves.png        ROC curves on xBD test set

Usage:
    cd 3_experiments
    python plot_results.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

matplotlib.use("Agg")   # non-interactive backend, safe for all environments

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")

HISTORY_PATHS = {
    "A": BASE / "2_models_trainings" / "model_A" / "outputs" / "history.json",
    "B": BASE / "2_models_trainings" / "model_B" / "outputs" / "history.json",
    "C": BASE / "2_models_trainings" / "model_C" / "outputs" / "history.json",
}
EVAL_PATHS = {
    "A": BASE / "3_experiments" / "model_A" / "evaluation_xbd.json",
    "B": BASE / "3_experiments" / "model_B" / "evaluation_xbd.json",
    "C": BASE / "3_experiments" / "model_C" / "evaluation_xbd.json",
}
FIGURES_DIR = BASE / "3_experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
COLORS = {"A": "#534AB7", "B": "#0F6E56", "C": "#993C1D"}
LABELS = {
    "A": "Model A — End-to-end",
    "B": "Model B — Human-guided",
    "C": "Model C — Physics-guided",
}
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_history(path: Path) -> dict:
    with open(path) as f:
        h = json.load(f)

    epochs     = []
    train_loss = []
    val_loss   = []
    train_f1   = []
    val_f1     = []
    val_auc    = []
    offset     = 0

    for stage_key in sorted(h.keys()):
        for entry in h[stage_key]:
            epochs.append(offset + entry["epoch"])
            train_loss.append(entry["train"]["loss"])
            val_loss.append(entry["val"]["loss"])
            train_f1.append(entry["train"]["f1"])
            val_f1.append(entry["val"]["f1"])
            val_auc.append(entry["val"].get("auc", float("nan")))
        offset += len(h[stage_key])

    return {
        "epochs":     epochs,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "train_f1":   train_f1,
        "val_f1":     val_f1,
        "val_auc":    val_auc,
        "n_stages":   len(h),
        "stage_sizes": [len(h[k]) for k in sorted(h.keys())],
    }


def load_eval(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Figure 1: Loss curves ────────────────────────────────────────────────────

def plot_loss_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, (model_id, h) in zip(axes, histories.items()):
        ax.plot(h["epochs"], h["train_loss"],
                label="Train", color=COLORS[model_id], linewidth=2)
        ax.plot(h["epochs"], h["val_loss"],
                label="Validation", color=COLORS[model_id],
                linewidth=2, linestyle="--")
        ax.set_title(LABELS[model_id], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # Mark stage 1 / stage 2 boundary for A and B
        if model_id in ("A", "B") and h["n_stages"] == 2:
            boundary = h["stage_sizes"][0] + 0.5
            ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.6)
            ax.text(boundary + 0.3, ax.get_ylim()[1] * 0.97,
                    "Stage 2", fontsize=8, color="gray", va="top")

    fig.suptitle("Training and validation loss curves", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / "fig1_loss_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 2: Validation F1 ─────────────────────────────────────────────────

def plot_val_f1(histories, evaluations):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id, h in histories.items():
        ax.plot(h["epochs"], h["val_f1"],
                label=LABELS[model_id], color=COLORS[model_id], linewidth=2)

        if model_id in evaluations:
            test_f1 = evaluations[model_id]["metrics"]["f1"]
            ax.axhline(y=test_f1, color=COLORS[model_id],
                       linestyle=":", alpha=0.7, linewidth=1.5)
            ax.text(max(h["epochs"]) * 1.01, test_f1,
                    f"xBD {test_f1:.3f}", fontsize=9,
                    color=COLORS[model_id], va="center")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 — damaged class")
    ax.set_title("Validation F1 over training\n(dotted = xBD test F1)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "fig2_val_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 3: Validation AUC ────────────────────────────────────────────────

def plot_val_auc(histories, evaluations):
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id, h in histories.items():
        ax.plot(h["epochs"], h["val_auc"],
                label=LABELS[model_id], color=COLORS[model_id], linewidth=2)

        if model_id in evaluations:
            test_auc = evaluations[model_id]["metrics"]["auc"]
            ax.axhline(y=test_auc, color=COLORS[model_id],
                       linestyle=":", alpha=0.7, linewidth=1.5)
            ax.text(max(h["epochs"]) * 1.01, test_auc,
                    f"xBD {test_auc:.3f}", fontsize=9,
                    color=COLORS[model_id], va="center")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Validation AUC over training\n(dotted = xBD test AUC)")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "fig3_val_auc.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 4: Summary table ──────────────────────────────────────────────────

def plot_summary_table(histories, evaluations):
    rows = []
    for model_id in ("A", "B", "C"):
        if model_id not in histories or model_id not in evaluations:
            continue
        h  = histories[model_id]
        ev = evaluations[model_id]["metrics"]

        rows.append([
            LABELS[model_id],
            f"{max(h['val_f1']):.3f}",
            f"{max(h['val_auc']):.3f}",
            f"{ev['f1']:.3f}",
            f"{ev['recall']:.3f}",
            f"{ev['precision']:.3f}",
            f"{ev['auc']:.3f}",
            f"{ev['tp']} / {ev['fn']}",
        ])

    col_labels = [
        "Model",
        "Best val F1", "Best val AUC",
        "xBD F1", "xBD Recall", "xBD Precision", "xBD AUC",
        "TP / FN",
    ]

    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText  = rows,
        colLabels = col_labels,
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#EEEDFE")
        table[0, j].set_text_props(weight="bold")

    for i, model_id in enumerate(("A", "B", "C"), start=1):
        if i <= len(rows):
            table[i, 0].set_facecolor(COLORS[model_id] + "33")

    ax.set_title("Results summary: QQB validation vs xBD test",
                 fontsize=12, pad=20)
    fig.tight_layout()
    path = FIGURES_DIR / "fig4_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 5: Per-disaster breakdown ────────────────────────────────────────

def plot_per_disaster(evaluations):
    disasters = sorted({
        d for ev in evaluations.values()
        for d in ev["per_disaster"].keys()
    })

    x     = np.arange(len(disasters))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax in zip(["f1", "recall"], axes):
        for i, (model_id, ev) in enumerate(evaluations.items()):
            vals = [
                ev["per_disaster"].get(d, {}).get(metric, 0)
                for d in disasters
            ]
            bars = ax.bar(
                x + (i - 1) * width, vals, width,
                label=LABELS[model_id],
                color=COLORS[model_id], alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("-", "\n") for d in disasters], fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"xBD test {metric.upper()} per disaster event")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)

    fig.tight_layout()
    path = FIGURES_DIR / "fig5_per_disaster.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 6: ROC curves ────────────────────────────────────────────────────

def plot_roc_curves(evaluations):
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_id, ev in evaluations.items():
        labels = [s["label"] for s in ev["per_sample"]]
        probs  = [s["prob"]  for s in ev["per_sample"]]
        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc     = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLORS[model_id], linewidth=2,
                    label=f"{LABELS[model_id]} (AUC = {roc_auc:.3f})")
        except Exception as e:
            print(f"  ROC curve failed for Model {model_id}: {e}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — xBD test set")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "fig6_roc_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...\n")

    histories = {}
    for m, p in HISTORY_PATHS.items():
        if p.exists():
            histories[m] = load_history(p)
            print(f"  Loaded history: Model {m} ({len(histories[m]['epochs'])} epochs)")
        else:
            print(f"  WARNING: history not found for Model {m}: {p}")

    evaluations = {}
    for m, p in EVAL_PATHS.items():
        if p.exists():
            evaluations[m] = load_eval(p)
            print(f"  Loaded evaluation: Model {m}")
        else:
            print(f"  WARNING: evaluation not found for Model {m}: {p}")
            print(f"           Run evaluate_{m}.py first.")

    if not histories and not evaluations:
        print("\nNo data found. Exiting.")
        return

    print(f"\nGenerating figures in: {FIGURES_DIR}\n")

    if histories:
        plot_loss_curves(histories)
        plot_val_f1(histories, evaluations)
        plot_val_auc(histories, evaluations)

    if histories and evaluations:
        plot_summary_table(histories, evaluations)

    if evaluations:
        plot_per_disaster(evaluations)
        plot_roc_curves(evaluations)

    print(f"\nDone. All figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
