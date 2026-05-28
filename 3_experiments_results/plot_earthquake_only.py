"""
plot_earthquake_only.py — Thesis figures using earthquake-only xBD test set
Includes mexico-earthquake only (120 samples).

Place in: 3_experiments/
Reads:    evaluation_xbd_earthquake_only.json from each model folder
Saves:    3_experiments/figures_earthquake_only/

Usage:
    cd 3_experiments
    python plot_earthquake_only.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

matplotlib.use("Agg")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")

HISTORY_PATHS = {
    "A": BASE / "2_models_trainings" / "model_A" / "outputs" / "history.json",
    "B": BASE / "2_models_trainings" / "model_B" / "outputs" / "history.json",
    "C": BASE / "2_models_trainings" / "model_C" / "outputs" / "history.json",
}
EVAL_PATHS = {
    "A": BASE / "3_experiments" / "model_A" / "evaluation_xbd_earthquake_only.json",
    "B": BASE / "3_experiments" / "model_B" / "evaluation_xbd_earthquake_only.json",
    "C": BASE / "3_experiments" / "model_C" / "evaluation_xbd_earthquake_only.json",
}
FIGURES_DIR = BASE / "3_experiments" / "figures_earthquake_only"
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
    """
    Handles two history formats:
        Models A/B: {"stage1": [...], "stage2": [...]}
        Model C:    [{...}, {...}, ...]
    """
    with open(path) as f:
        h = json.load(f)

    if isinstance(h, list):
        entries     = h
        n_stages    = 1
        stage_sizes = [len(h)]
    else:
        entries     = []
        stage_sizes = []
        for stage_key in sorted(h.keys()):
            entries.extend(h[stage_key])
            stage_sizes.append(len(h[stage_key]))
        n_stages = len(h)

    epochs     = list(range(1, len(entries) + 1))
    train_loss = [e["train"]["loss"] for e in entries]
    val_loss   = [e["val"]["loss"]   for e in entries]
    train_f1   = [e["train"]["f1"]   for e in entries]
    val_f1     = [e["val"]["f1"]     for e in entries]
    val_auc    = [e["val"].get("auc", float("nan")) for e in entries]

    return {
        "epochs":      epochs,
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "train_f1":    train_f1,
        "val_f1":      val_f1,
        "val_auc":     val_auc,
        "n_stages":    n_stages,
        "stage_sizes": stage_sizes,
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
        ax.set_title(LABELS[model_id])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        if model_id in ("A", "B") and h["n_stages"] == 2:
            boundary = h["stage_sizes"][0] + 0.5
            ax.axvline(x=boundary, color="gray", linestyle=":", alpha=0.6)
            ax.text(boundary + 0.3, ax.get_ylim()[1] * 0.97,
                    "Stage 2", fontsize=8, color="gray", va="top")
    fig.suptitle("Training and validation loss — all models", fontsize=13)
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
    ax.set_title("Validation F1 during training\n(dotted = xBD test F1, earthquake only)")
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
    ax.set_title("Validation AUC during training\n(dotted = xBD test AUC, earthquake only)")
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
        "Model", "Best val F1", "Best val AUC",
        "xBD F1", "xBD Recall", "xBD Precision", "xBD AUC", "TP / FN",
    ]
    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#EEEDFE")
        table[0, j].set_text_props(weight="bold")
    for i, model_id in enumerate(("A", "B", "C"), start=1):
        if i <= len(rows):
            table[i, 0].set_facecolor(COLORS[model_id] + "33")
    ax.set_title("Results: QQB validation vs xBD test (earthquake only — Mexico 2017)",
                 fontsize=12, pad=20)
    fig.tight_layout()
    path = FIGURES_DIR / "fig4_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 5: Confusion matrix per model ────────────────────────────────────

def plot_confusion_matrices(evaluations):
    """
    With only one disaster (Mexico earthquake), per-disaster bar chart
    is not meaningful. Instead show confusion matrices per model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (model_id, ev) in zip(axes, evaluations.items()):
        m   = ev["metrics"]
        cm  = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        im  = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred intact", "Pred damaged"])
        ax.set_yticklabels(["Actual intact", "Actual damaged"])
        ax.set_title(LABELS[model_id], fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Confusion matrices — xBD test (earthquake only)", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / "fig5_confusion_matrices.png"
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
            print(f"  ROC failed for Model {model_id}: {e}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — xBD test set (earthquake only)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "fig6_roc_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...\n")
    histories, evaluations = {}, {}

    for m, p in HISTORY_PATHS.items():
        if p.exists():
            histories[m] = load_history(p)
            print(f"  History loaded: Model {m} ({len(histories[m]['epochs'])} epochs)")
        else:
            print(f"  WARNING: history not found for Model {m}")

    for m, p in EVAL_PATHS.items():
        if p.exists():
            evaluations[m] = load_eval(p)
            print(f"  Evaluation loaded: Model {m}")
        else:
            print(f"  WARNING: evaluation not found for Model {m}")
            print(f"           Run evaluate_{m}.py with xbd_test_earthquake_only.csv first")

    print(f"\nGenerating figures in: {FIGURES_DIR}\n")
    if histories:
        plot_loss_curves(histories)
        plot_val_f1(histories, evaluations)
        plot_val_auc(histories, evaluations)
    if histories and evaluations:
        plot_summary_table(histories, evaluations)
    if evaluations:
        plot_confusion_matrices(evaluations)
        plot_roc_curves(evaluations)

    print(f"\nDone. Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
