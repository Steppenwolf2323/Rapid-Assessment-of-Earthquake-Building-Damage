"""
plot_clustering.py — RQ2 figures

Generates:
    fig1_delta_f1_by_radius.png      ΔF1 vs homogeneity threshold, one line per radius
    fig2_delta_f1_by_model.png       ΔF1 vs radius at best homogeneity threshold
    fig3_f1_comparison.png           Baseline vs best clustering F1 bar chart
    fig4_neighbourhood_size.png      Mean neighbourhood size vs radius
    fig5_summary_table.png           Full results table

Usage:
    cd 4_spatial_context
    python plot_clustering.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.use("Agg")

RESULTS_FILE = Path(__file__).parent / "results" / "clustering_results.json"
FIGURES_DIR  = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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


# ─── Figure 1: ΔF1 vs homogeneity threshold, one panel per model ─────────────

def plot_delta_f1_by_radius(results):
    """
    For each model: shows ΔF1 at each homogeneity threshold,
    with one line per radius. Answers: at what threshold does each
    radius start helping or hurting?
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, (model_id, res) in zip(axes, results.items()):
        radius_results = res["radius_results"]
        n_radii        = len(radius_results)
        palette        = cm.get_cmap("YlOrRd", n_radii + 2)

        for i, rr in enumerate(radius_results):
            thresholds = [t["homogeneity_threshold"] for t in rr["threshold_results"]]
            deltas     = [t["delta_f1"]              for t in rr["threshold_results"]]
            ax.plot(thresholds, deltas,
                    label=f"{rr['radius_px']}px ({rr['radius_m']:.0f}m)",
                    color=palette(i + 1), linewidth=2,
                    marker="o", markersize=4)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
        ax.fill_between([0, 0.55],  0,  0.1, alpha=0.05, color="green")
        ax.fill_between([0, 0.55], -0.1, 0,  alpha=0.05, color="red")
        ax.set_title(LABELS[model_id], fontsize=10)
        ax.set_xlabel("Homogeneity threshold")
        if ax == axes[0]:
            ax.set_ylabel("ΔF1 (clustered − baseline)")
        ax.legend(fontsize=8, title="Radius")
        ax.set_xlim(0.02, 0.53)

    fig.suptitle("ΔF1 vs homogeneity threshold at different spatial radii\n"
                 "Green = improvement, Red = degradation", fontsize=12)
    fig.tight_layout()
    path = FIGURES_DIR / "fig1_delta_f1_by_radius.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 2: ΔF1 vs radius, all models ─────────────────────────────────────

def plot_delta_f1_by_model(results):
    """
    Best ΔF1 at each radius for each model.
    Answers: does a larger radius help or hurt generalisation?
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id, res in results.items():
        radii_m   = [rr["radius_m"]      for rr in res["radius_results"]]
        best_deltas = [rr["best_delta_f1"] for rr in res["radius_results"]]
        ax.plot(radii_m, best_deltas,
                label=LABELS[model_id], color=COLORS[model_id],
                linewidth=2, marker="o", markersize=6)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    ax.fill_between([0, 900],  0,  0.05, alpha=0.05, color="green",
                    label="Improvement zone")
    ax.fill_between([0, 900], -0.05, 0,  alpha=0.05, color="red",
                    label="Degradation zone")
    ax.set_xlabel("Radius (metres)")
    ax.set_ylabel("Best ΔF1 at optimal homogeneity threshold")
    ax.set_title("Best F1 improvement vs spatial radius\n"
                 "(each point = best homogeneity threshold for that radius)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "fig2_delta_f1_by_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 3: Baseline vs best F1 bar chart ─────────────────────────────────

def plot_f1_comparison(results):
    models      = list(results.keys())
    baseline_f1 = [results[m]["baseline"]["f1"] for m in models]
    best_f1     = [results[m]["best_f1"]        for m in models]
    best_r      = [results[m]["best_radius_m"]  for m in models]
    best_t      = [results[m]["best_hom_threshold"] for m in models]

    x     = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, baseline_f1, width,
                   label="Baseline (no clustering)",
                   color=[COLORS[m] for m in models], alpha=0.4)
    bars2 = ax.bar(x + width/2, best_f1, width,
                   label="Best adaptive clustering",
                   color=[COLORS[m] for m in models], alpha=0.9)

    for bar, val in zip(bars1, baseline_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val, r, t in zip(bars2, best_f1, best_r, best_t):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{val:.4f}\n({r:.0f}m, t={t})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
    ax.set_ylabel("F1 — damaged class")
    ax.set_title("Baseline F1 vs best adaptive clustering F1")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "fig3_f1_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 4: Neighbourhood size vs radius ──────────────────────────────────

def plot_neighbourhood_size(results):
    """
    Shows mean neighbourhood size at each radius.
    Provides context for interpreting the clustering results —
    how many buildings are in each spatial group?
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use first model (neighbourhood sizes are the same for all models)
    first_model = list(results.values())[0]
    radii_px    = [rr["radius_px"]            for rr in first_model["radius_results"]]
    radii_m     = [rr["radius_m"]             for rr in first_model["radius_results"]]
    mean_sizes  = [rr["neighbourhood_mean"]   for rr in first_model["radius_results"]]

    ax.bar(radii_m, mean_sizes, width=40,
           color="#534AB7", alpha=0.7)
    for r, s in zip(radii_m, mean_sizes):
        ax.text(r, s + 0.5, f"{s:.1f}", ha="center", va="bottom", fontsize=10)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(radii_m)
    ax2.set_xticklabels([f"{r}px" for r in radii_px], fontsize=9)
    ax2.set_xlabel("Radius (pixels)", fontsize=10)

    ax.set_xlabel("Radius (metres)")
    ax.set_ylabel("Mean neighbourhood size (buildings)")
    ax.set_title("Mean number of neighbours per building at each radius")
    fig.tight_layout()
    path = FIGURES_DIR / "fig4_neighbourhood_size.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Figure 5: Summary table ──────────────────────────────────────────────────

def plot_summary_table(results):
    rows = []
    for model_id, res in results.items():
        b = res["baseline"]
        rows.append([
            LABELS[model_id],
            f"{b['f1']:.4f}",
            f"{b['recall']:.4f}",
            f"{b['precision']:.4f}",
            f"{res['best_radius_px']}px / {res['best_radius_m']:.0f}m",
            f"{res['best_hom_threshold']}",
            f"{res['best_f1']:.4f}",
            f"{res['best_delta_f1']:+.4f}",
        ])

    col_labels = [
        "Model",
        "Baseline F1", "Baseline Recall", "Baseline Precision",
        "Best radius", "Best hom. threshold",
        "Best clustered F1", "ΔF1",
    ]

    fig, ax = plt.subplots(figsize=(17, 2.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#EEEDFE")
        table[0, j].set_text_props(weight="bold")

    for i, (model_id, res) in enumerate(results.items(), start=1):
        table[i, 0].set_facecolor(COLORS[model_id] + "33")
        delta = res["best_delta_f1"]
        table[i, -1].set_facecolor("#d4edda" if delta >= 0 else "#f8d7da")

    ax.set_title("RQ2: Adaptive spatial voting — summary of best results",
                 fontsize=12, pad=20)
    fig.tight_layout()
    path = FIGURES_DIR / "fig5_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run evaluate_clustering.py first.")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    print(f"Loaded results for models: {list(results.keys())}")
    print(f"Generating figures in: {FIGURES_DIR}\n")

    plot_delta_f1_by_radius(results)
    plot_delta_f1_by_model(results)
    plot_f1_comparison(results)
    plot_neighbourhood_size(results)
    plot_summary_table(results)

    print(f"\nDone. All figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
