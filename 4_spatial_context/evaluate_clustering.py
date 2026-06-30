"""
evaluate_clustering.py — RQ2: Radius-based Adaptive Spatial Voting

Sweeps over:
    - radius_px:              [50, 100, 150, 200, 300] pixels
    - homogeneity_threshold:  [0.05, 0.10, ..., 0.50]

For each combination, applies adaptive voting and records metrics.

GSD = 2.65 m/px → radius in real-world metres:
    50px  → 132m   (half a city block)
    100px → 265m   (one city block)
    150px → 397m   (1.5 blocks)
    200px → 530m   (2 blocks)
    300px → 795m   (3 blocks)

Saves: results/clustering_results.json

Usage:
    cd 4_spatial_context
    python evaluate_clustering.py
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
from clustering_models import (
    load_predictions,
    load_centroids,
    build_radius_neighbourhoods,
    apply_adaptive_voting,
    compute_metrics,
)


BASE = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")

EVAL_PATHS = {
    "A": BASE / "3_experiments" / "model_A" / "evaluation_xbd.json",
    "B": BASE / "3_experiments" / "model_B" / "evaluation_xbd.json",
    "C": BASE / "3_experiments" / "model_C" / "evaluation_xbd.json",
}

CENTROIDS_CSV = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_building_centroids.csv"
OUTPUT_DIR    = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GSD_M_PER_PX = 2.65   # metres per pixel for xBD Mexico earthquake

# Radii to test (pixels)
RADII_PX = [50, 100, 150, 200, 300]

HOMOGENEITY_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.55, 0.05)]


def evaluate_model(model_id: str, eval_path: Path, uid_to_data: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Model {model_id}")
    print(f"{'─'*60}")

    if not eval_path.exists():
        print(f"  WARNING: {eval_path} not found.")
        return {}

    data           = load_predictions(eval_path)
    paths          = data["paths"]
    labels         = data["labels"]
    probs          = data["probs"]
    preds          = data["preds"]
    best_threshold = data["best_threshold"]

    print(f"  Samples: {len(paths)} "
          f"(damaged: {sum(labels)}, intact: {len(labels)-sum(labels)})")
    print(f"  Best threshold from evaluation: {best_threshold}")

    # 
    baseline = compute_metrics(labels, preds, probs)
    print(f"\n  Baseline (no clustering):")
    print(f"    F1={baseline['f1']}  Recall={baseline['recall']}  "
          f"Precision={baseline['precision']}  AUC={baseline['auc']}")

    radius_results = []

    for radius_px in RADII_PX:
        real_world_m = radius_px * GSD_M_PER_PX
        print(f"\n  Radius = {radius_px}px (~{real_world_m:.0f}m real world)")

        # Build neighbourhoods for this radius
        neighbourhoods, sizes = build_radius_neighbourhoods(
            paths, uid_to_data, radius_px
        )
        print(f"    Neighbourhood sizes: min={min(sizes)}, "
              f"max={max(sizes)}, mean={np.mean(sizes):.1f}, "
              f"median={np.median(sizes):.1f}")

        threshold_results = []
        best_f1_for_radius = 0.0
        best_t_for_radius  = HOMOGENEITY_THRESHOLDS[0]

        print(f"    {'Threshold':>10}  {'F1':>8}  {'Recall':>8}  "
              f"{'Precision':>10}  {'ΔF1':>8}  {'Voted%':>8}")
        print(f"    {'-'*58}")

        for t in HOMOGENEITY_THRESHOLDS:
            new_preds, vote_applied, cluster_stds = apply_adaptive_voting(
                probs, preds, neighbourhoods, t, best_threshold
            )
            metrics   = compute_metrics(labels, new_preds, probs)
            delta_f1  = metrics["f1"] - baseline["f1"]
            pct_voted = 100 * sum(vote_applied) / len(vote_applied)

            print(f"    {t:>10.2f}  {metrics['f1']:>8.4f}  "
                  f"{metrics['recall']:>8.4f}  "
                  f"{metrics['precision']:>10.4f}  "
                  f"{delta_f1:>+8.4f}  {pct_voted:>7.1f}%")

            threshold_results.append({
                "homogeneity_threshold": t,
                "metrics":   metrics,
                "delta_f1":  round(delta_f1, 4),
                "pct_voted": round(pct_voted, 2),
            })

            if metrics["f1"] > best_f1_for_radius:
                best_f1_for_radius = metrics["f1"]
                best_t_for_radius  = t

        radius_results.append({
            "radius_px":          radius_px,
            "radius_m":           round(real_world_m, 1),
            "neighbourhood_mean": round(float(np.mean(sizes)), 1),
            "best_threshold":     best_t_for_radius,
            "best_f1":            best_f1_for_radius,
            "best_delta_f1":      round(best_f1_for_radius - baseline["f1"], 4),
            "threshold_results":  threshold_results,
        })

    best_overall = max(
        radius_results,
        key=lambda r: r["best_f1"]
    )

    print(f"\n  Best overall: radius={best_overall['radius_px']}px "
          f"({best_overall['radius_m']}m), "
          f"threshold={best_overall['best_threshold']} → "
          f"F1={best_overall['best_f1']} "
          f"(ΔF1={best_overall['best_delta_f1']:+.4f})")

    return {
        "model":          model_id,
        "baseline":       baseline,
        "best_threshold": best_threshold,
        "radius_results": radius_results,
        "best_radius_px": best_overall["radius_px"],
        "best_radius_m":  best_overall["radius_m"],
        "best_hom_threshold": best_overall["best_threshold"],
        "best_f1":        best_overall["best_f1"],
        "best_delta_f1":  best_overall["best_delta_f1"],
    }


def main():
    print("RQ2: Radius-based Adaptive Spatial Voting")
    print("=" * 60)

    if not CENTROIDS_CSV.exists():
        print(f"\nERROR: centroids file not found: {CENTROIDS_CSV}")
        print("Run extract_centroids.py first.")
        return

    print(f"\nLoading centroids from {CENTROIDS_CSV.name}...")
    uid_to_data = load_centroids(CENTROIDS_CSV)
    print(f"Loaded {len(uid_to_data):,} building centroids.")

    print(f"\nRadius reference (GSD = {GSD_M_PER_PX}m/px):")
    for r in RADII_PX:
        print(f"  {r:>4}px → {r * GSD_M_PER_PX:.0f}m")

    all_results = {}
    for model_id, eval_path in EVAL_PATHS.items():
        result = evaluate_model(model_id, eval_path, uid_to_data)
        if result:
            all_results[model_id] = result

    print(f"\n{'='*60}")
    print("Summary: best result per model")
    print(f"{'='*60}")
    print(f"  {'Model':>8}  {'Base F1':>8}  {'Best F1':>8}  "
          f"{'ΔF1':>8}  {'Radius':>10}  {'Hom.T':>8}")
    print(f"  {'-'*60}")
    for model_id, res in all_results.items():
        print(f"  {model_id:>8}  {res['baseline']['f1']:>8.4f}  "
              f"{res['best_f1']:>8.4f}  "
              f"{res['best_delta_f1']:>+8.4f}  "
              f"{res['best_radius_px']:>6}px/{res['best_radius_m']:>4.0f}m  "
              f"{res['best_hom_threshold']:>8.2f}")

    output_path = OUTPUT_DIR / "clustering_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
