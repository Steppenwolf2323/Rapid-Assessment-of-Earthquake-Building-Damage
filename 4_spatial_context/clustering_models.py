"""
clustering.py — Core clustering logic for RQ2

Implements radius-based spatial clustering with adaptive voting.

Proximity definition:
    Two buildings are neighbours if their pixel centroids are within
    RADIUS pixels of each other in the original scene image.

    Since buildings from different scenes cannot be neighbours
    (different geographic areas), proximity is only computed within
    each scene. This is efficient and geographically correct.

Adaptive voting rule:
    For each building's neighbourhood (itself + all buildings within radius):
        std = standard deviation of predicted probabilities in neighbourhood
        if std < homogeneity_threshold:
            apply majority vote → everyone gets the majority class
        else:
            keep individual prediction unchanged
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


# ─── Load predictions ─────────────────────────────────────────────────────────

def load_predictions(eval_json_path: Path) -> dict:
    """
    Loads per-sample predictions from evaluation_xbd.json.
    Returns a dict with lists: paths, labels, probs, preds, subtypes.
    """
    with open(eval_json_path) as f:
        data = json.load(f)

    samples = data["per_sample"]
    return {
        "paths":     [s["path"]              for s in samples],
        "labels":    [s["label"]             for s in samples],
        "probs":     [s["prob"]              for s in samples],
        "preds":     [s["pred"]              for s in samples],
        "disasters": [s["disaster"]          for s in samples],
        "subtypes":  [s.get("subtype", "unknown") for s in samples],
        "best_threshold": data.get("best_threshold", 0.5),
    }


# ─── Load centroids ───────────────────────────────────────────────────────────

def load_centroids(centroids_csv_path: Path) -> dict:
    """
    Loads uid → (cx, cy, scene_path) mapping from the centroids CSV.
    Buildings with missing coordinates are excluded from clustering
    but their individual predictions are kept unchanged.
    """
    import pandas as pd

    df = pd.read_csv(centroids_csv_path)
    df = df.dropna(subset=["cx", "cy"])

    uid_to_data = {}
    for _, row in df.iterrows():
        uid_to_data[row["uid"]] = {
            "cx":         float(row["cx"]),
            "cy":         float(row["cy"]),
            "scene_path": str(row["scene_path"]),
        }
    return uid_to_data


# ─── Build radius-based neighbourhoods ───────────────────────────────────────

def build_radius_neighbourhoods(
    paths:          list,
    uid_to_data:    dict,
    radius_px:      float,
) -> list[list[int]]:
    """
    For each building, finds all neighbours within radius_px pixels
    in the same scene. Returns a list of neighbourhoods — one per building.

    Each neighbourhood is a list of sample indices (including the building itself).
    Buildings without known coordinates get a singleton neighbourhood [i].

    Strategy:
        1. Group buildings by scene (only buildings in same scene can be neighbours)
        2. Within each scene, compute pairwise pixel distances
        3. For each building, collect all buildings within radius
    """
    import pandas as pd

    # Map sample index → uid and centroid data
    index_to_uid  = {}
    index_to_data = {}
    for i, path in enumerate(paths):
        uid = Path(path).stem
        index_to_uid[i]  = uid
        index_to_data[i] = uid_to_data.get(uid, None)

    # Group indices by scene
    scene_to_indices = {}
    for i, path in enumerate(paths):
        uid  = Path(path).stem
        data = uid_to_data.get(uid)
        if data:
            scene_key = data["scene_path"]
        else:
            scene_key = f"__no_coords_{i}"   # singleton
        if scene_key not in scene_to_indices:
            scene_to_indices[scene_key] = []
        scene_to_indices[scene_key].append(i)

    # Build neighbourhood per building
    neighbourhoods = [[i] for i in range(len(paths))]  # default: singleton

    for scene_key, indices in scene_to_indices.items():
        if scene_key.startswith("__no_coords"):
            continue   # singleton, already set

        # Get coordinates for all buildings in this scene
        coords = []
        valid_indices = []
        for i in indices:
            data = index_to_data.get(i)
            if data:
                coords.append([data["cx"], data["cy"]])
                valid_indices.append(i)

        if len(valid_indices) < 2:
            continue

        coords_arr = np.array(coords)   # (N, 2)

        # Compute pairwise distances (Euclidean in pixel space)
        # For N buildings: N×N distance matrix
        diff       = coords_arr[:, None, :] - coords_arr[None, :, :]   # (N,N,2)
        dist_matrix = np.sqrt((diff**2).sum(axis=2))                    # (N,N)

        # For each building, find all neighbours within radius
        for local_idx, global_idx in enumerate(valid_indices):
            neighbour_local_indices = np.where(
                dist_matrix[local_idx] <= radius_px
            )[0]
            neighbour_global_indices = [
                valid_indices[j] for j in neighbour_local_indices
            ]
            neighbourhoods[global_idx] = neighbour_global_indices

    # Stats
    sizes = [len(n) for n in neighbourhoods]
    return neighbourhoods, sizes


# ─── Adaptive voting ──────────────────────────────────────────────────────────

def apply_adaptive_voting(
    probs:                 list,
    preds:                 list,
    neighbourhoods:        list,
    homogeneity_threshold: float,
    best_threshold:        float = 0.5,
) -> tuple:
    """
    Applies adaptive majority voting using radius-based neighbourhoods.

    For each building:
        1. Collect probabilities of all neighbours (including itself)
        2. Compute std of those probabilities
        3. If std < homogeneity_threshold:
               majority = 1 if >50% of neighbours have prob >= best_threshold
               assign majority to this building
           Else:
               keep original pred

    Note: each building is evaluated independently using its own neighbourhood.
    This means nearby buildings may get different treatments if their
    neighbourhood compositions differ (edge effects at scene boundaries).

    Returns:
        new_preds:    corrected predictions
        vote_applied: bool per sample
    """
    new_preds    = list(preds)
    vote_applied = [False] * len(preds)
    cluster_stds = []

    for i, neighbours in enumerate(neighbourhoods):
        if len(neighbours) < 2:
            cluster_stds.append(0.0)
            continue

        neighbour_probs = [probs[j] for j in neighbours]
        std             = float(np.std(neighbour_probs))
        cluster_stds.append(std)

        if std < homogeneity_threshold:
            # Homogeneous neighbourhood → majority vote
            n_damaged = sum(1 for j in neighbours if probs[j] >= best_threshold)
            majority  = 1 if n_damaged > len(neighbours) / 2 else 0
            new_preds[i]    = majority
            vote_applied[i] = True

    return new_preds, vote_applied, cluster_stds


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs) -> dict:
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    return {
        "f1":        round(f1_score(labels, preds,        zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds,    zero_division=0), 4),
        "auc":       round(float(auc), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "n_samples": len(labels),
        "n_damaged": int(sum(labels)),
    }
