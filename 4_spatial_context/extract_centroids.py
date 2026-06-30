"""
extract_centroids.py — Step 1 of RQ2: Extract building centroids

Reads the xBD test buildings CSV and the original scene JSON files
to extract the pixel centroid of each building polygon.

Saves centroids to: xbd_building_centroids.csv
    columns: uid, scene_path, cx, cy, real_world_m_per_px

Run this ONCE before evaluate_clustering.py.

GSD (ground sample distance) for xBD Mexico earthquake:
    ~2.65 metres per pixel (multispectral)
    from metadata: 'gsd': 2.65

So a radius of:
    50px  → ~132m   (half a city block)
    100px → ~265m   (one city block)
    150px → ~397m   (1.5 city blocks)
    200px → ~530m   (2 city blocks)

Usage:
    cd 4_spatial_context
    python extract_centroids.py
"""

import json
from pathlib import Path

import pandas as pd
from shapely.wkt import loads as wkt_loads


BASE = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")

XBD_CSV        = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_test_buildings.csv"
OUTPUT_CENTROIDS = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_building_centroids.csv"

# GSD from xBD Mexico earthquake metadata (metres per pixel)
GSD_M_PER_PX = 2.65


def main():
    df = pd.read_csv(XBD_CSV)
    print(f"Loaded {len(df)} buildings from CSV.")
    print(f"Extracting centroids from scene JSON files...\n")

    scene_cache = {}

    rows        = []
    n_found     = 0
    n_missing   = 0

    for _, row in df.iterrows():
        uid        = Path(row["path"]).stem
        scene_path = Path(row["scene_path"])

        scene_key = str(scene_path)
        if scene_key not in scene_cache:
            json_path = scene_path.parent.parent / "labels" / \
                        scene_path.with_suffix(".json").name
            if not json_path.exists():
                scene_cache[scene_key] = {}
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                # Build uid → centroid mapping for this scene
                uid_map = {}
                for building in data["features"]["xy"]:
                    b_uid = building["properties"].get("uid")
                    if b_uid:
                        try:
                            polygon = wkt_loads(building["wkt"])
                            uid_map[b_uid] = (polygon.centroid.x,
                                              polygon.centroid.y)
                        except Exception:
                            pass
                scene_cache[scene_key] = uid_map
            except Exception as e:
                print(f"  WARNING: could not load {json_path.name}: {e}")
                scene_cache[scene_key] = {}

        uid_map = scene_cache.get(scene_key, {})

        if uid in uid_map:
            cx, cy = uid_map[uid]
            rows.append({
                "uid":              uid,
                "scene_path":       str(scene_path),
                "cx":               round(cx, 2),
                "cy":               round(cy, 2),
                "gsd_m_per_px":     GSD_M_PER_PX,
            })
            n_found += 1
        else:
            rows.append({
                "uid":          uid,
                "scene_path":   str(scene_path),
                "cx":           None,
                "cy":           None,
                "gsd_m_per_px": GSD_M_PER_PX,
            })
            n_missing += 1

        if (n_found + n_missing) % 5000 == 0:
            print(f"  Progress: {n_found + n_missing}/{len(df)} "
                  f"(found: {n_found}, missing: {n_missing})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CENTROIDS, index=False)

    print(f"\nDone.")
    print(f"  Found:   {n_found:,} centroids")
    print(f"  Missing: {n_missing:,}")
    print(f"  Saved → {OUTPUT_CENTROIDS}")

    print(f"\nRadius reference table (GSD = {GSD_M_PER_PX}m/px):")
    for r_px in [50, 100, 150, 200, 300]:
        print(f"  {r_px:>4}px → {r_px * GSD_M_PER_PX:.0f}m real world")


if __name__ == "__main__":
    main()
