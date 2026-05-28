import json
from pathlib import Path

# Load one xBD JSON
label_path = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\xBD_train_images_labels_targets\train\labels\mexico-earthquake_00000055_post_disaster.json")

with open(label_path) as f:
    data = json.load(f)

buildings = data["features"]["lng_lat"]
print(f"Number of building polygons in this file: {len(buildings)}")
print(f"\nFirst building:")
print(f"  uid:      {buildings[0]['properties']['uid']}")
print(f"  subtype:  {buildings[0]['properties']['subtype']}")
print(f"  wkt:      {buildings[0]['wkt'][:80]}...")

# Check image name to understand scope
print(f"\nImage this JSON belongs to: {data['metadata']['img_name']}")