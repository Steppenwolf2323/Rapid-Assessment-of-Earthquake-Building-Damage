import json
from pathlib import Path
import pandas as pd
from collections import Counter

df = pd.read_csv(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\xbd_dataset\xbd_test.csv")

mexico = df[df["disaster"] == "mexico-earthquake"]
subtypes = Counter()

for path in mexico['path']:
    json_path = Path(path).parent.parent / "labels" / Path(path).with_suffix(".json").name
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for building in data["features"]["lng_lat"]:
            subtypes[building["properties"]["subtype"]] += 1

print("Mexico earthquake building subtypes:")
for subtype, count in subtypes.most_common():
    print(f"  {subtype}: {count}")
print(f"\nTotal after excluding un-classified:")
damaged = subtypes['destroyed'] + subtypes['major-damage'] + subtypes['minor-damage']
intact  = subtypes['no-damage']
print(f"  Intact:  {intact}")
print(f"  Damaged: {damaged}")
print(f"  Total:   {intact + damaged}")
print(f"  Ratio:   {intact/max(damaged,1):.1f}:1")