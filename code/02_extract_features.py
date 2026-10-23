
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import re

# STEP 1: Find the image folder
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "Data"

# STEP 2: Find all depth images
files = sorted(data_folder.rglob("*.png"))

results = []
seen = set()

# STEP 3: Analyse each depth image
for file in files:

    # Identify tree ID and season
    match = re.match(r"(R\d+N\d+)_(Summer|Winter)_D-", file.name)

    if match is None:
        continue

    tree_id = match.group(1)
    season = match.group(2)

    # Prevent duplicate observations
    key = (tree_id, season)
    if key in seen:
        print("Duplicate image:", file.name)
        continue
    seen.add(key)

    # Load the image as grayscale
    image = Image.open(file).convert("L")
    depth = np.array(image)

    # Separate zero and nonzero pixels
    nonzero = depth[depth > 0]

    # Calculate numerical features
    zero_percent = np.mean(depth == 0) * 100
    nonzero_percent = np.mean(depth > 0) * 100

    if len(nonzero) > 0:
        mean_depth = np.mean(nonzero)
        std_depth = np.std(nonzero)
    else:
        mean_depth = np.nan
        std_depth = np.nan

    # Store the results
    results.append([
        tree_id,
        season,
        round(float(mean_depth), 2),
        round(float(std_depth), 2),
        round(float(zero_percent), 2),
        round(float(nonzero_percent), 2)
    ])

# STEP 4: Save the results
output_folder = project_folder / "results"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "tree_features.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Tree_ID",
        "Season",
        "Mean_Depth",
        "Std_Depth",
        "Zero_Percent",
        "Nonzero_Percent"
    ])

    writer.writerows(results)

# STEP 5: Display summary
tree_ids = set(row[0] for row in results)

print("Unique trees:", len(tree_ids))
print("Depth images analysed:", len(results))
print("CSV saved to:", output_file)

print("\nFirst five results:")
for row in results[:5]:
    print(row)
