
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Find the ground-truth images
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = sorted(data_folder.rglob("*_Summer_GT-*.png"))

print("Ground-truth images found:", len(files))


# STEP 2: Measure the left and right edges
def get_edges(mask, fraction):
    y = int(mask.shape[0] * fraction)

    white = np.where(mask[y, :] > 127)[0]

    if len(white) < 2:
        return np.nan, np.nan

    return float(white[0]), float(white[-1])


# STEP 3: Extract refined geometry features
results = []

for file in files:
    tree_id = file.name.split("_")[0]

    mask = np.array(Image.open(file).convert("L"))

    # Measure the tree at three heights
    L25, R25 = get_edges(mask, 0.25)
    L50, R50 = get_edges(mask, 0.50)
    L75, R75 = get_edges(mask, 0.75)

    # Horizontal widths
    W25 = R25 - L25
    W50 = R50 - L50
    W75 = R75 - L75

    # Branch midpoints
    C25 = (L25 + R25) / 2
    C75 = (L75 + R75) / 2

    # Horizontal centre offset
    asymmetry = C25 - C75

    # Total width expansion
    width_expansion = W25 - W75

    # Vertical distance between measurement rows
    y25 = int(mask.shape[0] * 0.25)
    y75 = int(mask.shape[0] * 0.75)
    vertical_distance = y75 - y25

    # Width expansion per pixel of vertical distance
    expansion_rate = width_expansion / vertical_distance

    results.append([
        tree_id,
        W25,
        W50,
        W75,
        asymmetry,
        width_expansion,
        expansion_rate
    ])


# STEP 4: Save the results
df = pd.DataFrame(
    results,
    columns=[
        "Tree_ID",
        "Width_25",
        "Width_50",
        "Width_75",
        "Asymmetry",
        "Width_Expansion",
        "Expansion_Rate"
    ]
)

output_folder = project / "results"
output_folder.mkdir(exist_ok=True)

df.to_csv(
    output_folder / "refined_geometry.csv",
    index=False
)

print("\nSummary statistics:")
print(df.describe().round(2))

print("\nMissing measurements:")
print(df.isna().sum())


# STEP 5: Compare expansion rates across trees
plt.figure(figsize=(8, 5))

plt.hist(df["Expansion_Rate"].dropna(), bins=25)

plt.xlabel("Width expansion rate (pixels/pixel)")
plt.ylabel("Number of trees")
plt.title("Apple Tree Width Expansion Rate")
plt.tight_layout()
plt.show()


# STEP 6: Compare expansion rate against asymmetry
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Expansion_Rate"],
    df["Asymmetry"],
    alpha=0.5
)

plt.xlabel("Width expansion rate (pixels/pixel)")
plt.ylabel("Horizontal centre offset (pixels)")
plt.title("Tree Expansion Rate vs Asymmetry")
plt.grid(True)
plt.tight_layout()
plt.show()


# STEP 7: Check previously investigated trees
check_trees = [
    "R01N03",
    "R20N03",
    "R01N43",
    "R09N18",
    "R09N28"
]

print("\nPreviously validated trees:")
print(
    df[df["Tree_ID"].isin(check_trees)]
    .round(2)
    .to_string(index=False)
)
