from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: File locations
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"

geometry_file = results_folder / "geometry_features.csv"
df = pd.read_csv(geometry_file)

print("Geometry rows found:", len(df))

# STEP 2: Helper function to find left and right edges at a given height
def get_edges(mask, fraction):
    y = int(mask.shape[0] * fraction)
    white = np.where(mask[y, :] > 127)[0]

    if len(white) < 2:
        return y, np.nan, np.nan

    return y, float(white[0]), float(white[-1])

# STEP 3: Choose trees to inspect
# We will check:
# - smallest opening angle
# - median opening angle
# - largest opening angle
# - most left-leaning asymmetry
# - most right-leaning asymmetry

smallest_angle = df.loc[df["Opening_Angle"].idxmin()]
largest_angle = df.loc[df["Opening_Angle"].idxmax()]
median_angle = df.iloc[(df["Opening_Angle"] - df["Opening_Angle"].median()).abs().argsort()[:1]].iloc[0]
left_asym = df.loc[df["Asymmetry"].idxmin()]
right_asym = df.loc[df["Asymmetry"].idxmax()]

selected = pd.DataFrame([
    smallest_angle,
    median_angle,
    largest_angle,
    left_asym,
    right_asym
]).drop_duplicates(subset="Tree_ID")

print("\nTrees selected for validation:")
print(selected[["Tree_ID", "Opening_Angle", "Asymmetry"]].round(2))

# STEP 4: Plot each selected tree
for _, row in selected.iterrows():
    tree_id = row["Tree_ID"]

    # Find the GT file for this tree
    matches = list(data_folder.rglob(f"{tree_id}_Summer_GT-*.png"))

    if not matches:
        print(f"GT file not found for {tree_id}")
        continue

    gt_file = matches[0]
    mask = np.array(Image.open(gt_file).convert("L"))

    # Measure rows and edges again directly from the mask
    y25, left25, right25 = get_edges(mask, 0.25)
    y50, left50, right50 = get_edges(mask, 0.50)
    y75, left75, right75 = get_edges(mask, 0.75)

    # Centres
    centre25 = (left25 + right25) / 2
    centre75 = (left75 + right75) / 2

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")

    # Draw horizontal measurement rows
    for y in [y25, y50, y75]:
        plt.axhline(y=y, color="yellow", linestyle="--", linewidth=1)

    # Mark left and right edge points
    plt.scatter([left25, right25], [y25, y25], marker="o", s=40)
    plt.scatter([left50, right50], [y50, y50], marker="o", s=40)
    plt.scatter([left75, right75], [y75, y75], marker="o", s=40)

    # Mark centres
    plt.scatter([centre25, centre75], [y25, y75], marker="x", s=80)

    title = (
        f"{tree_id}\n"
        f"Angle = {row['Opening_Angle']:.2f}°, "
        f"Asymmetry = {row['Asymmetry']:.2f} px\n"
        f"W25 = {row['Width_25']:.0f}, "
        f"W50 = {row['Width_50']:.0f}, "
        f"W75 = {row['Width_75']:.0f}"
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()