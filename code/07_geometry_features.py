
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Find the GT images
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = sorted(data_folder.rglob("*_Summer_GT-*.png"))

print("GT images found:", len(files))


# STEP 2: Find the left and right edges of each tree
def get_edges(mask, fraction):
    y = int(mask.shape[0] * fraction)

    white = np.where(mask[y, :] > 127)[0]

    if len(white) < 2:
        return np.nan, np.nan

    return float(white[0]), float(white[-1])


# STEP 3: Extract geometric features
results = []

for file in files:
    tree_id = file.name.split("_")[0]

    mask = np.array(Image.open(file).convert("L"))

    # Edges at three heights
    left25, right25 = get_edges(mask, 0.25)
    left50, right50 = get_edges(mask, 0.50)
    left75, right75 = get_edges(mask, 0.75)

    # Branch widths
    width25 = right25 - left25
    width50 = right50 - left50
    width75 = right75 - left75

    # Estimate opening angle between 25% and 50%
    vertical_distance = 0.25 * mask.shape[0]

    left_angle = np.degrees(
        np.arctan2(left50 - left25, vertical_distance)
    )

    right_angle = np.degrees(
        np.arctan2(right25 - right50, vertical_distance)
    )

    opening_angle = left_angle + right_angle

    # Estimate asymmetry relative to the lower trunk
    upper_centre = (left25 + right25) / 2
    lower_centre = (left75 + right75) / 2

    asymmetry = upper_centre - lower_centre

    results.append([
        tree_id,
        width25,
        width50,
        width75,
        opening_angle,
        asymmetry
    ])


# STEP 4: Save results to CSV
df = pd.DataFrame(
    results,
    columns=[
        "Tree_ID",
        "Width_25",
        "Width_50",
        "Width_75",
        "Opening_Angle",
        "Asymmetry"
    ]
)

output_folder = project / "results"
output_folder.mkdir(exist_ok=True)

df.to_csv(
    output_folder / "geometry_features.csv",
    index=False
)

print("\nFirst five trees:")
print(df.head().round(2))

print("\nSummary statistics:")
print(df.describe().round(2))

print("\nMissing measurements:")
print(df.isna().sum())


# STEP 5: Plot opening angles
plt.figure(figsize=(8, 5))
plt.hist(df["Opening_Angle"].dropna(), bins=25)

plt.xlabel("Estimated opening angle (degrees)")
plt.ylabel("Number of trees")
plt.title("Apple Tree Branch Opening Angle")
plt.tight_layout()
plt.show()


# STEP 6: Plot asymmetry
plt.figure(figsize=(8, 5))
plt.hist(df["Asymmetry"].dropna(), bins=25)

plt.xlabel("Horizontal centre offset (pixels)")
plt.ylabel("Number of trees")
plt.title("Apple Tree Branch Asymmetry")
plt.tight_layout()
plt.show()


# STEP 7: Compare opening angle with upper width
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Width_25"],
    df["Opening_Angle"],
    alpha=0.5
)

plt.xlabel("Width at 25% height (pixels)")
plt.ylabel("Estimated opening angle (degrees)")
plt.title("Tree Width vs Branch Opening Angle")
plt.grid(True)
plt.tight_layout()
plt.show()
