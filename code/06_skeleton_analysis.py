
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Locate the ground-truth images
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

gt_files = sorted(data_folder.rglob("*_Summer_GT-*.png"))

print("Ground-truth images found:", len(gt_files))

# STEP 2: Function to measure branch width
def measure_width(mask, height_fraction):
    height = mask.shape[0]

    # Select a horizontal row in the image
    row = int(height * height_fraction)

    # Find all white pixels in that row
    white_pixels = np.where(mask[row, :] > 127)[0]

    if len(white_pixels) < 2:
        return np.nan

    # Distance from leftmost to rightmost white pixel
    width = white_pixels[-1] - white_pixels[0]

    return width


# STEP 3: Analyse every GT image
results = []

for file in gt_files:
    tree_id = file.name.split("_")[0]

    image = Image.open(file).convert("L")
    mask = np.array(image)

    # Measure width at 25%, 50%, 75% of image height
    width_25 = measure_width(mask, 0.25)
    width_50 = measure_width(mask, 0.50)
    width_75 = measure_width(mask, 0.75)

    results.append([
        tree_id,
        width_25,
        width_50,
        width_75
    ])


# STEP 4: Save the extracted measurements
output_folder = project / "results"
output_folder.mkdir(exist_ok=True)

df = pd.DataFrame(
    results,
    columns=["Tree_ID", "Width_25", "Width_50", "Width_75"]
)

df.to_csv(
    output_folder / "skeleton_features.csv",
    index=False
)

print("\nFirst five results:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())


# STEP 5: Plot the branch widths
plt.figure(figsize=(8, 5))

plt.hist(df["Width_25"].dropna(), bins=25, alpha=0.6, label="25% height")
plt.hist(df["Width_50"].dropna(), bins=25, alpha=0.6, label="50% height")
plt.hist(df["Width_75"].dropna(), bins=25, alpha=0.6, label="75% height")

plt.xlabel("Horizontal tree width (pixels)")
plt.ylabel("Number of trees")
plt.title("Ground-Truth Tree Width Distribution")
plt.legend()
plt.tight_layout()
plt.show()
