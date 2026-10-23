
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Load the winter depth image
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = list(data_folder.rglob("R01N01_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Winter depth image not found")

depth = np.array(Image.open(files[0]).convert("L"))

height, width = depth.shape

# STEP 2: Apply the original threshold
threshold = 80
binary = depth >= threshold


# STEP 3: Test different V-shaped search widths
Y, X = np.indices((height, width))

centre_x = 0.50 * width
junction_y = 0.78 * height
top_spread = 0.27 * width

fraction = np.clip(
    (junction_y - Y) / junction_y, 0, 1
)

left_branch = centre_x - top_spread * fraction
right_branch = centre_x + top_spread * fraction

band_sizes = [0.10, 0.20, 0.30, 0.40]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, band_fraction in enumerate(band_sizes):

    band_width = band_fraction * width

    left_region = np.abs(X - left_branch) <= band_width / 2
    right_region = np.abs(X - right_branch) <= band_width / 2

    upper_region = (Y <= junction_y) & (
        left_region | right_region
    )

    trunk_region = (Y > junction_y) & (
        np.abs(X - centre_x) <= band_width / 2
    )

    roi = upper_region | trunk_region
    prediction = binary & roi

    axes[i].imshow(prediction, cmap="gray")
    axes[i].set_title(f"Band width = {band_fraction:.0%}")
    axes[i].axis("off")

    retained = prediction.sum() / binary.sum() * 100

    print(
        f"Band {band_fraction:.0%}: "
        f"{retained:.1f}% of thresholded pixels retained"
    )

plt.tight_layout()
plt.show()
