
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Locate the winter depth image
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = sorted(data_folder.rglob("R01N01_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Winter depth image not found")

# Load one image as grayscale
depth = np.array(Image.open(files[0]).convert("L"))

print("Image shape:", depth.shape)
print("Minimum value:", depth.min())
print("Maximum value:", depth.max())


# STEP 2: Choose numerical thresholds
thresholds = [20, 50, 80, 110]

# Display the original image and threshold results
fig, axes = plt.subplots(1, 5, figsize=(16, 4))

axes[0].imshow(depth, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original depth")
axes[0].axis("off")

for i, threshold in enumerate(thresholds):

    # Keep pixels at or above the threshold
    binary = depth >= threshold

    axes[i + 1].imshow(binary, cmap="gray")
    axes[i + 1].set_title(f"Threshold = {threshold}")
    axes[i + 1].axis("off")

    # Percentage of pixels selected
    percent = np.mean(binary) * 100
    print(f"Threshold {threshold}: {percent:.1f}% selected")

plt.tight_layout()
plt.show()


# STEP 3: Investigate a smaller central region
# This may reduce some background interference.

height, width = depth.shape

# Crop central 60% of the image width
left = int(width * 0.20)
right = int(width * 0.80)

cropped = depth[:, left:right]

threshold = 80
binary_crop = cropped >= threshold

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cropped, cmap="gray", vmin=0, vmax=255)
plt.title("Central depth region")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_crop, cmap="gray")
plt.title("Central region: threshold 80")
plt.axis("off")

plt.tight_layout()
plt.show()

