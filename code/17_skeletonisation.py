
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    skeletonize,
    disk
)

# STEP 1: Settings
TREE_ID = "R20N03"
THRESHOLD = 80
MIN_OBJECT_SIZE = 50
CLOSING_RADIUS = 8

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = list(data_folder.rglob(f"{TREE_ID}_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Image not found")

depth = np.array(Image.open(files[0]).convert("L"))

height, width = depth.shape

# STEP 2: Threshold the image
binary = depth >= THRESHOLD

# Temporarily exclude the bottom 18% of the image.
# This reduces ground interference but may cut off the trunk.
cutoff = int(height * 0.82)
binary[cutoff:, :] = False

# STEP 3: Clean the binary image
cleaned = binary_closing(
    binary,
    footprint=disk(CLOSING_RADIUS)
)

cleaned = remove_small_objects(
    cleaned,
    min_size=MIN_OBJECT_SIZE
)

# STEP 4: Identify connected regions
labels, number = ndimage.label(
    cleaned,
    structure=np.ones((3, 3))
)

print("Connected regions:", number)

sizes = np.bincount(labels.ravel())[1:]

if len(sizes) > 0:
    print("Largest component sizes:")
    print(sorted(sizes, reverse=True)[:10])

# STEP 5: Create a skeleton
skeleton = skeletonize(cleaned)

print("Foreground pixels:", np.sum(cleaned))
print("Skeleton pixels:", np.sum(skeleton))

# STEP 6: Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].imshow(depth, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original winter depth")

axes[0, 1].imshow(binary, cmap="gray")
axes[0, 1].set_title("Thresholded image")

axes[1, 0].imshow(cleaned, cmap="gray")
axes[1, 0].set_title("Cleaned foreground")

axes[1, 1].imshow(depth, cmap="gray", vmin=0, vmax=255)

# Overlay skeleton as red pixels
overlay = np.ma.masked_where(~skeleton, skeleton)

axes[1, 1].imshow(
    overlay,
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.9
)

axes[1, 1].set_title("Skeleton over depth image")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

# STEP 7: Display connected components
plt.figure(figsize=(8, 5))

plt.imshow(labels, cmap="nipy_spectral")
plt.title("Connected foreground components")
plt.axis("off")
plt.tight_layout()
plt.show()

# STEP 8: Test different morphological closing radii

radii = [2, 4, 6, 8]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, radius in enumerate(radii):

    test_cleaned = binary_closing(
        binary,
        footprint=disk(radius)
    )

    test_cleaned = remove_small_objects(
        test_cleaned,
        min_size=MIN_OBJECT_SIZE
    )

    # Calculate connected components
    test_labels, count = ndimage.label(
        test_cleaned,
        structure=np.ones((3, 3))
    )

    # Calculate skeleton
    test_skeleton = skeletonize(test_cleaned)

    axes[i].imshow(depth, cmap="gray", vmin=0, vmax=255)

    overlay = np.ma.masked_where(
        ~test_skeleton, test_skeleton
    )

    axes[i].imshow(
        overlay,
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.9
    )

    axes[i].set_title(f"Radius = {radius}\nComponents = {count}")
    axes[i].axis("off")

    print(
        f"Radius {radius}: "
        f"{count} connected components"
    )

plt.tight_layout()
plt.show()

# Check whether the two branches belong to the same component

# Coordinates are (x, y), taken from points on the branches
plt.figure(figsize=(9, 6))
plt.imshow(cleaned, cmap="gray")
plt.title("Click the left branch, then the right branch")
plt.axis("off")

points = plt.ginput(2, timeout=0)
plt.close()

if len(points) == 2:
    component_ids = []

    for x, y in points:
        x = int(round(x))
        y = int(round(y))

        component_id = labels[y, x]
        component_ids.append(component_id)

    print("Left branch component:", component_ids[0])
    print("Right branch component:", component_ids[1])

    if component_ids[0] == component_ids[1] != 0:
        print("SUCCESS: Both branches are connected!")
    else:
        print("Branches are not in the same component.")
