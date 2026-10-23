
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
CLOSING_RADIUS = 8
MIN_OBJECT_SIZE = 50

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = sorted(data_folder.rglob(f"{TREE_ID}_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Winter depth image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape

# STEP 2: Select the trunk position
# This helps us identify the correct tree in the image.
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap="gray")
plt.title("Click the trunk just below the main V junction")
plt.axis("off")

points = plt.ginput(1, timeout=0)
plt.close()

if not points:
    raise RuntimeError("No trunk position selected")

trunk_x, trunk_y = points[0]
trunk_x = int(round(trunk_x))
trunk_y = int(round(trunk_y))

# STEP 3: Threshold and clean the image
binary = depth >= THRESHOLD

# Exclude the bottom 18% as in our previous experiment
cutoff = int(height * 0.82)
binary[cutoff:, :] = False

cleaned = binary_closing(
    binary,
    footprint=disk(CLOSING_RADIUS)
)

cleaned = remove_small_objects(
    cleaned,
    min_size=MIN_OBJECT_SIZE
)

# STEP 4: Reduce ground interference
# In the lower 25% of the processed region,
# retain only pixels near the selected trunk.
#
# This is an experimental geometric restriction.

lower_start = int(height * 0.65)
trunk_half_width = 65

for y in range(lower_start, cutoff):
    left = max(0, trunk_x - trunk_half_width)
    right = min(width, trunk_x + trunk_half_width)

    cleaned[y, :left] = False
    cleaned[y, right:] = False

# STEP 5: Find the component containing the trunk
labels, count = ndimage.label(
    cleaned,
    structure=np.ones((3, 3))
)

# Find a foreground pixel near the selected location
search_y1 = max(0, trunk_y - 15)
search_y2 = min(height, trunk_y + 16)
search_x1 = max(0, trunk_x - 15)
search_x2 = min(width, trunk_x + 16)

area = labels[search_y1:search_y2, search_x1:search_x2]
nearby = area[area > 0]

if len(nearby) == 0:
    raise RuntimeError(
        "No foreground near the selected trunk. "
        "Try a different point or threshold."
    )

component_id = np.bincount(nearby).argmax()
tree_mask = labels == component_id

print("Connected components:", count)
print("Selected component:", component_id)

# STEP 6: Skeletonise the selected tree
skeleton = skeletonize(tree_mask)

# Count the eight neighbours around each skeleton pixel
kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0

neighbours = ndimage.convolve(
    skeleton.astype(int),
    kernel,
    mode="constant",
    cval=0
)

# Candidate junctions have at least 3 neighbours
candidates = skeleton & (neighbours >= 3)

# Focus on the lower-middle tree, away from the ground
Y, X = np.indices(skeleton.shape)

search_region = (
    (Y >= int(height * 0.45)) &
    (Y <= int(height * 0.80)) &
    (np.abs(X - trunk_x) <= 110)
)

candidates = candidates & search_region

# STEP 7: Group neighbouring candidate pixels
candidate_labels, candidate_count = ndimage.label(
    candidates,
    structure=np.ones((3, 3))
)

centres = []

for label_id in range(1, candidate_count + 1):
    ys, xs = np.where(candidate_labels == label_id)

    centres.append([
        float(np.mean(xs)),
        float(np.mean(ys))
    ])

print("Junction candidate groups:", len(centres))

for x, y in centres:
    print(f"Candidate at x={x:.1f}, y={y:.1f}")

# STEP 8: Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(depth, cmap="gray")
axes[0].set_title("Original winter depth")

axes[1].imshow(tree_mask, cmap="gray")
axes[1].set_title("Selected tree foreground")

axes[2].imshow(depth, cmap="gray")

overlay = np.ma.masked_where(~skeleton, skeleton)

axes[2].imshow(
    overlay,
    cmap="autumn",
    alpha=0.8
)

for x, y in centres:
    axes[2].scatter(
        x, y,
        color="cyan",
        s=65,
        marker="x"
    )

axes[2].scatter(
    trunk_x, trunk_y,
    color="lime",
    s=60,
    label="Selected trunk"
)

axes[2].set_title("Skeleton and junction candidates")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
