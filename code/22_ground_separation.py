
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy import ndimage
from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    skeletonize,
    disk
)

# STEP 1: SETTINGS
TREE_ID = "R01N03"
THRESHOLD = 80
CLOSING_RADIUS = 8
HALF_WIDTH = 180

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"
results_folder.mkdir(exist_ok=True)

files = sorted(
    data_folder.rglob(f"{TREE_ID}_Winter_D-*.png")
)

if not files:
    raise FileNotFoundError("Winter depth image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape


# STEP 2: SELECT LOWER TRUNK
plt.figure(figsize=(10, 7))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
plt.title("Click on the lower curved trunk")
plt.axis("off")

points = plt.ginput(1, timeout=0)
plt.close()

if not points:
    raise RuntimeError("No trunk selected")

trunk_x, trunk_y = map(
    lambda v: int(round(v)), points[0]
)


# STEP 3: ORIGINAL PREPROCESSING
binary = depth >= THRESHOLD

cutoff = int(height * 0.82)
binary[cutoff:, :] = False

cleaned = binary_closing(
    binary,
    footprint=disk(CLOSING_RADIUS)
)

cleaned = remove_small_objects(
    cleaned, min_size=50
)

# Wide selection to preserve curved trunk
lower_start = int(height * 0.65)

for y in range(lower_start, cutoff):
    left = max(0, trunk_x - HALF_WIDTH)
    right = min(width, trunk_x + HALF_WIDTH)

    cleaned[y, :left] = False
    cleaned[y, right:] = False

# Select connected component near trunk
labels, _ = ndimage.label(
    cleaned, structure=np.ones((3, 3), dtype=int)
)

y1 = max(0, trunk_y - 15)
y2 = min(height, trunk_y + 16)
x1 = max(0, trunk_x - 15)
x2 = min(width, trunk_x + 16)

nearby = labels[y1:y2, x1:x2]
nearby = nearby[nearby > 0]

if len(nearby) == 0:
    raise RuntimeError("No foreground near trunk")

component_id = np.bincount(nearby).argmax()
original_mask = labels == component_id

original_skeleton = skeletonize(original_mask)


# STEP 4: DRAW TREE REGION
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(depth, cmap="gray", vmin=0, vmax=255)

ax.set_title(
    "Click around the OUTSIDE of the target tree.\n"
    "Press Enter when finished."
)
ax.axis("off")

print("\nClick points around the tree boundary.")
print("Include the complete lower curved trunk.")
print("Exclude neighbouring trees and ground.")
print("Press Enter to finish.")

boundary = plt.ginput(n=-1, timeout=0)
plt.close(fig)

if len(boundary) < 3:
    raise RuntimeError("At least 3 boundary points needed")

# Convert selected polygon into image mask
yy, xx = np.indices((height, width))

pixel_positions = np.column_stack([
    xx.ravel(), yy.ravel()
])

polygon = MplPath(boundary)

region_mask = polygon.contains_points(
    pixel_positions
).reshape(height, width)

# STEP 5: APPLY SELECTED REGION
refined_mask = original_mask & region_mask

# Keep only meaningful connected foreground pieces
refined_mask = remove_small_objects(
    refined_mask, min_size=50
)

refined_skeleton = skeletonize(refined_mask)

print("\n--- GROUND SEPARATION ---")
print("Original foreground pixels:",
      int(original_mask.sum()))
print("Refined foreground pixels:",
      int(refined_mask.sum()))
print("Original skeleton pixels:",
      int(original_skeleton.sum()))
print("Refined skeleton pixels:",
      int(refined_skeleton.sum()))


# STEP 6: DISPLAY COMPARISON
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].imshow(depth, cmap="gray",
                  vmin=0, vmax=255)
axes[0, 0].set_title("Original winter depth")

axes[0, 1].imshow(original_mask, cmap="gray")
axes[0, 1].set_title("Before ground separation")

axes[1, 0].imshow(refined_mask, cmap="gray")
axes[1, 0].set_title("After polygon selection")

axes[1, 1].imshow(depth, cmap="gray",
                  vmin=0, vmax=255)

overlay = np.ma.masked_where(
    ~refined_skeleton, refined_skeleton
)

axes[1, 1].imshow(
    overlay, cmap="autumn",
    vmin=0, vmax=1, alpha=0.9
)

axes[1, 1].set_title("Refined skeleton")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    results_folder / f"{TREE_ID}_ground_separation.png",
    dpi=150
)

plt.show()

# Save masks for later analysis
np.save(
    results_folder / f"{TREE_ID}_refined_mask.npy",
    refined_mask
)

np.save(
    results_folder / f"{TREE_ID}_refined_skeleton.npy",
    refined_skeleton
)

print("Results saved.")
