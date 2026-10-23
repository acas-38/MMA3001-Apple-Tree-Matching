
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import (
    binary_closing, remove_small_objects,
    skeletonize, disk
)

# STEP 1: Settings
TREE_ID = "R20N03"
THRESHOLD = 80
CLOSING_RADIUS = 8

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = sorted(data_folder.rglob(
    f"{TREE_ID}_Winter_D-*.png"
))

if not files:
    raise FileNotFoundError("Tree image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape

# STEP 2: Manually select lower trunk
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
plt.title("Click on the trunk BELOW the main junction")
plt.axis("off")

clicked = plt.ginput(1, timeout=0)
plt.close()

if not clicked:
    raise RuntimeError("No trunk point selected")

trunk_x, trunk_y = map(
    lambda v: int(round(v)), clicked[0]
)

# STEP 3: Create binary tree mask
binary = depth >= THRESHOLD

# Exclude bottom ground region
cutoff = int(height * 0.82)
binary[cutoff:, :] = False

cleaned = binary_closing(
    binary, footprint=disk(CLOSING_RADIUS)
)

cleaned = remove_small_objects(
    cleaned, min_size=50
)

# Restrict the lower region near the trunk
lower_start = int(height * 0.65)
half_width = 65

for y in range(lower_start, cutoff):
    left = max(0, trunk_x - half_width)
    right = min(width, trunk_x + half_width)

    cleaned[y, :left] = False
    cleaned[y, right:] = False

# STEP 4: Identify selected tree component
structure = np.ones((3, 3), dtype=int)

labels, count = ndimage.label(
    cleaned, structure=structure
)

y1 = max(0, trunk_y - 15)
y2 = min(height, trunk_y + 16)
x1 = max(0, trunk_x - 15)
x2 = min(width, trunk_x + 16)

nearby = labels[y1:y2, x1:x2]
nearby = nearby[nearby > 0]

if len(nearby) == 0:
    raise RuntimeError("No foreground found near trunk")

component_id = np.bincount(nearby).argmax()
tree_mask = labels == component_id

# STEP 5: Skeletonisation
skeleton = skeletonize(tree_mask)

# Count neighbours of every skeleton pixel
kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0

neighbours = ndimage.convolve(
    skeleton.astype(int),
    kernel, mode="constant"
)

# Find possible junction pixels
candidates = skeleton & (neighbours >= 3)

Y, X = np.indices(skeleton.shape)

# Limit search to lower-middle portion
candidates &= (
    (Y > height * 0.45) &
    (Y < height * 0.80) &
    (np.abs(X - trunk_x) < 110)
)

candidate_labels, n = ndimage.label(
    candidates, structure=structure
)

# STEP 6: Score candidate junctions
# Removing a small neighbourhood around each junction
# lets us investigate the paths connected to it.

junction_results = []

for label_id in range(1, n + 1):

    ys, xs = np.where(candidate_labels == label_id)

    cx = int(round(np.mean(xs)))
    cy = int(round(np.mean(ys)))

    # Remove a small circle around the candidate
    radius = 6

    cut = skeleton.copy()

    local_circle = (
        (X - cx)**2 + (Y - cy)**2 <= radius**2
    )
    cut[local_circle] = False

    arm_labels, arm_count = ndimage.label(
        cut, structure=structure
    )

    # Inspect connected fragments that touch
    # the removed circular neighbourhood
    nearby_ring = (
        ((X - cx)**2 + (Y - cy)**2 <= (radius + 2)**2)
        & ~local_circle
    )

    touching_ids = np.unique(arm_labels[nearby_ring])
    touching_ids = touching_ids[touching_ids > 0]

    left_arms = []
    right_arms = []
    lower_arms = []

    for arm_id in touching_ids:
        ay, ax = np.where(arm_labels == arm_id)

        if len(ax) < 10:
            continue

        # How far does this arm extend?
        top_y = np.min(ay)
        bottom_y = np.max(ay)

        # Position near the upper part of the arm
        upper = ay < cy - 15

        if np.any(upper):
            upper_x = np.mean(ax[upper])

            upward_length = cy - top_y

            if upward_length >= 35:
                if upper_x < cx - 8:
                    left_arms.append(upward_length)
                elif upper_x > cx + 8:
                    right_arms.append(upward_length)

        # A lower path should extend below junction
        downward_length = bottom_y - cy

        if downward_length >= 10:
            lower_arms.append(downward_length)

    # Candidate needs left, right and lower paths
    if left_arms and right_arms and lower_arms:

        left_length = max(left_arms)
        right_length = max(right_arms)
        lower_length = max(lower_arms)

        # Rank by shortest main arm,
        # with a smaller contribution from trunk length
        score = (
            min(left_length, right_length)
            + 0.3 * lower_length
        )

        junction_results.append([
            cx, cy, score,
            left_length, right_length, lower_length
        ])

# STEP 7: Choose highest scoring candidate
junction_results.sort(
    key=lambda item: item[2], reverse=True
)

print("\n--- MAIN JUNCTION RESULTS ---")
print("Connected foreground components:", count)
print("Junction candidates:", n)
print("Candidates passing path test:", len(junction_results))

if junction_results:

    best = junction_results[0]

    print(f"Estimated junction: x={best[0]}, y={best[1]}")
    print(f"Score: {best[2]:.1f}")
    print(f"Left path vertical extent: {best[3]:.1f} px")
    print(f"Right path vertical extent: {best[4]:.1f} px")
    print(f"Lower path vertical extent: {best[5]:.1f} px")

else:
    best = None
    print("No main junction found")

# STEP 8: Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(depth, cmap="gray")
axes[0].set_title("Original winter depth")

axes[1].imshow(tree_mask, cmap="gray")
axes[1].set_title("Isolated tree foreground")

axes[2].imshow(depth, cmap="gray")

overlay = np.ma.masked_where(~skeleton, skeleton)
axes[2].imshow(
    overlay, cmap="autumn",
    vmin=0, vmax=1, alpha=0.8
)

# Show manually selected trunk
axes[2].scatter(
    trunk_x, trunk_y,
    color="lime", s=75,
    label="Selected trunk"
)

# Show best junction candidate
if best is not None:
    axes[2].scatter(
        best[0], best[1],
        color="cyan", s=130,
        marker="x",
        linewidths=3,
        label="Estimated main junction"
    )

axes[2].set_title("Estimated main junction")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
