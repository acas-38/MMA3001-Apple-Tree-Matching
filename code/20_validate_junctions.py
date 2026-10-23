
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import (
    binary_closing, remove_small_objects,
    skeletonize, disk
)

# STEP 1: Settings
TREES = [
    "R20N03", "R01N03", "R01N43",
    "R09N18", "R09N28"
]

THRESHOLD = 80
CLOSING_RADIUS = 8

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"
results_folder.mkdir(exist_ok=True)

results = []


# STEP 2: Select a reference point manually
def select_point(depth, title):
    plt.figure(figsize=(9, 6))
    plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

    points = plt.ginput(1, timeout=0)
    plt.close()

    if not points:
        return None

    return points[0]


# STEP 3: Detect the main junction
def detect_junction(depth, trunk_x, trunk_y):
    height, width = depth.shape

    binary = depth >= THRESHOLD
    cutoff = int(height * 0.82)
    binary[cutoff:, :] = False

    cleaned = binary_closing(
        binary, footprint=disk(CLOSING_RADIUS)
    )
    cleaned = remove_small_objects(
        cleaned, min_size=50
    )

    # Restrict ground region around selected trunk
    lower_start = int(height * 0.65)
    half_width = 65

    for y in range(lower_start, cutoff):
        left = max(0, trunk_x - half_width)
        right = min(width, trunk_x + half_width)
        cleaned[y, :left] = False
        cleaned[y, right:] = False

    structure = np.ones((3, 3), dtype=int)
    labels, _ = ndimage.label(
        cleaned, structure=structure
    )

    # Find component associated with trunk
    y1 = max(0, trunk_y - 15)
    y2 = min(height, trunk_y + 16)
    x1 = max(0, trunk_x - 15)
    x2 = min(width, trunk_x + 16)

    nearby = labels[y1:y2, x1:x2]
    nearby = nearby[nearby > 0]

    if len(nearby) == 0:
        return None, np.zeros_like(binary)

    component_id = np.bincount(nearby).argmax()
    tree_mask = labels == component_id
    skeleton = skeletonize(tree_mask)

    # Find skeleton junction candidates
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0

    neighbours = ndimage.convolve(
        skeleton.astype(int),
        kernel, mode="constant", cval=0
    )

    Y, X = np.indices(skeleton.shape)

    candidates = skeleton & (neighbours >= 3)
    candidates &= (
        (Y > height * 0.45) &
        (Y < height * 0.80) &
        (np.abs(X - trunk_x) < 110)
    )

    candidate_labels, n = ndimage.label(
        candidates, structure=structure
    )

    junction_results = []

    # STEP 4: Score candidate junctions
    for label_id in range(1, n + 1):
        ys, xs = np.where(candidate_labels == label_id)

        cx = int(round(np.mean(xs)))
        cy = int(round(np.mean(ys)))

        radius = 6
        distance2 = (X - cx)**2 + (Y - cy)**2

        local_circle = distance2 <= radius**2
        cut = skeleton.copy()
        cut[local_circle] = False

        arm_labels, _ = ndimage.label(
            cut, structure=structure
        )

        ring = (
            (distance2 <= (radius + 2)**2)
            & ~local_circle
        )

        touching_ids = np.unique(arm_labels[ring])
        touching_ids = touching_ids[touching_ids > 0]

        left_arms = []
        right_arms = []
        lower_arms = []

        for arm_id in touching_ids:
            ay, ax = np.where(arm_labels == arm_id)

            if len(ax) < 10:
                continue

            top_y = np.min(ay)
            bottom_y = np.max(ay)
            upper = ay < cy - 15

            if np.any(upper):
                upper_x = np.mean(ax[upper])
                upward_length = cy - top_y

                if upward_length >= 35:
                    if upper_x < cx - 8:
                        left_arms.append(upward_length)
                    elif upper_x > cx + 8:
                        right_arms.append(upward_length)

            downward_length = bottom_y - cy

            if downward_length >= 10:
                lower_arms.append(downward_length)

        if left_arms and right_arms and lower_arms:
            score = (
                min(max(left_arms), max(right_arms))
                + 0.3 * max(lower_arms)
            )

            junction_results.append((cx, cy, score))

    if not junction_results:
        return None, skeleton

    best = max(junction_results, key=lambda r: r[2])
    return (best[0], best[1]), skeleton


# STEP 5: Validate each tree
for tree_id in TREES:
    print("\nAnalysing:", tree_id)

    files = sorted(
        data_folder.rglob(f"{tree_id}_Winter_D-*.png")
    )

    if not files:
        print("Image missing:", tree_id)
        results.append([tree_id, "Missing image",
                        np.nan, np.nan, np.nan,
                        np.nan, np.nan])
        continue

    depth = np.array(Image.open(files[0]).convert("L"))

    # First click lower trunk
    trunk = select_point(
        depth,
        f"{tree_id}: Click the trunk BELOW the junction"
    )

    if trunk is None:
        raise RuntimeError("Trunk selection cancelled")

    trunk_x, trunk_y = [int(round(v)) for v in trunk]

    # Next click actual junction BEFORE viewing prediction
    reference = select_point(
        depth,
        f"{tree_id}: Click the MAIN branch junction"
    )

    if reference is None:
        raise RuntimeError("Reference selection cancelled")

    ref_x, ref_y = reference

    prediction, skeleton = detect_junction(
        depth, trunk_x, trunk_y
    )

    # Calculate positional error
    if prediction is not None:
        pred_x, pred_y = prediction

        error = np.hypot(
            pred_x - ref_x,
            pred_y - ref_y
        )

        status = "Detected"
        print(f"Junction error: {error:.2f} pixels")
    else:
        pred_x, pred_y = np.nan, np.nan
        error = np.nan
        status = "Failed"
        print("Junction detection failed")

    results.append([
        tree_id, status,
        ref_x, ref_y,
        pred_x, pred_y, error
    ])

    # Display estimated and reference junctions
    plt.figure(figsize=(9, 6))
    plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

    overlay = np.ma.masked_where(~skeleton, skeleton)
    plt.imshow(
        overlay, cmap="autumn",
        vmin=0, vmax=1, alpha=0.7
    )

    plt.scatter(
        ref_x, ref_y, color="lime",
        marker="o", s=90, label="Manual reference"
    )

    if prediction is not None:
        plt.scatter(
            pred_x, pred_y, color="cyan",
            marker="x", s=120,
            linewidths=3, label="Prediction"
        )

    plt.title(f"{tree_id}: {status}")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        results_folder / f"{tree_id}_junction_validation.png",
        dpi=150
    )
    plt.show()


# STEP 6: Save and summarise results
df = pd.DataFrame(
    results,
    columns=[
        "Tree_ID", "Status",
        "Reference_X", "Reference_Y",
        "Predicted_X", "Predicted_Y",
        "Error_Pixels"
    ]
)

df.to_csv(
    results_folder / "junction_validation.csv",
    index=False
)

valid = df["Error_Pixels"].dropna()

print("\n--- JUNCTION VALIDATION RESULTS ---")
print(df.round(2).to_string(index=False))
print("Successful detections:", len(valid), "/", len(df))

if len(valid):
    print(f"Mean error: {valid.mean():.2f} px")
    print(f"RMSE: {np.sqrt(np.mean(valid**2)):.2f} px")
    print(f"Maximum error: {valid.max():.2f} px")
