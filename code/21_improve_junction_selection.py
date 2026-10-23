
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import (
    binary_closing,
    remove_small_objects,
    skeletonize,
    disk
)

# STEP 1: SETTINGS
TREES = [
    "R20N03",
    "R01N03",
    "R01N43",
    "R09N18",
    "R09N28"
]

THRESHOLD = 80
CLOSING_RADIUS = 8
DISTANCE_WEIGHT = 1.0

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"
results_folder.mkdir(exist_ok=True)

# Reuse the reference coordinates from script 20
reference_file = results_folder / "junction_validation.csv"

if not reference_file.exists():
    raise FileNotFoundError(
        "Run script 20 first to save reference coordinates"
    )

references = pd.read_csv(reference_file)
references = references.set_index("Tree_ID")

# Save manually selected trunk seeds for repeatable tests
seed_file = results_folder / "trunk_seeds.csv"

if seed_file.exists():
    seeds = pd.read_csv(seed_file).set_index("Tree_ID")
else:
    seeds = pd.DataFrame(
        columns=["Trunk_X", "Trunk_Y"]
    )
    seeds.index.name = "Tree_ID"


# STEP 2: GET OR REUSE TRUNK POINT
def get_trunk(tree_id, depth):

    if tree_id in seeds.index:
        return (
            int(seeds.loc[tree_id, "Trunk_X"]),
            int(seeds.loc[tree_id, "Trunk_Y"])
        )

    plt.figure(figsize=(9, 6))
    plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{tree_id}: Click lower trunk")
    plt.axis("off")

    points = plt.ginput(1, timeout=0)
    plt.close()

    if not points:
        raise RuntimeError("No trunk point selected")

    x, y = [int(round(v)) for v in points[0]]

    seeds.loc[tree_id] = [x, y]
    seeds.to_csv(seed_file)

    return x, y


# STEP 3: EXTRACT AND SCORE JUNCTION CANDIDATES
def detect_candidates(depth, trunk_x, trunk_y):

    height, width = depth.shape
    binary = depth >= THRESHOLD

    cutoff = int(height * 0.82)
    binary[cutoff:, :] = False

    cleaned = binary_closing(
        binary,
        footprint=disk(CLOSING_RADIUS)
    )
    cleaned = remove_small_objects(
        cleaned,
        min_size=50
    )

    
    # Allow a wider region for curved and leaning trunks
    lower_start = int(height * 0.65)
    half_width = 180

    for y in range(lower_start, cutoff):
        left = max(0, trunk_x - half_width)
        right = min(width, trunk_x + half_width)

        cleaned[y, :left] = False
        cleaned[y, right:] = False


    structure = np.ones((3, 3), dtype=int)

    labels, _ = ndimage.label(
        cleaned,
        structure=structure
    )

    y1 = max(0, trunk_y - 15)
    y2 = min(height, trunk_y + 16)
    x1 = max(0, trunk_x - 15)
    x2 = min(width, trunk_x + 16)

    nearby = labels[y1:y2, x1:x2]
    nearby = nearby[nearby > 0]

    if len(nearby) == 0:
        return [], np.zeros_like(binary)

    component_id = np.bincount(nearby).argmax()
    skeleton = skeletonize(labels == component_id)

    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0

    neighbours = ndimage.convolve(
        skeleton.astype(int),
        kernel,
        mode="constant",
        cval=0
    )

    Y, X = np.indices(skeleton.shape)

    candidates = skeleton & (neighbours >= 3)
    candidates &= (
        (Y > height * 0.45)
        & (Y < height * 0.80)
        & (np.abs(X - trunk_x) < 110)
    )

    candidate_labels, count = ndimage.label(
        candidates,
        structure=structure
    )

    results = []

    for candidate_id in range(1, count + 1):

        ys, xs = np.where(
            candidate_labels == candidate_id
        )

        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))

        radius = 6

        distance2 = (
            (X - cx)**2 + (Y - cy)**2
        )

        circle = distance2 <= radius**2
        cut = skeleton.copy()
        cut[circle] = False

        arm_labels, _ = ndimage.label(
            cut,
            structure=structure
        )

        ring = (
            (distance2 <= (radius + 2)**2)
            & ~circle
        )

        touching = np.unique(arm_labels[ring])
        touching = touching[touching > 0]

        left_arms = []
        right_arms = []
        lower_arms = []

        for arm_id in touching:

            ay, ax = np.where(arm_labels == arm_id)

            if len(ax) < 10:
                continue

            top_y = ay.min()
            bottom_y = ay.max()

            upper = ay < cy - 15

            if np.any(upper):
                mean_x = ax[upper].mean()
                upward = cy - top_y

                if upward >= 35:
                    if mean_x < cx - 8:
                        left_arms.append(upward)
                    elif mean_x > cx + 8:
                        right_arms.append(upward)

            downward = bottom_y - cy

            if downward >= 10:
                lower_arms.append(downward)

        if not (
            left_arms
            and right_arms
            and lower_arms
        ):
            continue

        left_length = max(left_arms)
        right_length = max(right_arms)
        lower_length = max(lower_arms)

        # Original scoring from script 20
        original_score = (
            min(left_length, right_length)
            + 0.3 * lower_length
        )

        # Distance from manually selected trunk
        distance = np.hypot(
            cx - trunk_x,
            cy - trunk_y
        )

        # New trunk-aware score
        new_score = (
            original_score
            - DISTANCE_WEIGHT * distance
        )

        results.append({
            "X": cx,
            "Y": cy,
            "Original_Score": original_score,
            "Trunk_Distance": distance,
            "New_Score": new_score
        })

    return results, skeleton


# STEP 4: COMPARE BOTH METHODS
results = []

for tree_id in TREES:

    print("\nAnalysing:", tree_id)

    files = sorted(
        data_folder.rglob(
            f"{tree_id}_Winter_D-*.png"
        )
    )

    if not files:
        print("Missing image")
        continue

    depth = np.array(
        Image.open(files[0]).convert("L")
    )

    trunk_x, trunk_y = get_trunk(tree_id, depth)

    candidates, skeleton = detect_candidates(
        depth, trunk_x, trunk_y
    )

    ref_x = float(
        references.loc[tree_id, "Reference_X"]
    )
    ref_y = float(
        references.loc[tree_id, "Reference_Y"]
    )

    original = None
    improved = None

    if candidates:

        original = max(
            candidates,
            key=lambda c: c["Original_Score"]
        )

        improved = max(
            candidates,
            key=lambda c: c["New_Score"]
        )

        print("Candidate count:", len(candidates))

        for c in sorted(
            candidates,
            key=lambda c: c["New_Score"],
            reverse=True
        )[:5]:

            print(
                f"({c['X']}, {c['Y']}) "
                f"Original={c['Original_Score']:.1f}, "
                f"New={c['New_Score']:.1f}"
            )

    else:
        print("No valid junction candidates")

    def error(candidate):
        if candidate is None:
            return np.nan

        return np.hypot(
            candidate["X"] - ref_x,
            candidate["Y"] - ref_y
        )

    original_error = error(original)
    improved_error = error(improved)

    print(f"Original error: {original_error:.2f} px")
    print(f"New error: {improved_error:.2f} px")

    results.append({
        "Tree_ID": tree_id,
        "Original_Error": original_error,
        "New_Error": improved_error,
        "Original_X": original["X"] if original else np.nan,
        "Original_Y": original["Y"] if original else np.nan,
        "New_X": improved["X"] if improved else np.nan,
        "New_Y": improved["Y"] if improved else np.nan
    })

    # STEP 5: VISUAL COMPARISON
    plt.figure(figsize=(9, 6))
    plt.imshow(
        depth, cmap="gray", vmin=0, vmax=255
    )

    overlay = np.ma.masked_where(
        ~skeleton, skeleton
    )
    plt.imshow(
        overlay,
        cmap="autumn",
        vmin=0,
        vmax=1,
        alpha=0.7
    )

    # Show every eligible candidate
    for c in candidates:
        plt.scatter(
            c["X"], c["Y"],
            color="yellow", s=20
        )

    # Reference
    plt.scatter(
        ref_x, ref_y,
        color="lime", s=100,
        label="Manual reference",
        zorder=5
    )

    if original:
        plt.scatter(
            original["X"], original["Y"],
            color="red", marker="x",
            s=120, linewidths=2.5,
            label="Original prediction",
            zorder=6
        )

    if improved:
        plt.scatter(
            improved["X"], improved["Y"],
            facecolors="none",
            edgecolors="cyan",
            marker="o",
            s=160, linewidths=2.5,
            label="Trunk-aware prediction",
            zorder=7
        )

    plt.title(f"{tree_id}: Junction comparison")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        results_folder /
        f"{tree_id}_junction_comparison.png",
        dpi=150
    )
    plt.show()


# STEP 6: SAVE RESULTS
df = pd.DataFrame(results)

df.to_csv(
    results_folder /
    "junction_scoring_comparison.csv",
    index=False
)

print("\n--- SCORING COMPARISON ---")
print(df.round(2).to_string(index=False))

for column in ["Original_Error", "New_Error"]:

    valid = df[column].dropna()

    print(f"\n{column}")
    print(f"Detections: {len(valid)}/{len(df)}")

    if len(valid):
        print(f"Mean error: {valid.mean():.2f} px")
        print(
            f"RMSE: "
            f"{np.sqrt(np.mean(valid**2)):.2f} px"
        )
