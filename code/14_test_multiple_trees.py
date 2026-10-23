
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Locate the dataset
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
output_folder = project / "results"
output_folder.mkdir(exist_ok=True)

trees = [
    "R01N03",
    "R20N03",
    "R01N43",
    "R09N18",
    "R09N28"
]

THRESHOLD = 80
results = []


# STEP 2: Find points along a possible branch
def find_point(binary, row, expected_x, radius):
    xs = np.where(binary[row])[0]
    xs = xs[np.abs(xs - expected_x) < radius]

    if len(xs) == 0:
        return np.nan

    groups = np.split(
        xs, np.where(np.diff(xs) > 1)[0] + 1
    )
    groups = [g for g in groups if len(g) >= 2]

    if not groups:
        return np.nan

    best = min(
        groups,
        key=lambda g: abs(np.mean(g) - expected_x)
    )

    return float(np.mean(best))


# STEP 3: Fit and validate a polynomial
def fit_and_validate(points, degree):
    if len(points) < 6:
        return np.nan, None

    indices = np.arange(len(points))
    test_mask = indices % 5 == 0

    train = points[~test_mask]
    test = points[test_mask]

    if len(train) < degree + 1:
        return np.nan, None

    coefficients = np.polyfit(
        train[:, 0], train[:, 1], degree
    )

    predictions = np.polyval(
        coefficients, test[:, 0]
    )

    rmse = np.sqrt(
        np.mean((test[:, 1] - predictions) ** 2)
    )

    return float(rmse), coefficients


# STEP 4: Analyse each tree
for tree_id in trees:

    files = list(
        data_folder.rglob(f"{tree_id}_Winter_D-*.png")
    )

    if not files:
        print("Missing image:", tree_id)
        results.append([tree_id, 0, 0, np.nan, np.nan])
        continue

    depth = np.array(Image.open(files[0]).convert("L"))
    height, width = depth.shape
    binary = depth >= THRESHOLD

    sample_rows = np.arange(20, 290, 10)

    # Same search settings as file 13
    left_top = 0.18 * width
    left_bottom = 0.35 * width
    right_top = 0.68 * width
    right_bottom = 0.60 * width
    radius = 0.10 * width

    left_points = []
    right_points = []

    for y in sample_rows:
        fraction = (y - 20) / (290 - 20)

        expected_left = left_top + (
            left_bottom - left_top
        ) * fraction

        expected_right = right_top + (
            right_bottom - right_top
        ) * fraction

        xl = find_point(binary, y, expected_left, radius)
        xr = find_point(binary, y, expected_right, radius)

        if np.isfinite(xl):
            left_points.append([y, xl])

        if np.isfinite(xr):
            right_points.append([y, xr])

    left_points = np.array(left_points).reshape(-1, 2)
    right_points = np.array(right_points).reshape(-1, 2)

    # Validate quadratic fits
    left_rmse, left_coeff = fit_and_validate(
        left_points, 2
    )
    right_rmse, right_coeff = fit_and_validate(
        right_points, 2
    )

    results.append([
        tree_id,
        len(left_points),
        len(right_points),
        left_rmse,
        right_rmse
    ])

    print(f"\nTree: {tree_id}")
    print("Left points:", len(left_points))
    print("Right points:", len(right_points))
    print(f"Left RMSE: {left_rmse:.2f} px")
    print(f"Right RMSE: {right_rmse:.2f} px")

    # STEP 5: Visualise each tree
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.imshow(depth, cmap="gray", vmin=0, vmax=255)

    for points, coeff, colour, label in [
        (left_points, left_coeff, "orange", "Left"),
        (right_points, right_coeff, "cyan", "Right")
    ]:

        if len(points) > 0:
            ax.scatter(
                points[:, 1], points[:, 0],
                color=colour, s=20,
                label=f"{label} points"
            )

        if coeff is not None:
            yy = np.linspace(
                points[:, 0].min(),
                points[:, 0].max(), 150
            )
            xx = np.polyval(coeff, yy)

            ax.plot(xx, yy, color=colour, linewidth=2)

    ax.set_title(f"{tree_id}: Branch detection")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        output_folder / f"{tree_id}_branch_detection.png",
        dpi=150
    )
    plt.show()


# STEP 6: Save the numerical results
df = pd.DataFrame(
    results,
    columns=[
        "Tree_ID",
        "Left_Points",
        "Right_Points",
        "Left_RMSE",
        "Right_RMSE"
    ]
)

df.to_csv(
    output_folder / "multiple_tree_validation.csv",
    index=False
)

print("\n--- MULTIPLE TREE RESULTS ---")
print(df.to_string(index=False))
