
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

# STEP 2: Threshold the image
threshold = 80
binary = depth >= threshold

# We initially analyse the upper and middle branches.
# The ground causes more interference near the bottom.
sample_rows = np.arange(20, 290, 10)

# Approximate search positions, expressed as fractions
# of image width. These are starting assumptions.
left_top = 0.18 * width
left_bottom = 0.35 * width

right_top = 0.68 * width
right_bottom = 0.60 * width

search_radius = 0.10 * width


# STEP 3: Find a candidate branch point in one row
def find_point(row, expected_x):

    # Find white pixels near the expected branch
    xs = np.where(binary[row])[0]

    xs = xs[
        np.abs(xs - expected_x) < search_radius
    ]

    if len(xs) == 0:
        return np.nan

    # Divide the pixels into consecutive groups
    groups = np.split(
        xs, np.where(np.diff(xs) > 1)[0] + 1
    )

    # Ignore isolated single-pixel detections
    groups = [g for g in groups if len(g) >= 2]

    if not groups:
        return np.nan

    # Select the group closest to the expected position
    best = min(
        groups,
        key=lambda g: abs(np.mean(g) - expected_x)
    )

    return float(np.mean(best))


# STEP 4: Detect left and right branch points
left_points = []
right_points = []

for y in sample_rows:

    fraction = (y - 20) / (290 - 20)

    expected_left = (
        left_top +
        (left_bottom - left_top) * fraction
    )

    expected_right = (
        right_top +
        (right_bottom - right_top) * fraction
    )

    x_left = find_point(y, expected_left)
    x_right = find_point(y, expected_right)

    if np.isfinite(x_left):
        left_points.append([y, x_left])

    if np.isfinite(x_right):
        right_points.append([y, x_right])

left_points = np.array(left_points).reshape(-1, 2)
right_points = np.array(right_points).reshape(-1, 2)

print("Left branch points:", len(left_points))
print("Right branch points:", len(right_points))


# STEP 5: Fit straight and quadratic curves
def fit_branch(points, degree):

    if len(points) < degree + 1:
        raise ValueError("Not enough branch points")

    y = points[:, 0]
    x = points[:, 1]

    coefficients = np.polyfit(y, x, degree)

    predicted_x = np.polyval(coefficients, y)

    rmse = np.sqrt(
        np.mean((x - predicted_x) ** 2)
    )

    return coefficients, rmse


if len(left_points) < 3 or len(right_points) < 3:
    raise ValueError(
        "Too few points detected. Inspect search parameters."
    )

left_linear, left_linear_rmse = fit_branch(
    left_points, 1
)

right_linear, right_linear_rmse = fit_branch(
    right_points, 1
)

left_quad, left_quad_rmse = fit_branch(
    left_points, 2
)

right_quad, right_quad_rmse = fit_branch(
    right_points, 2
)

print("\n--- CURVE FITTING RESULTS ---")
print(f"Left linear RMSE: {left_linear_rmse:.2f} px")
print(f"Left quadratic RMSE: {left_quad_rmse:.2f} px")
print(f"Right linear RMSE: {right_linear_rmse:.2f} px")
print(f"Right quadratic RMSE: {right_quad_rmse:.2f} px")


# STEP 6: Display detected points and fitted curves
y_curve = np.linspace(sample_rows.min(), sample_rows.max(), 250)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(depth, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original winter depth")

# Selected branch points
axes[1].imshow(depth, cmap="gray", vmin=0, vmax=255)

axes[1].scatter(
    left_points[:, 1], left_points[:, 0],
    color="orange", s=25, label="Left points"
)

axes[1].scatter(
    right_points[:, 1], right_points[:, 0],
    color="cyan", s=25, label="Right points"
)

axes[1].set_title("Detected branch points")
axes[1].legend(fontsize=8)

# Quadratic curve fits
axes[2].imshow(depth, cmap="gray", vmin=0, vmax=255)

axes[2].plot(
    np.polyval(left_quad, y_curve),
    y_curve,
    color="orange", linewidth=2,
    label="Left quadratic"
)

axes[2].plot(
    np.polyval(right_quad, y_curve),
    y_curve,
    color="cyan", linewidth=2,
    label="Right quadratic"
)

axes[2].set_title("Quadratic curve fits")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

plt.tight_layout()
plt.show()


# STEP 7: Compare linear and quadratic fits
plt.figure(figsize=(7, 4))

labels = [
    "Left linear",
    "Left quadratic",
    "Right linear",
    "Right quadratic"
]

errors = [
    left_linear_rmse,
    left_quad_rmse,
    right_linear_rmse,
    right_quad_rmse
]

plt.bar(labels, errors)
plt.ylabel("RMSE (pixels)")
plt.title("Curve Fitting Error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
