
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Load winter depth image
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

tree_id = "R01N01"

files = list(data_folder.rglob(f"{tree_id}_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Winter depth image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape

# STEP 2: Original detection settings
threshold = 80
binary = depth >= threshold

sample_rows = np.arange(20, 290, 10)

left_top = 0.18 * width
left_bottom = 0.35 * width

right_top = 0.68 * width
right_bottom = 0.60 * width

search_radius = 0.10 * width


# STEP 3: Detect branch points
# Same method as file 12
def find_point(row, expected_x):

    xs = np.where(binary[row])[0]
    xs = xs[np.abs(xs - expected_x) < search_radius]

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

    x_left = find_point(y, expected_left)
    x_right = find_point(y, expected_right)

    if np.isfinite(x_left):
        left_points.append([y, x_left])

    if np.isfinite(x_right):
        right_points.append([y, x_right])

left_points = np.array(left_points).reshape(-1, 2)
right_points = np.array(right_points).reshape(-1, 2)


# STEP 4: Training and validation
def validate_branch(points, degree):

    # Sort points vertically
    points = points[np.argsort(points[:, 0])]

    if len(points) < 6:
        raise ValueError("Not enough detected points")

    # Hold out approximately every fifth point
    indices = np.arange(len(points))
    test_mask = indices % 5 == 0

    train = points[~test_mask]
    test = points[test_mask]

    # Fit using training points only
    coefficients = np.polyfit(
        train[:, 0],
        train[:, 1],
        degree
    )

    # Predict positions of held-out points
    predicted = np.polyval(
        coefficients,
        test[:, 0]
    )

    # Calculate validation RMSE
    rmse = np.sqrt(
        np.mean((test[:, 1] - predicted) ** 2)
    )

    return coefficients, rmse, train, test


# STEP 5: Compare linear and quadratic fits
results = {}

for name, points in [
    ("Left", left_points),
    ("Right", right_points)
]:

    linear = validate_branch(points, 1)
    quadratic = validate_branch(points, 2)

    results[name] = {
        "linear": linear,
        "quadratic": quadratic
    }

    print(f"\n{name} branch")
    print("Total detected points:", len(points))
    print("Training points:", len(linear[2]))
    print("Validation points:", len(linear[3]))
    print(f"Linear validation RMSE: {linear[1]:.2f} px")
    print(f"Quadratic validation RMSE: {quadratic[1]:.2f} px")


# STEP 6: Visualise validation
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, name in zip(axes, ["Left", "Right"]):

    linear = results[name]["linear"]
    quadratic = results[name]["quadratic"]

    train = linear[2]
    test = linear[3]

    y_curve = np.linspace(
        train[:, 0].min(),
        train[:, 0].max(),
        200
    )

    ax.imshow(depth, cmap="gray", vmin=0, vmax=255)

    # Training and validation points
    ax.scatter(
        train[:, 1], train[:, 0],
        color="orange", s=30,
        label="Training points"
    )

    ax.scatter(
        test[:, 1], test[:, 0],
        color="lime", s=55,
        marker="x",
        label="Validation points"
    )

    # Linear curve
    ax.plot(
        np.polyval(linear[0], y_curve),
        y_curve,
        color="yellow",
        linewidth=2,
        label="Linear fit"
    )

    # Quadratic curve
    ax.plot(
        np.polyval(quadratic[0], y_curve),
        y_curve,
        color="cyan",
        linewidth=2,
        label="Quadratic fit"
    )

    ax.set_title(f"{name} branch")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()


# STEP 7: Compare validation errors
labels = [
    "Left linear",
    "Left quadratic",
    "Right linear",
    "Right quadratic"
]

errors = [
    results["Left"]["linear"][1],
    results["Left"]["quadratic"][1],
    results["Right"]["linear"][1],
    results["Right"]["quadratic"][1]
]

plt.figure(figsize=(8, 5))
plt.bar(labels, errors)
plt.ylabel("Validation RMSE (pixels)")
plt.title("Linear vs Quadratic: Held-out Points")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
