
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Select the tree
TREE_ID = "R09N28"
THRESHOLD = 80
SEARCH_RADIUS = 35
ROW_STEP = 5

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = list(data_folder.rglob(f"{TREE_ID}_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Tree image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape

binary = depth >= THRESHOLD


# STEP 2: Select a starting point on the trunk
plt.figure(figsize=(9, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
plt.title("Click once on the visible trunk, below the branch junction")
plt.axis("off")

clicked = plt.ginput(1, timeout=0)
plt.close()

if not clicked:
    raise RuntimeError("No starting point selected")

start_x, start_y = clicked[0]

print(f"Starting position: x={start_x:.1f}, y={start_y:.1f}")


# STEP 3: Find foreground groups on a row
def find_groups(row, centre_x, radius):

    xs = np.where(binary[row])[0]

    # Keep candidates near the current position
    xs = xs[np.abs(xs - centre_x) <= radius]

    if len(xs) == 0:
        return []

    groups = np.split(
        xs, np.where(np.diff(xs) > 1)[0] + 1
    )

    return [
        group for group in groups
        if len(group) >= 2
    ]


# STEP 4: Follow the structure upwards
tracked_points = []

current_x = start_x
start_row = int(start_y)

for y in range(start_row, 19, -ROW_STEP):

    groups = find_groups(
        y, current_x, SEARCH_RADIUS
    )

    if not groups:
        print("Tracking stopped: no pixels at row", y)
        break

    # Choose the group closest to the last position
    best_group = min(
        groups,
        key=lambda g: abs(np.mean(g) - current_x)
    )

    current_x = float(np.mean(best_group))
    tracked_points.append([y, current_x])

tracked_points = np.array(tracked_points).reshape(-1, 2)

print("Tracked points:", len(tracked_points))


# STEP 5: Inspect possible branch separation
# Look for two distinct foreground groups near the track.
junction_candidates = []

for y, x in tracked_points:

    y = int(y)
    groups = find_groups(y, x, 100)

    # Keep substantial groups near the tracked position
    groups = [
        g for g in groups if len(g) >= 3
    ]

    if len(groups) < 2:
        continue

    centres = sorted(float(np.mean(g)) for g in groups)

    # Check whether two separated groups exist
    separations = np.diff(centres)

    if len(separations) and np.max(separations) >= 15:
        junction_candidates.append((x, y))

print("Possible split rows found:", len(junction_candidates))


# STEP 6: Display the tracking result
plt.figure(figsize=(10, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

if len(tracked_points):
    plt.plot(
        tracked_points[:, 1],
        tracked_points[:, 0],
        "cyan",
        linewidth=2,
        label="Tracked foreground path"
    )

plt.scatter(
    start_x, start_y,
    color="lime", s=80,
    label="Selected trunk point"
)

if junction_candidates:
    candidate = junction_candidates[0]

    plt.scatter(
        candidate[0], candidate[1],
        color="orange",
        s=80,
        label="Possible branch split"
    )

plt.title(f"{TREE_ID}: Trunk tracking experiment")
plt.xlim(0, width)
plt.ylim(height, 0)
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()
