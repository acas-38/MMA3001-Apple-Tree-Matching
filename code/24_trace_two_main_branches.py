
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# STEP 1: SETTINGS
TREE_ID = "R01N03"

MIN_PATH_LENGTH = 80
MIN_ENDPOINT_SEPARATION = 60
UPPER_HEIGHT_FRACTION = 0.65

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"
results_folder.mkdir(exist_ok=True)

skeleton_file = (
    results_folder / f"{TREE_ID}_main_skeleton.npy"
)

if not skeleton_file.exists():
    raise FileNotFoundError(
        "Run script 26 first to save the main skeleton"
    )

skeleton = np.load(skeleton_file).astype(bool)

files = sorted(
    data_folder.rglob(f"{TREE_ID}_Winter_D-*.png")
)

if not files:
    raise FileNotFoundError("Winter depth image missing")

depth = np.array(Image.open(files[0]).convert("L"))

height, width = skeleton.shape

if depth.shape != skeleton.shape:
    raise ValueError("Skeleton and depth sizes do not match")


# STEP 2: SELECT LOWER TRUNK
plt.figure(figsize=(9, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

overlay = np.ma.masked_where(~skeleton, skeleton)

plt.imshow(
    overlay,
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.8
)

plt.title("Click on the LOWER TRUNK")
plt.axis("off")

clicked = plt.ginput(1, timeout=0)
plt.close()

if not clicked:
    raise RuntimeError("No trunk selected")

trunk_x, trunk_y = clicked[0]


# STEP 3: CREATE A GRAPH FROM THE SKELETON
ys, xs = np.where(skeleton)

if len(xs) == 0:
    raise RuntimeError("Skeleton is empty")

# Each skeleton pixel becomes a graph node.
pixel_index = np.full(
    skeleton.shape, -1, dtype=int
)

pixel_index[ys, xs] = np.arange(len(xs))

rows = []
cols = []
weights = []

for dy in [-1, 0, 1]:
    for dx in [-1, 0, 1]:

        if dy == 0 and dx == 0:
            continue

        ny = ys + dy
        nx = xs + dx

        valid = (
            (ny >= 0)
            & (ny < height)
            & (nx >= 0)
            & (nx < width)
        )

        sources = np.where(valid)[0]
        targets = pixel_index[
            ny[valid], nx[valid]
        ]

        connected = targets >= 0

        sources = sources[connected]
        targets = targets[connected]

        rows.extend(sources.tolist())
        cols.extend(targets.tolist())

        step = np.hypot(dx, dy)
        weights.extend([step] * len(sources))

graph = csr_matrix(
    (weights, (rows, cols)),
    shape=(len(xs), len(xs))
)


# STEP 4: LOCATE THE TRUNK START
dist2 = (
    (xs - trunk_x)**2
    + (ys - trunk_y)**2
)

start_index = int(np.argmin(dist2))

if dist2[start_index] > 30**2:
    raise RuntimeError(
        "No skeleton within 30 pixels of trunk click"
    )

start_x = int(xs[start_index])
start_y = int(ys[start_index])

# Shortest distances from trunk to every skeleton pixel
distances, predecessors = dijkstra(
    graph,
    directed=False,
    indices=start_index,
    return_predecessors=True
)


# STEP 5: FIND SKELETON ENDPOINTS
kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0

neighbours = ndimage.convolve(
    skeleton.astype(int),
    kernel,
    mode="constant",
    cval=0
)

endpoints = skeleton & (neighbours == 1)

end_y, end_x = np.where(endpoints)
end_indices = pixel_index[end_y, end_x]

# Keep upper endpoints with sufficiently long paths
valid_endpoints = []

for x, y, idx in zip(
    end_x, end_y, end_indices
):

    length = distances[idx]

    if not np.isfinite(length):
        continue

    if y >= height * UPPER_HEIGHT_FRACTION:
        continue

    if length < MIN_PATH_LENGTH:
        continue

    valid_endpoints.append({
        "x": int(x),
        "y": int(y),
        "index": int(idx),
        "length": float(length)
    })

print("\n--- ENDPOINT ANALYSIS ---")
print("Total endpoints:", len(end_x))
print("Valid upper endpoints:", len(valid_endpoints))

if len(valid_endpoints) < 2:
    raise RuntimeError(
        "Not enough upper endpoints for two-branch tracing"
    )


# STEP 6: SELECT TWO SEPARATED UPPER ENDPOINTS
# Choose a pair with long paths and good horizontal separation.
pairs = []

for i in range(len(valid_endpoints)):
    for j in range(i + 1, len(valid_endpoints)):

        a = valid_endpoints[i]
        b = valid_endpoints[j]

        separation = abs(a["x"] - b["x"])

        if separation < MIN_ENDPOINT_SEPARATION:
            continue

        # Prefer two long paths rather than one long,
        # one very short path.
        score = (
            min(a["length"], b["length"])
            + 0.25 * separation
        )

        pairs.append((score, a, b))

if not pairs:
    raise RuntimeError(
        "No endpoint pair has sufficient separation"
    )

pairs.sort(
    key=lambda item: item[0],
    reverse=True
)

_, endpoint_a, endpoint_b = pairs[0]

# Order by left/right image position
if endpoint_a["x"] > endpoint_b["x"]:
    endpoint_a, endpoint_b = endpoint_b, endpoint_a

print("\nSelected left endpoint:", endpoint_a)
print("Selected right endpoint:", endpoint_b)


# STEP 7: TRACE EACH PATH BACK TO THE TRUNK
def trace_path(target_index):

    path = []
    current = int(target_index)

    while current != start_index:

        if current < 0:
            raise RuntimeError("Path is disconnected")

        path.append(current)

        current = int(predecessors[current])

    path.append(start_index)

    # Return path in trunk-to-endpoint order
    path.reverse()

    return path


left_path = trace_path(endpoint_a["index"])
right_path = trace_path(endpoint_b["index"])


# STEP 8: IDENTIFY SHARED TRUNK AND JUNCTION
shared_count = 0

for left_node, right_node in zip(
    left_path, right_path
):

    if left_node == right_node:
        shared_count += 1
    else:
        break

if shared_count == 0:
    raise RuntimeError("No shared path found")

# Last pixel common to both paths
junction_index = left_path[shared_count - 1]

junction_x = int(xs[junction_index])
junction_y = int(ys[junction_index])

# Split into three path sections
trunk_path = left_path[:shared_count]

left_branch = left_path[shared_count - 1:]
right_branch = right_path[shared_count - 1:]


# STEP 9: CALCULATE PATH LENGTHS
def calculate_length(path):

    if len(path) < 2:
        return 0.0

    path_x = xs[path]
    path_y = ys[path]

    dx = np.diff(path_x)
    dy = np.diff(path_y)

    return float(
        np.sum(np.hypot(dx, dy))
    )


trunk_length = calculate_length(trunk_path)
left_length = calculate_length(left_branch)
right_length = calculate_length(right_branch)

print("\n--- MAIN JUNCTION RESULTS ---")
print("Estimated junction:", junction_x, junction_y)
print(f"Shared trunk length: {trunk_length:.2f} px")
print(f"Left branch length: {left_length:.2f} px")
print(f"Right branch length: {right_length:.2f} px")


# STEP 10: PLOT ALL THREE PATHS
def plot_path(ax, path, colour, label, linewidth=3):

    if not path:
        return

    ax.plot(
        xs[path],
        ys[path],
        color=colour,
        linewidth=linewidth,
        label=label
    )


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original connected skeleton
axes[0].imshow(
    depth, cmap="gray", vmin=0, vmax=255
)
axes[0].imshow(
    overlay,
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.8
)
axes[0].set_title("Original connected skeleton")

# Three traced paths
axes[1].imshow(
    depth, cmap="gray", vmin=0, vmax=255
)

plot_path(
    axes[1], trunk_path, "limegreen", "Shared trunk"
)
plot_path(
    axes[1], left_branch, "orange", "Left branch"
)
plot_path(
    axes[1], right_branch, "cyan", "Right branch"
)

# Trunk starting point
axes[1].scatter(
    start_x, start_y,
    color="lime", s=100,
    label="Trunk start",
    zorder=5
)

# Estimated junction
axes[1].scatter(
    junction_x, junction_y,
    color="red",
    marker="x",
    s=150,
    linewidths=3,
    label="Estimated junction",
    zorder=6
)

# Selected upper endpoints
axes[1].scatter(
    endpoint_a["x"], endpoint_a["y"],
    color="orange", s=80
)
axes[1].scatter(
    endpoint_b["x"], endpoint_b["y"],
    color="cyan", s=80
)

axes[1].set_title("Two main branches and shared trunk")
axes[1].legend(fontsize=8)

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    results_folder /
    f"{TREE_ID}_two_branch_trace.png",
    dpi=150
)

plt.show()


# STEP 11: SAVE RESULTS
import pandas as pd

summary = pd.DataFrame([{
    "Tree_ID": TREE_ID,
    "Junction_X": junction_x,
    "Junction_Y": junction_y,
    "Trunk_Length": trunk_length,
    "Left_Length": left_length,
    "Right_Length": right_length,
    "Left_Endpoint_X": endpoint_a["x"],
    "Left_Endpoint_Y": endpoint_a["y"],
    "Right_Endpoint_X": endpoint_b["x"],
    "Right_Endpoint_Y": endpoint_b["y"]
}])

summary.to_csv(
    results_folder /
    f"{TREE_ID}_two_branch_trace.csv",
    index=False
)

print("\nResults saved.")
