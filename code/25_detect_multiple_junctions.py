
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ==================================================
# 1. SETTINGS
# ==================================================

TREE_ID = "R20N03"

MIN_PATH_LENGTH = 70
MIN_ENDPOINT_SEPARATION = 30
MIN_BRANCH_LENGTH = 25
MERGE_DISTANCE = 12

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"
results_folder = project / "results"

results_folder.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD EXISTING DATA
# ==================================================

files = sorted(
    data_folder.rglob(f"{TREE_ID}_Winter_D-*.png")
)

if not files:
    raise FileNotFoundError(
        f"No winter depth image found for {TREE_ID}"
    )

depth = np.array(
    Image.open(files[0]).convert("L")
)

skeleton_file = (
    results_folder / f"{TREE_ID}_main_skeleton.npy"
)

if not skeleton_file.exists():
    raise FileNotFoundError(
        "Run 23_trace_tree_skeleton.py first"
    )

skeleton = np.load(skeleton_file).astype(bool)

height, width = skeleton.shape

if depth.shape != skeleton.shape:
    raise ValueError(
        "Depth image and skeleton dimensions do not match"
    )


# ==================================================
# 3. SELECT LOWER TRUNK
# ==================================================

plt.figure(figsize=(9, 6))

plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

overlay = np.ma.masked_where(
    ~skeleton, skeleton
)

plt.imshow(
    overlay,
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.8
)

plt.title("Click the LOWER TRUNK")
plt.axis("off")

clicked = plt.ginput(1, timeout=0)
plt.close()

if not clicked:
    raise RuntimeError("No trunk point selected")

trunk_x, trunk_y = clicked[0]


# ==================================================
# 4. CONVERT SKELETON TO GRAPH
# ==================================================

ys, xs = np.where(skeleton)

if len(xs) == 0:
    raise RuntimeError("Skeleton contains no pixels")

pixel_index = np.full(
    skeleton.shape,
    -1,
    dtype=int
)

pixel_index[ys, xs] = np.arange(len(xs))

rows = []
cols = []
weights = []

for dy in [-1, 0, 1]:
    for dx in [-1, 0, 1]:

        if dx == 0 and dy == 0:
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
            ny[valid],
            nx[valid]
        ]

        connected = targets >= 0

        sources = sources[connected]
        targets = targets[connected]

        rows.extend(sources.tolist())
        cols.extend(targets.tolist())

        step = np.hypot(dx, dy)

        weights.extend(
            [step] * len(sources)
        )

graph = csr_matrix(
    (weights, (rows, cols)),
    shape=(len(xs), len(xs))
)


# ==================================================
# 5. FIND TRUNK START
# ==================================================

dist2 = (
    (xs - trunk_x)**2
    + (ys - trunk_y)**2
)

start_index = int(np.argmin(dist2))

if dist2[start_index] > 30**2:
    raise RuntimeError(
        "No skeleton within 30 pixels of trunk selection"
    )

start_x = int(xs[start_index])
start_y = int(ys[start_index])

distances, predecessors = dijkstra(
    graph,
    directed=False,
    indices=start_index,
    return_predecessors=True
)


# ==================================================
# 6. FIND SKELETON ENDPOINTS
# ==================================================

kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0

neighbours = ndimage.convolve(
    skeleton.astype(int),
    kernel,
    mode="constant",
    cval=0
)

endpoint_mask = (
    skeleton & (neighbours == 1)
)

end_y, end_x = np.where(endpoint_mask)

valid_endpoints = []

for x, y in zip(end_x, end_y):

    idx = pixel_index[y, x]
    distance = distances[idx]

    if not np.isfinite(distance):
        continue

    if distance < MIN_PATH_LENGTH:
        continue

    # Endpoints must be above the trunk
    if y >= start_y - 35:
        continue

    valid_endpoints.append({
        "x": int(x),
        "y": int(y),
        "index": int(idx),
        "distance": float(distance)
    })


# ==================================================
# 7. FILTER NEARBY ENDPOINTS
# ==================================================

# Prefer endpoints with longer paths
valid_endpoints.sort(
    key=lambda e: e["distance"],
    reverse=True
)

selected_endpoints = []

for endpoint in valid_endpoints:

    too_close = False

    for selected in selected_endpoints:

        separation = np.hypot(
            endpoint["x"] - selected["x"],
            endpoint["y"] - selected["y"]
        )

        if separation < MIN_ENDPOINT_SEPARATION:
            too_close = True
            break

    if not too_close:
        selected_endpoints.append(endpoint)


print("\n--- ENDPOINT RESULTS ---")
print("Total skeleton endpoints:", len(end_x))
print("Valid upper endpoints:", len(valid_endpoints))
print("Selected endpoints:", len(selected_endpoints))


# ==================================================
# 8. TRACE PATHS FROM TRUNK TO ENDPOINTS
# ==================================================

def trace_path(target_index):

    path = []
    current = int(target_index)

    while current != start_index:

        if current < 0:
            return []

        path.append(current)

        current = int(predecessors[current])

    path.append(start_index)
    path.reverse()

    return path


paths = []

for endpoint in selected_endpoints:

    path = trace_path(endpoint["index"])

    if path:
        paths.append({
            "endpoint": endpoint,
            "path": path
        })


# ==================================================
# 9. FIND BRANCH POINTS FROM PATH DIVERGENCE
# ==================================================

# Each endpoint path is compared against every other.
# Their last shared point is a candidate junction.

junction_candidates = []

for i in range(len(paths)):
    for j in range(i + 1, len(paths)):

        path_a = paths[i]["path"]
        path_b = paths[j]["path"]

        shared = 0

        for a, b in zip(path_a, path_b):

            if a == b:
                shared += 1
            else:
                break

        if shared == 0:
            continue

        junction_idx = path_a[shared - 1]

        # Length of each path after divergence
        branch_a = path_a[shared - 1:]
        branch_b = path_b[shared - 1:]

        def path_length(path):

            if len(path) < 2:
                return 0.0

            dx = np.diff(xs[path])
            dy = np.diff(ys[path])

            return float(
                np.sum(np.hypot(dx, dy))
            )

        length_a = path_length(branch_a)
        length_b = path_length(branch_b)

        if min(length_a, length_b) < MIN_BRANCH_LENGTH:
            continue

        junction_candidates.append({
            "x": int(xs[junction_idx]),
            "y": int(ys[junction_idx]),
            "length_a": length_a,
            "length_b": length_b
        })


# ==================================================
# 10. MERGE NEARBY JUNCTION CANDIDATES
# ==================================================

# Group close candidates to avoid multiple detections
# of the same physical junction.

groups = []

for candidate in junction_candidates:

    matching_group = None

    for group in groups:

        centre_x = np.mean([c["x"] for c in group])
        centre_y = np.mean([c["y"] for c in group])

        distance = np.hypot(
            candidate["x"] - centre_x,
            candidate["y"] - centre_y
        )

        if distance < MERGE_DISTANCE:
            matching_group = group
            break

    if matching_group is None:
        groups.append([candidate])
    else:
        matching_group.append(candidate)


junctions = []

for group in groups:

    junctions.append({
        "x": int(round(np.mean([c["x"] for c in group]))),
        "y": int(round(np.mean([c["y"] for c in group]))),
        "support": len(group)
    })

# Order from lower to higher junction
junctions.sort(
    key=lambda j: j["y"],
    reverse=True
)


# ==================================================
# 11. PRINT RESULTS
# ==================================================

print("\n--- MULTIPLE JUNCTION RESULTS ---")
print("Tree:", TREE_ID)
print("Endpoint paths traced:", len(paths))
print("Junction candidates:", len(junction_candidates))
print("Merged junctions:", len(junctions))

for i, junction in enumerate(junctions, start=1):

    print(
        f"Junction {i}: "
        f"x={junction['x']}, "
        f"y={junction['y']}, "
        f"support={junction['support']}"
    )


# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

# Original connected skeleton
axes[0].imshow(
    depth,
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[0].imshow(
    overlay,
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.8
)

axes[0].set_title("Original connected skeleton")

# Traced branches and junctions
axes[1].imshow(
    depth,
    cmap="gray",
    vmin=0,
    vmax=255
)

# Colour each endpoint path
colours = plt.cm.tab10(
    np.linspace(0, 1, max(len(paths), 1))
)

for i, item in enumerate(paths):

    path = item["path"]

    axes[1].plot(
        xs[path],
        ys[path],
        color=colours[i],
        linewidth=2,
        alpha=0.8
    )

# Starting trunk point
axes[1].scatter(
    start_x,
    start_y,
    color="lime",
    s=100,
    label="Trunk start",
    zorder=5
)

# Selected endpoints
for endpoint in selected_endpoints:

    axes[1].scatter(
        endpoint["x"],
        endpoint["y"],
        color="orange",
        s=50,
        zorder=5
    )

# Main junctions
for i, junction in enumerate(junctions, start=1):

    x = junction["x"]
    y = junction["y"]

    axes[1].scatter(
        x,
        y,
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        zorder=6
    )

    axes[1].text(
        x + 7,
        y - 7,
        f"J{i}",
        color="red",
        fontsize=11,
        weight="bold"
    )

axes[1].set_title("Multiple branch paths and junctions")
axes[1].legend()

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    results_folder /
    f"{TREE_ID}_multiple_junctions.png",
    dpi=150
)

plt.show()


# ==================================================
# 13. SAVE RESULTS
# ==================================================

df = pd.DataFrame(junctions)

if df.empty:
    df = pd.DataFrame(
        columns=["x", "y", "support"]
    )

df.insert(0, "Tree_ID", TREE_ID)

df.to_csv(
    results_folder /
    f"{TREE_ID}_multiple_junctions.csv",
    index=False
)

print("\nResults saved.")
