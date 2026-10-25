
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# STEP 1: SETTINGS
TREE_ID = "R01N03"

project = Path(__file__).resolve().parent.parent
results_folder = project / "results"
data_folder = project / "Data"

skeleton_file = (
    results_folder / f"{TREE_ID}_refined_skeleton.npy"
)

if not skeleton_file.exists():
    raise FileNotFoundError(
        "Run script 25 first to create the skeleton"
    )

skeleton = np.load(skeleton_file).astype(bool)

files = sorted(
    data_folder.rglob(f"{TREE_ID}_Winter_D-*.png")
)

if not files:
    raise FileNotFoundError("Winter depth image missing")

from PIL import Image

depth = np.array(Image.open(files[0]).convert("L"))

if skeleton.shape != depth.shape:
    raise ValueError("Skeleton and image sizes do not match")

height, width = skeleton.shape

# STEP 2: CONNECTED COMPONENTS
# Use 8-connectivity, including diagonal neighbours
structure = np.ones((3, 3), dtype=int)

labels, component_count = ndimage.label(
    skeleton, structure=structure
)

sizes = np.bincount(labels.ravel())
sizes[0] = 0

print("\n--- SKELETON COMPONENTS ---")
print("Number of components:", component_count)

largest_ids = np.argsort(sizes)[::-1][:10]

for component_id in largest_ids:
    if sizes[component_id] > 0:
        print(
            f"Component {component_id}: "
            f"{sizes[component_id]} pixels"
        )


# STEP 3: SELECT LOWER TRUNK
plt.figure(figsize=(9, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

overlay = np.ma.masked_where(~skeleton, skeleton)
plt.imshow(
    overlay, cmap="autumn",
    vmin=0, vmax=1, alpha=0.8
)

plt.title("Click the LOWER TRUNK of the target tree")
plt.axis("off")

points = plt.ginput(1, timeout=0)
plt.close()

if not points:
    raise RuntimeError("No trunk selected")

trunk_x, trunk_y = points[0]

# Find nearest skeleton pixel to clicked location
ys, xs = np.where(skeleton)

if len(xs) == 0:
    raise RuntimeError("Skeleton is empty")

dist2 = (
    (xs - trunk_x)**2
    + (ys - trunk_y)**2
)

nearest = np.argmin(dist2)

start_x = int(xs[nearest])
start_y = int(ys[nearest])

if dist2[nearest] > 30**2:
    raise RuntimeError(
        "No skeleton within 30 pixels of selected trunk"
    )

selected_id = labels[start_y, start_x]

main_skeleton = labels == selected_id

print("\n--- SELECTED COMPONENT ---")
print("Starting pixel:", (start_x, start_y))
print("Component ID:", selected_id)
print("Component pixels:", int(main_skeleton.sum()))


# STEP 4: COUNT SKELETON NEIGHBOURS
kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0

neighbours = ndimage.convolve(
    main_skeleton.astype(int),
    kernel,
    mode="constant",
    cval=0
)

# Endpoints have exactly one neighbour
endpoints = main_skeleton & (neighbours == 1)

# Junction pixels have at least three neighbours
junctions = main_skeleton & (neighbours >= 3)

# Group adjacent junction pixels
junction_labels, junction_count = ndimage.label(
    junctions, structure=structure
)

end_y, end_x = np.where(endpoints)

junction_centres = []

for junction_id in range(1, junction_count + 1):
    jy, jx = np.where(junction_labels == junction_id)

    junction_centres.append((
        float(jx.mean()),
        float(jy.mean())
    ))

print("\n--- SKELETON STRUCTURE ---")
print("Endpoints:", len(end_x))
print("Junction regions:", len(junction_centres))


# STEP 5: TRACE DISTANCE FROM THE TRUNK
# Weighted 8-neighbour shortest paths
# Horizontal/vertical step = 1 pixel
# Diagonal step = sqrt(2) pixels
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

path_y, path_x = np.where(main_skeleton)

pixel_index = np.full(
    main_skeleton.shape, -1, dtype=int
)

pixel_index[path_y, path_x] = np.arange(len(path_x))

rows = []
cols = []
weights = []

for dy in [-1, 0, 1]:
    for dx in [-1, 0, 1]:
        if dx == 0 and dy == 0:
            continue

        ny = path_y + dy
        nx = path_x + dx

        valid = (
            (ny >= 0) & (ny < height)
            & (nx >= 0) & (nx < width)
        )

        source = np.where(valid)[0]
        targets = pixel_index[ny[valid], nx[valid]]

        connected = targets >= 0
        source = source[connected]
        targets = targets[connected]

        rows.extend(source.tolist())
        cols.extend(targets.tolist())

        step = np.hypot(dx, dy)
        weights.extend([step] * len(source))

graph = csr_matrix(
    (weights, (rows, cols)),
    shape=(len(path_x), len(path_x))
)

start_index = pixel_index[start_y, start_x]

# Save predecessor information for reconstructing paths
distances, predecessors = dijkstra(
    graph,
    directed=False,
    indices=start_index,
    return_predecessors=True
)

# Identify the furthest reachable endpoint
endpoint_indices = pixel_index[end_y, end_x]
endpoint_distances = distances[endpoint_indices]

if len(endpoint_indices) > 0:
    farthest = int(np.argmax(endpoint_distances))
    longest_length = endpoint_distances[farthest]
    target_index = int(endpoint_indices[farthest])
else:
    longest_length = np.nan
    target_index = None

# STEP 6: RECONSTRUCT LONGEST SHORTEST PATH
longest_path = []

if target_index is not None:
    current = target_index
    while current != start_index and current >= 0:
        longest_path.append(current)
        current = int(predecessors[current])

    if current == start_index:
        longest_path.append(start_index)
        longest_path.reverse()
    else:
        longest_path = []

if longest_path:
    line_x = path_x[longest_path]
    line_y = path_y[longest_path]
else:
    line_x = np.array([])
    line_y = np.array([])

print("\n--- PATH ANALYSIS ---")
print(
    f"Longest shortest path from trunk: "
    f"{longest_length:.2f} pixels"
)


# STEP 7: DISPLAY FOUR DIAGNOSTIC PANELS
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel 1: All connected skeleton components
display_labels = np.ma.masked_where(
    labels == 0, labels
)
axes[0, 0].imshow(
    display_labels,
    cmap="nipy_spectral",
    interpolation="nearest"
)
axes[0, 0].set_facecolor("black")
axes[0, 0].set_title(
    f"Connected skeleton components: {component_count}"
)

# Panel 2: Selected trunk component
axes[0, 1].imshow(main_skeleton, cmap="gray")
axes[0, 1].scatter(
    start_x, start_y,
    color="lime", s=90,
    label="Trunk start"
)
axes[0, 1].set_title("Selected connected skeleton")
axes[0, 1].legend()

# Panel 3: Endpoints and junction candidates
axes[1, 0].imshow(
    depth, cmap="gray", vmin=0, vmax=255
)

overlay = np.ma.masked_where(
    ~main_skeleton, main_skeleton
)
axes[1, 0].imshow(
    overlay, cmap="autumn",
    vmin=0, vmax=1, alpha=0.8
)

axes[1, 0].scatter(
    end_x, end_y,
    color="lime", s=25,
    label="Endpoints"
)

if junction_centres:
    jx, jy = np.array(junction_centres).T
    axes[1, 0].scatter(
        jx, jy,
        color="cyan", marker="x",
        s=60, label="Junction regions"
    )

axes[1, 0].set_title("Endpoints and junction candidates")
axes[1, 0].legend(fontsize=8)

# Panel 4: Longest shortest path
axes[1, 1].imshow(
    depth, cmap="gray", vmin=0, vmax=255
)

axes[1, 1].imshow(
    overlay, cmap="autumn",
    vmin=0, vmax=1, alpha=0.5
)

if len(line_x):
    axes[1, 1].plot(
        line_x, line_y,
        color="cyan", linewidth=2,
        label="Longest path"
    )

axes[1, 1].scatter(
    start_x, start_y,
    color="lime", s=90,
    label="Trunk start"
)

axes[1, 1].set_title("Longest path from trunk")
axes[1, 1].legend(fontsize=8)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    results_folder / f"{TREE_ID}_skeleton_trace.png",
    dpi=150
)

plt.show()


# STEP 8: SAVE NUMERICAL RESULTS
summary = pd.DataFrame([{
    "Tree_ID": TREE_ID,
    "Total_Components": component_count,
    "Selected_Component_Pixels": int(main_skeleton.sum()),
    "Endpoints": len(end_x),
    "Junction_Regions": len(junction_centres),
    "Longest_Path_Pixels": longest_length
}])

summary.to_csv(
    results_folder / f"{TREE_ID}_skeleton_trace.csv",
    index=False
)

np.save(
    results_folder / f"{TREE_ID}_main_skeleton.npy",
    main_skeleton
)

print("\n--- SUMMARY ---")
print(summary.to_string(index=False))
print("\nResults saved.")
