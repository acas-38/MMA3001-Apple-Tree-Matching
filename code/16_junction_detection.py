
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Settings
TREE_ID = "R20N03"     # Later try R09N28
THRESHOLD = 80
ROW_STEP = 5
SEARCH_RADIUS = 35
MIN_SEPARATION = 25
PERSISTENCE = 4

project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

files = list(data_folder.rglob(f"{TREE_ID}_Winter_D-*.png"))

if not files:
    raise FileNotFoundError("Winter depth image not found")

depth = np.array(Image.open(files[0]).convert("L"))
height, width = depth.shape
binary = depth >= THRESHOLD


# STEP 2: Manually select a point on the trunk
plt.figure(figsize=(9, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
plt.title("Click on the trunk below the branch junction")
plt.axis("off")

clicked = plt.ginput(1, timeout=0)
plt.close()

if not clicked:
    raise RuntimeError("No starting point selected")

start_x, start_y = clicked[0]


# STEP 3: Find connected foreground groups in one row
def get_groups(row, centre_x, radius):
    xs = np.where(binary[row])[0]
    xs = xs[np.abs(xs - centre_x) <= radius]

    if len(xs) == 0:
        return []

    groups = np.split(
        xs, np.where(np.diff(xs) > 1)[0] + 1
    )

    return [g for g in groups if len(g) >= 3]


# STEP 4: Track the trunk upwards
track = []
current_x = start_x

for y in range(int(start_y), 19, -ROW_STEP):

    groups = get_groups(y, current_x, SEARCH_RADIUS)

    if not groups:
        break

    best = min(
        groups,
        key=lambda g: abs(np.mean(g) - current_x)
    )

    current_x = float(np.mean(best))
    track.append([y, current_x])

track = np.array(track).reshape(-1, 2)

print("Tracked points:", len(track))


# STEP 5: Find persistent branch separation
# Examine a wider region around the tracked path.
# Two foreground groups must appear over several rows.

separation_rows = []

for y, x in track:

    y = int(y)

    groups = get_groups(y, x, 120)

    if len(groups) < 2:
        separation_rows.append(False)
        continue

    centres = sorted([np.mean(g) for g in groups])

    # Find largest horizontal gap between groups
    gap = max(np.diff(centres))

    separation_rows.append(gap >= MIN_SEPARATION)


# Search from the bottom upwards for persistent separation
junction = None

for i in range(len(separation_rows) - PERSISTENCE + 1):

    window = separation_rows[i:i + PERSISTENCE]

    if all(window):
        y, x = track[i]
        junction = (x, y)
        break


# STEP 6: Display result
plt.figure(figsize=(10, 6))
plt.imshow(depth, cmap="gray", vmin=0, vmax=255)

if len(track):
    plt.plot(
        track[:, 1],
        track[:, 0],
        color="cyan",
        linewidth=2,
        label="Tracked path"
    )

plt.scatter(
    start_x, start_y,
    color="lime",
    s=80,
    label="Starting trunk point"
)

if junction is not None:
    plt.scatter(
        junction[0],
        junction[1],
        color="red",
        s=100,
        label="Candidate junction"
    )
    print(f"Candidate junction: x={junction[0]:.1f}, y={junction[1]:.1f}")
else:
    print("No persistent junction detected")

plt.title(f"{TREE_ID}: Junction detection")
plt.xlim(0, width)
plt.ylim(height, 0)
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()
