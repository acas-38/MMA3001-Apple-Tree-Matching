
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Find all PNG images in the Data folder
data_folder = Path(__file__).resolve().parent.parent / "Data"
files = sorted(data_folder.rglob("*.png"))

print("Total images found:", len(files))

if not files:
    raise FileNotFoundError("No PNG images found in the Data folder")

# Display the first five images
for file in files[:5]:
    image = Image.open(file)
    data = np.array(image)

    print("\nFile:", file.name)
    print("Image type:", image.mode)
    print("Image dimensions:", image.size)
    print("Array shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum value:", data.min())
    print("Maximum value:", data.max())

    plt.figure(figsize=(7, 4))
    plt.imshow(image, cmap="gray" if image.mode in ("L", "I", "I;16") else None)
    plt.title(file.name)
    plt.axis("off")
    plt.show()

# Count each image type
summer_rgb = 0
summer_depth = 0
summer_gt = 0
winter_rgb = 0
winter_depth = 0

# Store unique tree IDs
trees = set()

for file in files:
    name = file.name

    # Tree ID is the first part of the filename
    tree_id = name.split("_")[0]
    trees.add(tree_id)

    if "_Summer_RGB" in name:
        summer_rgb += 1
    elif "_Summer_D-" in name:
        summer_depth += 1
    elif "_Summer_GT" in name:
        summer_gt += 1
    elif "_Winter_RGB" in name:
        winter_rgb += 1
    elif "_Winter_D-" in name:
        winter_depth += 1

print("\n--- DATASET SUMMARY ---")
print("Total unique tree IDs:", len(trees))
print("Summer RGB images:", summer_rgb)
print("Summer depth images:", summer_depth)
print("Summer ground-truth images:", summer_gt)
print("Winter RGB images:", winter_rgb)
print("Winter depth images:", winter_depth)

print("\nFirst 10 tree IDs:")
print(sorted(trees)[:10])

from collections import Counter

# Count the number of images for each tree
tree_counts = Counter()

for file in files:
    tree_id = file.name.split("_")[0]
    tree_counts[tree_id] += 1

# Find trees that do not have exactly 5 images
incomplete_trees = []

for tree_id, count in tree_counts.items():
    if count != 5:
        incomplete_trees.append((tree_id, count))

print("Trees with exactly 5 images:",
      sum(count == 5 for count in tree_counts.values()))

print("Trees with a different number of images:",
      len(incomplete_trees))

print("Examples:", incomplete_trees[:10])

# Compare the summer and winter depth images of R01N01

summer_file = data_folder / "R01N01_Summer_D-12-40-50.png"
winter_file = data_folder / "R01N01_Winter_D-11-23-49.png"

summer = np.array(Image.open(summer_file))
winter = np.array(Image.open(winter_file))

# Display the two images side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(summer)
plt.title("Summer Depth")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(winter)
plt.title("Winter Depth")
plt.axis("off")

plt.show()

# Print numerical information
print("Summer depth shape:", summer.shape)
print("Winter depth shape:", winter.shape)

print("Summer minimum:", summer.min())
print("Summer maximum:", summer.max())

print("Winter minimum:", winter.min())
print("Winter maximum:", winter.max())
