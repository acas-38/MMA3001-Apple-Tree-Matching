
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Find the dataset
project = Path(__file__).resolve().parent.parent
data_folder = project / "Data"

# Choose grid size
GRID = 3

# STEP 2: Extract features from one image
def extract_features(file, grid):
    image = np.array(Image.open(file).convert("L"))

    # Split into equally spaced regions
    rows = np.array_split(image, grid, axis=0)
    features = []

    for row in rows:
        columns = np.array_split(row, grid, axis=1)

        for region in columns:
            nonzero = region[region > 0]

            if len(nonzero) > 0:
                mean = np.mean(nonzero)
                std = np.std(nonzero)
            else:
                mean = 0
                std = 0

            zero = np.mean(region == 0) * 100

            features.extend([mean, std, zero])

    return features


# STEP 3: Find matching summer and winter images
summer_files = {}
winter_files = {}

for file in data_folder.rglob("*.png"):
    name = file.name
    tree_id = name.split("_")[0]

    if "_Summer_D-" in name:
        summer_files[tree_id] = file

    elif "_Winter_D-" in name:
        winter_files[tree_id] = file

tree_ids = sorted(set(summer_files) & set(winter_files))

print("Matched tree pairs:", len(tree_ids))

# STEP 4: Extract all image features
summer_features = []
winter_features = []

for tree in tree_ids:
    summer_features.append(
        extract_features(summer_files[tree], GRID)
    )

    winter_features.append(
        extract_features(winter_files[tree], GRID)
    )

summer_features = np.array(summer_features)
winter_features = np.array(winter_features)

print("Features per image:", summer_features.shape[1])


# STEP 5: Divide tree identities into development and validation
rng = np.random.default_rng(42)
indices = rng.permutation(len(tree_ids))

split = int(0.8 * len(indices))
development = indices[:split]
validation = indices[split:]

# Calculate separate seasonal scaling from development data
s_mean = summer_features[development].mean(axis=0)
s_std = summer_features[development].std(axis=0)

w_mean = winter_features[development].mean(axis=0)
w_std = winter_features[development].std(axis=0)

s_std[s_std == 0] = 1
w_std[w_std == 0] = 1

summer_scaled = (summer_features[validation] - s_mean) / s_std
winter_scaled = (winter_features[validation] - w_mean) / w_std


# STEP 6: Match each summer tree with a winter tree
results = []
top3_correct = 0

for i, summer_tree in enumerate(summer_scaled):

    distances = np.sqrt(
        np.sum((winter_scaled - summer_tree) ** 2, axis=1)
    )

    ranking = np.argsort(distances)

    actual_id = tree_ids[validation[i]]
    predicted_id = tree_ids[validation[ranking[0]]]

    top3_ids = [
        tree_ids[validation[j]] for j in ranking[:3]
    ]

    correct = actual_id == predicted_id

    if actual_id in top3_ids:
        top3_correct += 1

    results.append([actual_id, predicted_id, correct])


# STEP 7: Calculate results
results_df = pd.DataFrame(
    results,
    columns=["Actual_ID", "Predicted_ID", "Correct"]
)

accuracy = results_df["Correct"].mean() * 100
top3 = top3_correct / len(results_df) * 100

print("\n--- SPATIAL MATCHING RESULTS ---")
print("Grid size:", GRID, "x", GRID)
print("Validation trees:", len(validation))
print(f"Top-1 accuracy: {accuracy:.2f}%")
print(f"Top-3 recall: {top3:.2f}%")
print(f"Random top-1 chance: {100 / len(validation):.2f}%")

# STEP 8: Save results
output_folder = project / "results"
output_folder.mkdir(exist_ok=True)

results_df.to_csv(
    output_folder / f"spatial_matching_{GRID}x{GRID}.csv",
    index=False
)

# Plot correct and incorrect matches
correct_count = int(results_df["Correct"].sum())
incorrect_count = len(results_df) - correct_count

plt.bar(
    ["Correct", "Incorrect"],
    [correct_count, incorrect_count]
)

plt.ylabel("Number of trees")
plt.title(f"Spatial Matching Results ({GRID}x{GRID})")
plt.show()
