
from pathlib import Path
import pandas as pd
import numpy as np

# STEP 1: Load our extracted features
project = Path(__file__).resolve().parent.parent
data = pd.read_csv(project / "results" / "tree_features.csv")

features = ["Mean_Depth", "Std_Depth", "Zero_Percent"]

# STEP 2: Pair the known summer and winter trees
summer = data[data["Season"] == "Summer"]
winter = data[data["Season"] == "Winter"]

pairs = pd.merge(
    summer, winter,
    on="Tree_ID",
    suffixes=("_Summer", "_Winter"),
    validate="one_to_one"
)

# Remove pairs with invalid or missing features
columns = [
    f"{feature}_{season}"
    for feature in features
    for season in ["Summer", "Winter"]
]
pairs = pairs.replace([np.inf, -np.inf], np.nan)
pairs = pairs.dropna(subset=columns)

# STEP 3: Split by tree identity
rng = np.random.default_rng(42)
indices = rng.permutation(len(pairs))

split = int(0.8 * len(pairs))
train = pairs.iloc[indices[:split]]
test = pairs.iloc[indices[split:]]

print("Development trees:", len(train))
print("Test trees:", len(test))

# STEP 4: Create numerical feature arrays
summer_cols = [f + "_Summer" for f in features]
winter_cols = [f + "_Winter" for f in features]

summer_train = train[summer_cols].to_numpy()
winter_train = train[winter_cols].to_numpy()

summer_test = test[summer_cols].to_numpy()
winter_test = test[winter_cols].to_numpy()

# STEP 5: Standardise using development statistics
# We use separate scaling for each season.
s_mean = summer_train.mean(axis=0)
s_std = summer_train.std(axis=0)

w_mean = winter_train.mean(axis=0)
w_std = winter_train.std(axis=0)

# Avoid division by zero
s_std[s_std == 0] = 1
w_std[w_std == 0] = 1

summer_test = (summer_test - s_mean) / s_std
winter_test = (winter_test - w_mean) / w_std

# STEP 6: Match every summer tree to winter candidates
correct = 0
top3_correct = 0

results = []

for i, summer_tree in enumerate(summer_test):

    # Calculate distance to every winter test tree
    distances = np.sqrt(
        np.sum((winter_test - summer_tree) ** 2, axis=1)
    )

    # Rank candidate matches from closest to furthest
    ranking = np.argsort(distances)

    actual_id = test.iloc[i]["Tree_ID"]
    predicted_id = test.iloc[ranking[0]]["Tree_ID"]

    if actual_id == predicted_id:
        correct += 1

    top3_ids = test.iloc[ranking[:3]]["Tree_ID"].tolist()

    if actual_id in top3_ids:
        top3_correct += 1

    results.append([
        actual_id,
        predicted_id,
        round(float(distances[ranking[0]]), 3),
        actual_id == predicted_id
    ])

# STEP 7: Calculate accuracy
n = len(test)

print("\n--- MATCHING RESULTS ---")
print(f"Top-1 accuracy: {100 * correct / n:.2f}%")
print(f"Top-3 recall: {100 * top3_correct / n:.2f}%")
print(f"Random top-1 chance: {100 / n:.2f}%")

# STEP 8: Save results
output = pd.DataFrame(
    results,
    columns=["Actual_ID", "Predicted_ID",
             "Distance", "Correct"]
)

output.to_csv(
    project / "results" / "matching_results.csv",
    index=False
)

print("\nFirst 10 predictions:")
print(output.head(10).to_string(index=False))
