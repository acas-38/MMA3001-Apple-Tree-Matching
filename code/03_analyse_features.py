
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Load the CSV file
project_folder = Path(__file__).resolve().parent.parent
csv_file = project_folder / "results" / "tree_features.csv"

data = pd.read_csv(csv_file)

print("Total observations:", len(data))
print("\nDataset summary:")
print(data.groupby("Season")[[
    "Mean_Depth", "Std_Depth", "Zero_Percent"
]].agg(["mean", "std"]).round(2))

# STEP 2: Separate summer and winter data
summer = data[data["Season"] == "Summer"]
winter = data[data["Season"] == "Winter"]

# Match observations using tree IDs
paired = pd.merge(
    summer,
    winter,
    on="Tree_ID",
    suffixes=("_Summer", "_Winter"),
    validate="one_to_one"
)

print("\nMatched tree pairs:", len(paired))

# STEP 3: Compare each numerical feature
features = ["Mean_Depth", "Std_Depth", "Zero_Percent"]

for feature in features:

    summer_values = paired[feature + "_Summer"]
    winter_values = paired[feature + "_Winter"]

    # Calculate correlation
    correlation = summer_values.corr(winter_values)

    print(f"\n{feature}")
    print(f"Summer-winter correlation: {correlation:.3f}")

    # Plot the summer and winter values
    plt.figure(figsize=(7, 5))

    plt.scatter(
        summer_values,
        winter_values,
        alpha=0.5
    )

    plt.xlabel("Summer " + feature)
    plt.ylabel("Winter " + feature)
    plt.title(feature + " - Summer vs Winter")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# STEP 4: Compare seasonal distributions
plt.figure(figsize=(8, 5))

plt.hist(
    summer["Zero_Percent"],
    bins=25,
    alpha=0.6,
    label="Summer"
)

plt.hist(
    winter["Zero_Percent"],
    bins=25,
    alpha=0.6,
    label="Winter"
)

plt.xlabel("Zero-valued pixels (%)")
plt.ylabel("Number of images")
plt.title("Depth Data: Summer vs Winter")
plt.legend()
plt.tight_layout()
plt.show()
