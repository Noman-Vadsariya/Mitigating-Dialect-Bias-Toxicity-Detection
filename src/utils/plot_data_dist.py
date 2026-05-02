import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
# df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/train.csv")   # or your full dataset CSV
df = pd.read_csv("../../data/processed/twitterAAE/train.csv")   # or your full dataset CSV
# df = pd.read_csv("../../data/processed/hatexplain/test.csv")   # or your full dataset CSV

# Make sure labels are readable
# Toxicity: 1 = toxic, 0 = non-toxic
df["toxicity"] = df["label"].map({1: "Toxic", 0: "Non-toxic"}).fillna(df["label"].astype(str))

# Dialect: AAE / SAE
# df["dialect"] = df["dialect_strict"].astype(str)
df["dialect"] = df["dialect_strict"].astype(str)

# Joint table for the heatmap
joint = pd.crosstab(df["dialect"], df["toxicity"])

# Create a 1x3 figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1) Toxicity distribution
tox_counts = df["toxicity"].value_counts().reindex(["Non-toxic", "Toxic"])
axes[0].bar(tox_counts.index, tox_counts.values)
axes[0].set_title("Toxicity Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
for i, v in enumerate(tox_counts.values):
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

# 2) Dialect distribution
dia_counts = df["dialect"].value_counts().reindex(["SAE", "AAE"])
axes[1].bar(dia_counts.index, dia_counts.values)
axes[1].set_title("Dialect Distribution")
axes[1].set_xlabel("Dialect")
axes[1].set_ylabel("Count")
for i, v in enumerate(dia_counts.values):
    axes[1].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

# 3) Joint distribution heatmap-like plot
im = axes[2].imshow(joint.values, aspect="auto")
axes[2].set_title("Dialect × Toxicity")
axes[2].set_xticks(range(len(joint.columns)))
axes[2].set_xticklabels(joint.columns)
axes[2].set_yticks(range(len(joint.index)))
axes[2].set_yticklabels(joint.index)

for i in range(joint.shape[0]):
    for j in range(joint.shape[1]):
        axes[2].text(j, i, int(joint.values[i, j]), ha="center", va="center", fontweight="bold")

plt.tight_layout()
plt.show()