import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import html

# Import your AAE wrapper
from infer_label import get_aae_score

SEED = 42
np.random.seed(SEED)

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../../data/raw/twitterAAE.csv")

# Adjust column name if needed
TEXT_COL = "tweet" if "tweet" in df.columns else "text"

# =========================
# 2. CLEAN TEXT (light)
# =========================
def clean_text(text):
    text = str(text).lower()
    text = html.unescape(text)  # Decode HTML entities
    text = text.replace("&#128545;", "")  # Remove specific problematic emojis
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# Ensure tqdm works with pandas
tqdm.pandas()

# Filter out rows with missing or empty text
df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")]


# 0 = neither → non-toxic
# 1 = hate → toxic
# 2 = offensive → toxic
def label_func(x):
    if x == 0:
        return 1  # hate → toxic
    elif x == 1:
        return 2  # offensive → toxic
    else:
        return 0  # neither → non-toxic
    
df["label"] = df["class"].apply(label_func)

# =========================
# 4. COMPUTE AAE SCORES
# =========================
print("Computing AAE scores...")

# Safe wrapper for get_aae_score
def safe_get_aae_score(text):
    try:
        score = get_aae_score(text)
        if score is None:
            return 0.0  # Default value for None
        return score
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return 0.0  # Default value for errors

df["p_AAE"] = df[TEXT_COL].progress_apply(safe_get_aae_score)

# =========================
# 5. DIALECT DEFINITIONS
# =========================

# Strict (paper standard)
df["dialect_strict"] = df["p_AAE"].apply(
    lambda x: "AAE" if x >= 0.8 else "SAE"
)

# Relaxed (larger coverage)
df["dialect_relaxed"] = df["p_AAE"].apply(
    lambda x: "AAE" if x >= 0.6 else "SAE"
)

# Binary versions
df["dialect_strict_bin"] = df["dialect_strict"].map({"AAE": 1, "SAE": 0})
df["dialect_relaxed_bin"] = df["dialect_relaxed"].map({"AAE": 1, "SAE": 0})

print("Total samples:", len(df))
print("Toxic samples:", len(df[df["label"] == 1]))
print("Non-toxic samples:", len(df[df["label"] == 0]))

# =========================
# 6. BALANCE DATASET (MAX SIZE)
# =========================
df_toxic = df[df["label"] == 1]
df_clean = df[df["label"] == 0]

n = min(len(df_toxic), len(df_clean))

# df_balanced = pd.concat([
#     df_toxic.sample(n=n, random_state=SEED),
#     df_clean.sample(n=n, random_state=SEED)
# ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# print(f"Balanced dataset size: {len(df_balanced)}")

# =========================
# 7. TRAIN / VAL / TEST SPLIT
# =========================

train, temp = train_test_split(
    df,
    test_size=0.3,
    stratify=df[["label", "dialect_strict_bin"]],
    random_state=SEED
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp[["label", "dialect_strict_bin"]],
    random_state=SEED
)

# =========================
# 8. SAVE FILES
# =========================
train.to_csv("../../data/processed/twitterAAE/unbalanced_ternary/train.csv", index=False)
val.to_csv("../../data/processed/twitterAAE/unbalanced_ternary/val.csv", index=False)
test.to_csv("../../data/processed/twitterAAE/unbalanced_ternary/test.csv", index=False)

print("Saved processed datasets!")

# =========================
# 9. SANITY CHECKS
# =========================
def check_distribution(df, name):
    print(f"\n{name}")
    print("Label dist:\n", df["label"].value_counts(normalize=True))
    print("Strict dialect dist:\n", df["dialect_strict"].value_counts(normalize=True))
    print("Relaxed dialect dist:\n", df["dialect_relaxed"].value_counts(normalize=True))

check_distribution(train, "TRAIN")
check_distribution(val, "VAL")
check_distribution(test, "TEST")