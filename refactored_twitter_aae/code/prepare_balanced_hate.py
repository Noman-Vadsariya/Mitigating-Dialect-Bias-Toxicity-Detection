import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import html

# Import your AAE wrapper
from infer_label import get_aae_score

SEED = 42
np.random.seed(SEED)
tqdm.pandas()

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../../data/raw/hatexplain_processed.csv")

# Adjust column name if needed
TEXT_COL = "tweet" if "tweet" in df.columns else "text"

# =========================
# 2. CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = html.unescape(text)
    text = text.replace("&#128545;", "")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# Remove empty rows
df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")].copy()

# =========================
# 3. CREATE BINARY LABEL
# =========================
# If label already exists, keep it.
# Otherwise create it from the original class column.
if "label" not in df.columns:
    if "class" not in df.columns:
        raise ValueError("Expected either a 'label' column or a 'class' column.")
    # 0 = hate, 1 = offensive -> toxic (1)
    # 2 = neither -> non-toxic (0)
    df["label"] = df["class"].apply(lambda x: 1 if x in [0, 1] else 0)

# =========================
# 4. COMPUTE AAE SCORES
# =========================
print("Computing AAE scores...")

def safe_get_aae_score(text):
    try:
        score = get_aae_score(text)
        if score is None:
            return 0.0
        return float(score)
    except Exception as e:
        print(f"Error processing text: {text}\nError: {e}")
        return 0.0

df["p_AAE"] = df[TEXT_COL].progress_apply(safe_get_aae_score)

# =========================
# 5. DIALECT LABELS
# =========================
# Strict threshold used for main experiments
df["dialect_strict"] = df["p_AAE"].apply(lambda x: "AAE" if x >= 0.8 else "SAE")

# Relaxed threshold for analysis
df["dialect_relaxed"] = df["p_AAE"].apply(lambda x: "AAE" if x >= 0.6 else "SAE")

print("\n=== ORIGINAL DATASET ===")
print("Total samples:", len(df))
print("Label distribution:\n", df["label"].value_counts())
print("Strict dialect distribution:\n", df["dialect_strict"].value_counts())
print("\nLabel x Dialect (strict):\n", pd.crosstab(df["label"], df["dialect_strict"]))

# =========================
# 6. UNDERSAMPLING-ONLY BALANCING TO ~60/40
# =========================
def undersample_to_60_40(
    data,
    label_col="label",
    dialect_col="dialect_strict",
    aae_ratio=0.6,
    seed=42
):
    """
    Undersampling-only strategy:
    - No oversampling
    - For each label, keep as many AAE samples as possible
    - Reduce SAE so that the final group ratio is close to 60/40
    - Also keep labels balanced by using the same target size for both labels
    """
    rng = np.random.RandomState(seed)
    balanced_parts = []

    # Split by label
    label_groups = {}
    for label_value in sorted(data[label_col].unique()):
        subset = data[data[label_col] == label_value].copy()
        aae = subset[subset[dialect_col] == "AAE"].copy()
        sae = subset[subset[dialect_col] == "SAE"].copy()
        label_groups[label_value] = (aae, sae)

    # Find the largest per-label sample size that can satisfy the 60/40 ratio
    # without oversampling, for BOTH labels.
    feasible_sizes = []
    for label_value, (aae, sae) in label_groups.items():
        max_total_from_aae = int(np.floor(len(aae) / aae_ratio)) if aae_ratio > 0 else 0
        max_total_from_sae = int(np.floor(len(sae) / (1 - aae_ratio))) if aae_ratio < 1 else 0
        feasible_total = min(max_total_from_aae, max_total_from_sae)
        feasible_sizes.append(feasible_total)

    target_per_label = min(feasible_sizes)

    if target_per_label <= 0:
        raise ValueError("Not enough data to create a 60/40 split without oversampling.")

    print(f"\nTarget samples per label after undersampling: {target_per_label}")

    for label_value, (aae, sae) in label_groups.items():
        target_aae = int(round(target_per_label * aae_ratio))
        target_sae = target_per_label - target_aae

        # Adjust in case rounding creates a mismatch
        if target_aae > len(aae):
            target_aae = len(aae)
            target_sae = target_per_label - target_aae

        if target_sae > len(sae):
            target_sae = len(sae)
            target_aae = target_per_label - target_sae

        # Final safety check: keep only what is available
        target_aae = min(target_aae, len(aae))
        target_sae = min(target_sae, len(sae))

        # If after safety checks total is smaller, trim the larger side
        current_total = target_aae + target_sae
        if current_total > target_per_label:
            extra = current_total - target_per_label
            if target_sae >= extra:
                target_sae -= extra
            else:
                target_aae -= (extra - target_sae)
                target_sae = 0

        aae_sample = aae.sample(n=target_aae, random_state=seed) if target_aae > 0 else aae.iloc[0:0]
        sae_sample = sae.sample(n=target_sae, random_state=seed) if target_sae > 0 else sae.iloc[0:0]

        combined = pd.concat([aae_sample, sae_sample], axis=0)
        balanced_parts.append(combined)

    df_balanced = (
        pd.concat(balanced_parts, axis=0)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return df_balanced

df_balanced = undersample_to_60_40(
    df,
    label_col="label",
    dialect_col="dialect_strict",
    aae_ratio=0.6,
    seed=SEED
)

# =========================
# 7. SANITY CHECKS
# =========================
print("\n=== BALANCED DATASET ===")
print("Total samples:", len(df_balanced))
print("Label distribution:\n", df_balanced["label"].value_counts())
print("\nLabel proportions:\n", df_balanced["label"].value_counts(normalize=True))

print("\nStrict dialect distribution:\n", df_balanced["dialect_strict"].value_counts())
print("\nStrict dialect proportions:\n", df_balanced["dialect_strict"].value_counts(normalize=True))

print("\nLabel x Dialect (strict):\n", pd.crosstab(df_balanced["label"], df_balanced["dialect_strict"]))

# =========================
# 8. SAVE FILE
# =========================
out_path = "../../data/processed/hatexplain/test.csv"
df_balanced.to_csv(out_path, index=False)
print(f"\nSaved processed dataset to: {out_path}")

# =========================
# 9. OPTIONAL DISTRIBUTION CHECK
# =========================
def check_distribution(df_check, name):
    print(f"\n{name}")
    print("Label dist:\n", df_check["label"].value_counts(normalize=True))
    print("Strict dialect dist:\n", df_check["dialect_strict"].value_counts(normalize=True))
    print("Relaxed dialect dist:\n", df_check["dialect_relaxed"].value_counts(normalize=True))

check_distribution(df_balanced, "FINAL BALANCED SET")