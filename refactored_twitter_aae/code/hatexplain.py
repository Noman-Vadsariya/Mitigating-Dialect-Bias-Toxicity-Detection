import json
import pandas as pd
from collections import Counter

# -----------------------------
# Majority vote function
# -----------------------------
def majority_label(annotators):
    labels = [a["label"] for a in annotators]
    return Counter(labels).most_common(1)[0][0]


# -----------------------------
# Main conversion function
# -----------------------------
def hatexplain_json_to_csv(json_path, output_csv):
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for post_id, example in data.items():
        try:
            # 1. Convert tokens → text
            text = " ".join(example["post_tokens"])

            # 2. Majority label
            maj_label = majority_label(example["annotators"])

            # HateXplain labels:
            # 0 = hatespeech
            # 1 = normal
            # 2 = offensive

            # Convert to binary toxicity
            label = 0 if maj_label == "normal" else 1

            rows.append({
                "tweet": text,
                "label": label,
            })

        except Exception as e:
            print(f"Skipping example {post_id}: {e}")
            continue

    df = pd.DataFrame(rows)

    print("Dataset size:", len(df))
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    hatexplain_json_to_csv(
        json_path="../../data/raw/hatexplain.json",   # your downloaded file
        output_csv="../../data/raw/hatexplain_processed.csv"
    )