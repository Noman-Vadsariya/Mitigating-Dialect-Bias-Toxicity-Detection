import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "unitary/toxic-bert"
BATCH_SIZE = 32

# INPUT_FILE = "../data/processed/test.csv"
INPUT_FILE = "data/processed/twitterAAE/unbalanced/test.csv"
# OUTPUT_FILE = "../data/results/toxicbert_predictions.csv"
OUTPUT_FILE = "results/toxicbert_predictions.csv"

# LOAD MODEL
print("Loading ToxicBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

print(f"Using device: {DEVICE}")

# PREDICTION FUNCTION
def predict_batch(texts):
    all_scores = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)

            # Convert logits → probability
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

        # Extract toxicity score
        all_scores.extend(probs[:, 0])

    return all_scores

# LOAD DATA
df = pd.read_csv(INPUT_FILE)

texts = df["tweet"].astype(str).tolist()

# RUN INFERENCE
print("Running inference...")

df["toxicbert_score"] = predict_batch(texts)

# Convert to binary prediction
df["toxicbert_pred"] = (df["toxicbert_score"] > 0.5).astype(int)

# SAVE RESULTS
os.makedirs("results", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved predictions → {OUTPUT_FILE}")

# QUICK METRICS (optional)
if "label" in df.columns:
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(df["label"], df["toxicbert_pred"])
    f1 = f1_score(df["label"], df["toxicbert_pred"])

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")