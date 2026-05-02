import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

# =========================
# CONFIG
# =========================
MODEL_DIR = "../../models/bert_finetuned/checkpoint-730"   # folder saved by Trainer
TEST_CSV = "../../data/processed/hatexplain/test.csv"
# OUTPUT_CSV = "../../data/results/twitterAAE_baselines/bert_finetuned_predictions.csv"
OUTPUT_CSV = "../../data/results/hatexplain_baselines/bert_finetuned_predictions.csv"
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
test_df = pd.read_csv(TEST_CSV)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

test_enc = tokenize(test_df["tweet"])

# =========================
# DATASET
# =========================
class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

test_dataset = CustomDataset(test_enc, test_df["label"].values)

# =========================
# LOAD TRAINED MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)

# =========================
# PREDICTION HELPER
# =========================
trainer = Trainer(model=model)

print("Running inference on test set...")
predictions = trainer.predict(test_dataset)

logits = predictions.predictions

# For num_labels=2, use softmax and take class-1 probability
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
preds = np.argmax(logits, axis=1)

# =========================
# SAVE OUTPUT
# =========================
test_df["bert_prob"] = probs
test_df["bert_pred"] = preds

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
test_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved predictions to {OUTPUT_CSV}")

# =========================
# METRICS
# =========================
acc = accuracy_score(test_df["label"], preds)
f1 = f1_score(test_df["label"], preds)

print(f"Final Test Accuracy: {acc:.4f}")
print(f"Final Test F1 Score: {f1:.4f}")