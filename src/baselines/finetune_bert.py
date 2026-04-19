import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os

# =========================
# CONFIG
# =========================
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
# train_df = pd.read_csv("../../data/processed/twitterAAE/train.csv")
# val_df = pd.read_csv("../../data/processed/twitterAAE/val.csv")
# test_df = pd.read_csv("../../data/processed/twitterAAE/test.csv")

train_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/train.csv")
val_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/val.csv")
test_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/test.csv")

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128
    )

train_enc = tokenize(train_df["tweet"])
val_enc = tokenize(val_df["tweet"])
test_enc = tokenize(test_df["tweet"])

# =========================
# DATASET CLASS
# =========================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_enc, train_df["label"].values)
val_dataset = CustomDataset(val_enc, val_df["label"].values)
test_dataset = CustomDataset(test_enc, test_df["label"].values)

# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    # problem_type="single_label_classification"
)

model.to(DEVICE)

# =========================
# METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# =========================
# TRAINING CONFIG (v5 compatible)
# =========================
training_args = TrainingArguments(
    output_dir="../../models/bert_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    eval_strategy="epoch",   # correct for v5
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=1
)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =========================
# TRAIN
# =========================
print("Training BERT...")
trainer.train()

# =========================
# PREDICT ON TEST
# =========================
print("Running inference...")

predictions = trainer.predict(test_dataset)

logits = predictions.predictions
preds = np.argmax(logits, axis=1)
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

# =========================
# SAVE RESULTS
# =========================
test_df["bert_prob"] = probs
test_df["bert_pred"] = preds

test_df.to_csv("../../data/results/twitterAAE_baselines/bert_finetuned_predictions.csv", index=False)


# =========================
# FINAL METRICS
# =========================
acc = accuracy_score(test_df["label"], preds)
f1 = f1_score(test_df["label"], preds)

print(f"\nFinal Test Accuracy: {acc:.4f}")
print(f"Final Test F1 Score: {f1:.4f}")