from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# CONFIG
# =========================
SEED = 42
MODEL_NAME = "unitary/toxic-bert"

TEXT_COL = "tweet"
LABEL_COL = "label"   # 0 = non-toxic, 1 = hate, 2 = offensive

TRAIN_CSV = Path("../../data/processed/twitterAAE/unbalanced_ternary/train.csv")
VAL_CSV = Path("../../data/processed/twitterAAE/unbalanced_ternary/val.csv")
TEST_CSV = Path("../../data/processed/twitterAAE/unbalanced_ternary/test.csv")

RESULTS_DIR = Path("../../data/results/toxicbert_ternary")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 128

ID2LABEL = {0: "non-toxic", 1: "hate", 2: "offensive"}
LABEL2ID = {"non-toxic": 0, "hate": 1, "offensive": 2}

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# LOAD DATA
# =========================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for df in [train_df, val_df, test_df]:
        df[TEXT_COL] = df[TEXT_COL].astype(str)
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =========================
# DATASET
# =========================
class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[TEXT_COL])
        label = int(row[LABEL_COL])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =========================
# METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# =========================
# CUSTOM TRAINER
# =========================
class MulticlassTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Explicit multiclass loss
        loss = F.cross_entropy(logits, labels.long())

        return (loss, outputs) if return_outputs else loss


# =========================
# MODEL
# =========================
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"
    return model


# =========================
# MAIN
# =========================
def main():
    train_df, val_df, test_df = load_data()

    print("Train label distribution:")
    print(train_df[LABEL_COL].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(val_df[LABEL_COL].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df[LABEL_COL].value_counts().sort_index())

    train_ds = ToxicityDataset(train_df, tokenizer, max_len=MAX_LEN)
    val_ds = ToxicityDataset(val_df, tokenizer, max_len=MAX_LEN)
    test_ds = ToxicityDataset(test_df, tokenizer, max_len=MAX_LEN)

    model = load_model()

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
    )

    trainer = MulticlassTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nTraining ToxicBERT on ternary classification...")
    trainer.train()

    print("\nValidation results:")
    val_results = trainer.evaluate(val_ds)
    print(val_results)

    print("\nTest results:")
    test_output = trainer.predict(test_ds)
    test_logits = test_output.predictions
    test_labels = test_output.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    print("Test Accuracy:", accuracy_score(test_labels, test_preds))
    print("Test Macro F1:", f1_score(test_labels, test_preds, average="macro", zero_division=0))
    print("Test Weighted F1:", f1_score(test_labels, test_preds, average="weighted", zero_division=0))
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=["non-toxic", "hate", "offensive"],
            zero_division=0,
        )
    )

    # Save test predictions
    probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

    pred_df = test_df.copy()
    pred_df["pred_class"] = test_preds
    pred_df["pred_label"] = pred_df["pred_class"].map(ID2LABEL)
    pred_df["prob_non_toxic"] = probs[:, 0]
    pred_df["prob_hate"] = probs[:, 1]
    pred_df["prob_offensive"] = probs[:, 2]

    out_file = RESULTS_DIR / "test_predictions.csv"
    pred_df.to_csv(out_file, index=False)
    print(f"\nSaved predictions -> {out_file}")

    # Save final model
    final_dir = RESULTS_DIR / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model -> {final_dir}")


if __name__ == "__main__":
    main()