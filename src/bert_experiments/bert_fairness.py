from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
MODEL_NAME = "bert-base-uncased"

TEXT_COL = "tweet"
LABEL_COL = "label"
DIALECT_COL = "dialect_strict"   # expected values: AAE / SAE

TRAIN_CSV = Path("../../data/processed/twitterAAE/unbalanced/train.csv")
VAL_CSV = Path("../../data/processed/twitterAAE/unbalanced/val.csv")
TEST_CSV = Path("../../data/processed/twitterAAE/unbalanced/test.csv")

RESULTS_DIR = Path("../data/results/bert_fairness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 128
GAMMA_VALUES = [0.06, 0.78, 1.5]

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TRAIN_ARGS = dict(
    output_dir=str(RESULTS_DIR / "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    remove_unused_columns=False,  # needed for dialect
    seed=SEED,
    fp16=torch.cuda.is_available(),
)

# =========================
# HELPERS
# =========================
def normalize_dialect(series: pd.Series) -> np.ndarray:
    mapped = series.astype(str).str.strip().str.upper().map({"AAE": 1, "SAE": 0})
    if mapped.isna().any():
        bad = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Unexpected values in {DIALECT_COL}: {bad}")
    return mapped.values.astype(int)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for df in [train_df, val_df, test_df]:
        df[TEXT_COL] = df[TEXT_COL].astype(str)
        df[LABEL_COL] = df[LABEL_COL].astype(int)
        df[DIALECT_COL] = df[DIALECT_COL].astype(str)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[TEXT_COL])
        label = int(row[LABEL_COL])
        dialect = 1 if str(row[DIALECT_COL]).strip().upper() == "AAE" else 0

        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["dialect"] = torch.tensor(dialect, dtype=torch.long)
        return item


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, dialect: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = dialect == g
        yt = y_true[idx]
        yp = y_pred[idx]

        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))
        tp = np.sum((yt == 1) & (yp == 1))

        out[f"fpr_{name}"] = fp / (fp + tn + 1e-12)
        out[f"fnr_{name}"] = fn / (fn + tp + 1e-12)

    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def eval_predictions(df: pd.DataFrame, logits: np.ndarray) -> Dict[str, float]:
    y_true = df[LABEL_COL].values.astype(int)
    dialect = normalize_dialect(df[DIALECT_COL])
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    metrics = {}
    metrics.update(compute_overall_metrics(y_true, y_pred))
    metrics.update(compute_group_metrics(y_true, y_pred, dialect))
    return metrics


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


# =========================
# TRAINER
# =========================
class FairnessTrainer(Trainer):
    """
    End-to-end BERT with fairness-aware objective:

    Loss = (1 - gamma) * CE + gamma * (mean_p_AAE_non_toxic - mean_p_SAE_non_toxic)^2
    """
    def __init__(self, *args, gamma: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        dialect = inputs.pop("dialect")

        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = F.cross_entropy(logits, labels)

        # Toxic probability
        probs = torch.softmax(logits, dim=-1)[:, 1]

        # Only non-toxic samples
        aae_non = (dialect == 1) & (labels == 0)
        sae_non = (dialect == 0) & (labels == 0)

        if aae_non.any() and sae_non.any():
            mean_aae = probs[aae_non].mean()
            mean_sae = probs[sae_non].mean()
            fair_loss = (mean_aae - mean_sae).pow(2)
        else:
            fair_loss = torch.tensor(0.0, device=logits.device)

        loss = (1.0 - self.gamma) * ce_loss + self.gamma * fair_loss
        return (loss, outputs) if return_outputs else loss


def run_one_gamma(gamma: float, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_ds = ToxicityDataset(train_df, max_len=MAX_LEN)
    val_ds = ToxicityDataset(val_df, max_len=MAX_LEN)
    test_ds = ToxicityDataset(test_df, max_len=MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(**TRAIN_ARGS)
    args.output_dir = str(RESULTS_DIR / f"gamma_{gamma:.2f}".replace(".", "_"))

    trainer = FairnessTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        gamma=float(gamma),
    )

    trainer.train()

    pred_out = trainer.predict(test_ds)
    test_logits = pred_out.predictions
    metrics = eval_predictions(test_df, test_logits)

    probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    pred_df = test_df.copy()
    pred_df["prob_toxic"] = probs
    pred_df["pred"] = preds
    pred_df.to_csv(RESULTS_DIR / f"test_predictions_gamma_{gamma:.2f}.csv".replace(".", "_"), index=False)

    return metrics


def main():
    train_df, val_df, test_df = load_data()

    print("Train label distribution:\n", train_df[LABEL_COL].value_counts())
    print("\nTrain dialect distribution:\n", train_df[DIALECT_COL].value_counts())
    print("\nJoint distribution (train):")
    print(pd.crosstab(train_df[DIALECT_COL], train_df[LABEL_COL]))

    rows: List[Dict[str, float]] = []

    for gamma in GAMMA_VALUES:
        print(f"\n=== Fairness gamma={gamma:.2f} ===")
        metrics = run_one_gamma(float(gamma), train_df, val_df, test_df)
        metrics["gamma"] = float(gamma)
        rows.append(metrics)
        print(metrics)

    metrics_df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "fairness_sweep_metrics.csv", index=False)
    print("\nSaved:", RESULTS_DIR / "fairness_sweep_metrics.csv")

    # =========================
    # PLOTS
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["gamma"], metrics_df["fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(metrics_df["gamma"], metrics_df["fpr_SAE"], marker="o", label="FPR SAE")
    plt.plot(metrics_df["gamma"], metrics_df["fpr_gap"], marker="o", label="FPR Gap")
    plt.xlabel("Gamma")
    plt.ylabel("FPR")
    plt.title("FPR vs Fairness Strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fpr_vs_gamma.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["gamma"], metrics_df["fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(metrics_df["gamma"], metrics_df["fnr_SAE"], marker="o", label="FNR SAE")
    plt.plot(metrics_df["gamma"], metrics_df["fnr_gap"], marker="o", label="FNR Gap")
    plt.xlabel("Gamma")
    plt.ylabel("FNR")
    plt.title("FNR vs Fairness Strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fnr_vs_gamma.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["gamma"], metrics_df["f1"], marker="o", label="F1")
    plt.xlabel("Gamma")
    plt.ylabel("F1")
    plt.title("F1 vs Fairness Strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_vs_gamma.png", dpi=300)
    plt.close()

    print("Saved plots to:", RESULTS_DIR)


if __name__ == "__main__":
    main()