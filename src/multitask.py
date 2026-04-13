"""End-to-end multitask BERT script for toxicity + dialect prediction.

This script assumes you have three CSV files:
    - train.csv
    - val.csv
    - test.csv

And each CSV contains the following columns:
    - tweet           : raw text
    - label           : toxicity label (0/1)
    - dialect_strict  : dialect label (0/1)

What the script does:
    1. Loads train/val/test CSVs.
    2. Builds a shared BERT encoder with two heads:
         - toxicity head
         - dialect head
    3. Trains on the train split.
    4. Selects the best model using validation F1 for toxicity.
    5. Evaluates the best checkpoint on the test split.
    6. Prints performance and fairness metrics.

You can swap `bert-base-uncased` for a ToxicBERT checkpoint if you have one.
"""

from __future__ import annotations

import os
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    """Central place for hyperparameters and file paths."""

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    lambda_dialect: float = 0.5
    hidden_dropout: float = 0.1
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_multitask_bert.pt"


# -----------------------------
# Data loading
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV and normalize label formats.

    Expected columns:
        - tweet          : raw text
        - label          : toxicity label, already 0/1
        - dialect_strict : dialect label, stored as AAE/SAE

    We map:
        AAE -> 1
        SAE -> 0
    """
    df = pd.read_csv(path)

    required = {"tweet", "label", "dialect_strict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    # Keep only the columns we care about and rename them for convenience.
    df = df[["tweet", "label", "dialect_strict"]].copy()
    df = df.rename(
        columns={
            "tweet": "text",
            "label": "toxicity_label",
            "dialect_strict": "dialect_label",
        }
    )

    # Toxicity is already 0/1, but we cast to numeric and then int to be safe.
    df["toxicity_label"] = pd.to_numeric(df["toxicity_label"], errors="coerce")
    df["toxicity_label"] = df["toxicity_label"].astype("Int64")

    # Dialect is stored as strings like "AAE" and "SAE".
    # Convert to uppercase first so the mapping is robust to capitalization.
    df["dialect_label"] = df["dialect_label"].astype(str).str.strip().str.upper()
    df["dialect_label"] = df["dialect_label"].map({"AAE": 1, "SAE": 0})

    # Drop rows that became missing after conversion.
    df = df.dropna(subset=["text", "toxicity_label", "dialect_label"]).reset_index(drop=True)

    # Final integer conversion.
    df["toxicity_label"] = df["toxicity_label"].astype(int)
    df["dialect_label"] = df["dialect_label"].astype(int)
    df["text"] = df["text"].astype(str)

    return df


# -----------------------------
# Dataset class
# -----------------------------

class MultitaskTextDataset(Dataset):
    """Tokenizes text and returns both labels.

    Each item contains:
        input_ids, attention_mask, toxicity_label, dialect_label
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = row["text"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "toxicity_label": torch.tensor(row["toxicity_label"], dtype=torch.float),
            "dialect_label": torch.tensor(row["dialect_label"], dtype=torch.float),
        }


# -----------------------------
# Model
# -----------------------------

class MultitaskBERT(nn.Module):
    """Shared encoder with two binary classifiers.

    Architecture:
        text -> encoder -> [CLS] embedding -> dropout -> heads
                           /                     \
                    toxicity head           dialect head
    """

    def __init__(self, model_name: str = "bert-base-uncased", hidden_dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)
        self.toxicity_head = nn.Linear(hidden_size, 1)
        self.dialect_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token embedding as the text representation.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)

        toxicity_logits = self.toxicity_head(x).squeeze(-1)
        dialect_logits = self.dialect_head(x).squeeze(-1)

        return {
            "toxicity_logits": toxicity_logits,
            "dialect_logits": dialect_logits,
        }


# -----------------------------
# Loss
# -----------------------------

class MultitaskLoss(nn.Module):
    """Joint loss for the main task and the auxiliary task.

    total_loss = toxicity_loss + lambda_dialect * dialect_loss

    Using BCEWithLogitsLoss because the model outputs raw logits.
    """

    def __init__(self, lambda_dialect: float = 0.5):
        super().__init__()
        self.lambda_dialect = lambda_dialect
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        toxicity_logits: torch.Tensor,
        dialect_logits: torch.Tensor,
        toxicity_labels: torch.Tensor,
        dialect_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        tox_loss = self.bce(toxicity_logits, toxicity_labels)
        dia_loss = self.bce(dialect_logits, dialect_labels)
        total_loss = tox_loss + self.lambda_dialect * dia_loss

        return total_loss, {
            "tox_loss": float(tox_loss.item()),
            "dialect_loss": float(dia_loss.item()),
            "total_loss": float(total_loss.item()),
        }


# -----------------------------
# Metrics
# -----------------------------

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def group_fairness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute group-wise FPR/FNR using dialect as the group variable.

    Here we treat:
        - 1 = AAE (dialect_strict == 1)
        - 0 = SAE (dialect_strict == 0)
    """
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}

    for g_name, g_value in [("AAE", 1), ("SAE", 0)]:
        idx = group == g_value
        if idx.sum() == 0:
            out[f"fpr_{g_name}"] = np.nan
            out[f"fnr_{g_name}"] = np.nan
            continue

        yt = y_true[idx]
        yp = y_pred[idx]

        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        out[f"fpr_{g_name}"] = fpr
        out[f"fnr_{g_name}"] = fnr

    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# -----------------------------
# Train / eval loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: MultitaskLoss,
    device: str,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_tox_loss = 0.0
    running_dia_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tox_labels = batch["toxicity_label"].to(device)
        dia_labels = batch["dialect_label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, loss_dict = loss_fn(
            outputs["toxicity_logits"],
            outputs["dialect_logits"],
            tox_labels,
            dia_labels,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_tox_loss += loss_dict["tox_loss"]
        running_dia_loss += loss_dict["dialect_loss"]

        print(
            f"Batch {len(dataloader)} | "
            f"loss={loss_dict['total_loss']:.4f} | "
            f"tox_loss={loss_dict['tox_loss']:.4f} | "
            f"dia_loss={loss_dict['dialect_loss']:.4f}",
            end="\r",
        )

    n = max(1, len(dataloader))
    return {
        "loss": running_loss / n,
        "tox_loss": running_tox_loss / n,
        "dialect_loss": running_dia_loss / n,
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate toxicity performance and fairness metrics."""
    model.eval()

    tox_true: List[int] = []
    tox_prob: List[float] = []
    group: List[int] = []
    dia_true: List[int] = []
    dia_prob: List[float] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_tox = batch["toxicity_label"].cpu().numpy().astype(int)
        batch_dia = batch["dialect_label"].cpu().numpy().astype(int)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_tox_prob = torch.sigmoid(outputs["toxicity_logits"]).cpu().numpy()
        batch_dia_prob = torch.sigmoid(outputs["dialect_logits"]).cpu().numpy()

        tox_true.extend(batch_tox.tolist())
        tox_prob.extend(batch_tox_prob.tolist())
        group.extend(batch_dia.tolist())
        dia_true.extend(batch_dia.tolist())
        dia_prob.extend(batch_dia_prob.tolist())

    tox_true_arr = np.array(tox_true)
    tox_prob_arr = np.array(tox_prob)
    group_arr = np.array(group)

    tox_metrics = compute_binary_metrics(tox_true_arr, tox_prob_arr, threshold=threshold)
    fair_metrics = group_fairness_metrics(tox_true_arr, tox_prob_arr, group_arr, threshold=threshold)
    dia_metrics = compute_binary_metrics(np.array(dia_true), np.array(dia_prob), threshold=0.5)

    print(f"Eval results at threshold {threshold:.2f} | "
          f"toxicity accuracy: {tox_metrics['accuracy']:.4f} | "
          f"toxicity F1: {tox_metrics['f1']:.4f} | "
          f"FPR gap: {fair_metrics['fpr_gap']:.4f} | "
          f"FNR gap: {fair_metrics['fnr_gap']:.4f} | "
          f"dialect accuracy: {dia_metrics['accuracy']:.4f} | "
          f"dialect F1: {dia_metrics['f1']:.4f}")
    return {
        **tox_metrics,
        **fair_metrics,
        "dialect_accuracy": dia_metrics["accuracy"],
        "dialect_f1": dia_metrics["f1"],
    }


# -----------------------------
# Main training pipeline
# -----------------------------

def fit_model(train_df: pd.DataFrame, val_df: pd.DataFrame, config: Config) -> Tuple[nn.Module, Dict[str, float]]:
    """Train the multitask model and keep the best checkpoint by validation F1."""
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds = MultitaskTextDataset(train_df, tokenizer, max_length=config.max_length)
    val_ds = MultitaskTextDataset(val_df, tokenizer, max_length=config.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = MultitaskBERT(config.model_name, hidden_dropout=config.hidden_dropout).to(config.device)
    loss_fn = MultitaskLoss(lambda_dialect=config.lambda_dialect)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(config.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, config.device)
        val_stats = evaluate(model, val_loader, config.device)

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_acc={val_stats['accuracy']:.4f} | "
            f"val_f1={val_stats['f1']:.4f} | "
            f"val_fpr_gap={val_stats['fpr_gap']:.4f} | "
            f"val_fnr_gap={val_stats['fnr_gap']:.4f} | "
            f"val_dialect_acc={val_stats['dialect_accuracy']:.4f} | "
            f"val_dialect_f1={val_stats['dialect_f1']:.4f}"
        )

        # Keep the model that performs best on validation toxicity F1.
        if val_stats["f1"] > best_val_f1:
            best_val_f1 = val_stats["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_stats

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, config.save_path)
        print(f"Saved best checkpoint to: {config.save_path}")

    return model, best_metrics


# -----------------------------
# Utility for test evaluation
# -----------------------------

def evaluate_on_split(model: nn.Module, df: pd.DataFrame, config: Config) -> Dict[str, float]:
    """Run evaluation on a single split DataFrame."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ds = MultitaskTextDataset(df, tokenizer, max_length=config.max_length)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return evaluate(model, loader, config.device)


# -----------------------------
# CLI / entry point
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multitask BERT model for toxicity + dialect.")
    parser.add_argument("--train_csv", type=str, default="../data/processed/train.csv", help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="../data/processed/val.csv", help="Path to validation CSV file")
    parser.add_argument("--test_csv", type=str, default="../data/processed/test.csv", help="Path to test CSV file")
    parser.add_argument("--model_name", type=str, default="unitary/toxic-bert", help="HF model checkpoint")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_dialect", type=float, default=0.5)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="best_multitask_bert.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_dialect=args.lambda_dialect,
        hidden_dropout=args.hidden_dropout,
        seed=args.seed,
        save_path=args.save_path,
    )

    print("Loading data...")
    train_df = load_csv(args.train_csv)
    val_df = load_csv(args.val_csv)
    test_df = load_csv(args.test_csv)

    print("Train label counts:\n", train_df[["toxicity_label", "dialect_label"]].value_counts(dropna=False))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    print("\nTraining model...")
    model, best_val_metrics = fit_model(train_df, val_df, config)

    print("\nBest validation metrics:")
    for k, v in best_val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_on_split(model, test_df, config)

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Optional: save metrics to a CSV for later reporting.
    metrics_df = pd.DataFrame([{"split": "val", **best_val_metrics}, {"split": "test", **test_metrics}])
    metrics_df.to_csv("multitask_metrics.csv", index=False)
    print("\nSaved metrics to multitask_metrics.csv")


if __name__ == "__main__":
    main()
