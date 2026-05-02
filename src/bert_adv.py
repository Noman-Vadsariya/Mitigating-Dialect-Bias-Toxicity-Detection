"""BERT + adversarial dialect debiasing + sample reweighting.

This script trains a toxicity classifier with a dialect adversary and adds
sample reweighting so AAE non-toxic examples count more during training.

Files expected:
    - train.csv
    - val.csv
    - test.csv

Required columns in each CSV:
    - tweet          : raw text
    - label          : toxicity label (0/1)
    - dialect_strict : dialect label stored as AAE/SAE

Mapping used:
    - AAE -> 1
    - SAE -> 0

Reweighting idea:
    - AAE + non-toxic examples get larger weight
    - This helps reduce FPR on AAE, which is your main fairness issue

Main components:
    1. Load CSVs and map labels
    2. Build BERT encoder
    3. Add toxicity head
    4. Add dialect adversary head with gradient reversal
    5. Compute weighted toxicity loss
    6. Train, validate, test
    7. Save metrics and best checkpoint
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    """Hyperparameters and runtime settings."""

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    adv_lambda: float = 1.0
    hidden_dropout: float = 0.1
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_bert_adversarial_weighted.pt"
    aae_non_toxic_weight: float = 2.0


# -----------------------------
# Data loading
# -----------------------------

def load_csv(path: str, aae_non_toxic_weight: float = 2.0) -> pd.DataFrame:
    """Load and normalize the dataset.

    Expected columns:
        - tweet
        - label
        - dialect_strict

    Toxicity label:
        - already 0/1

    Dialect label mapping:
        - AAE -> 1
        - SAE -> 0

    Reweighting:
        - AAE non-toxic examples get sample_weight = aae_non_toxic_weight
        - all others get sample_weight = 1.0
    """
    df = pd.read_csv(path)

    required = {"tweet", "label", "dialect_strict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df[["tweet", "label", "dialect_strict"]].copy()
    df = df.rename(
        columns={
            "tweet": "text",
            "label": "toxicity_label",
            "dialect_strict": "dialect_label",
        }
    )

    # Convert toxicity to numeric and dialect to 0/1.
    df["toxicity_label"] = pd.to_numeric(df["toxicity_label"], errors="coerce")
    df["dialect_label"] = df["dialect_label"].astype(str).str.strip().str.upper()
    df["dialect_label"] = df["dialect_label"].map({"AAE": 1, "SAE": 0})

    # Drop rows that cannot be used.
    df = df.dropna(subset=["text", "toxicity_label", "dialect_label"]).reset_index(drop=True)

    # Final clean typing.
    df["toxicity_label"] = df["toxicity_label"].astype(int)
    df["dialect_label"] = df["dialect_label"].astype(int)
    df["text"] = df["text"].astype(str)

    # Add sample weights.
    df["sample_weight"] = 1.0
    aae_non_toxic_mask = (df["dialect_label"] == 1) & (df["toxicity_label"] == 0)
    df.loc[aae_non_toxic_mask, "sample_weight"] = float(aae_non_toxic_weight)

    return df


class TextDataset(Dataset):
    """Tokenize text and return labels plus sample weights."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            str(row["text"]),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "toxicity_label": torch.tensor(row["toxicity_label"], dtype=torch.float),
            "dialect_label": torch.tensor(row["dialect_label"], dtype=torch.float),
            "sample_weight": torch.tensor(row["sample_weight"], dtype=torch.float),
        }


# -----------------------------
# Gradient reversal layer
# -----------------------------

class GradientReversalFunction(Function):
    """Identity forward pass, negative gradient backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# -----------------------------
# Model
# -----------------------------

class BERTAdversarialToxicityModel(nn.Module):
    """Shared BERT encoder with a toxicity head and an adversarial dialect head."""

    def __init__(self, model_name: str = "bert-base-uncased", hidden_dropout: float = 0.1, adv_lambda: float = 1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout)
        self.toxicity_head = nn.Linear(hidden_size, 1)
        self.dialect_head = nn.Linear(hidden_size, 1)
        self.grl = GradientReversal(lambda_=adv_lambda)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token embedding as the sentence representation.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(cls_embedding)

        # Main task: toxicity prediction.
        toxicity_logits = self.toxicity_head(features).squeeze(-1)

        # Adversarial task: dialect prediction through gradient reversal.
        reversed_features = self.grl(features)
        dialect_logits = self.dialect_head(reversed_features).squeeze(-1)

        return {
            "toxicity_logits": toxicity_logits,
            "dialect_logits": dialect_logits,
        }


# -----------------------------
# Loss
# -----------------------------

class WeightedAdversarialLoss(nn.Module):
    """Weighted toxicity loss + adversarial dialect loss.

    Toxicity loss is weighted example-by-example.
    Dialect loss is kept standard.
    """

    def __init__(self, adv_weight: float = 1.0):
        super().__init__()
        self.adv_weight = adv_weight
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        toxicity_logits: torch.Tensor,
        dialect_logits: torch.Tensor,
        toxicity_labels: torch.Tensor,
        dialect_labels: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Per-example toxicity loss.
        tox_loss_per_example = self.bce_none(toxicity_logits, toxicity_labels)

        # Weighted average toxicity loss.
        weighted_tox_loss = (tox_loss_per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

        # Dialect adversary loss.
        adv_loss = self.bce(dialect_logits, dialect_labels)

        total_loss = weighted_tox_loss + self.adv_weight * adv_loss

        return total_loss, {
            "tox_loss": float(weighted_tox_loss.item()),
            "adv_loss": float(adv_loss.item()),
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


def fairness_metrics(y_true: np.ndarray, y_prob: np.ndarray, group: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute group-wise FPR/FNR and gaps.

    group is the dialect label:
        1 = AAE
        0 = SAE
    """
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {}

    for name, value in [("AAE", 1), ("SAE", 0)]:
        idx = group == value
        if idx.sum() == 0:
            out[f"fpr_{name}"] = np.nan
            out[f"fnr_{name}"] = np.nan
            continue

        yt = y_true[idx]
        yp = y_pred[idx]

        tp = np.sum((yt == 1) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        out[f"fpr_{name}"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        out[f"fnr_{name}"] = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# -----------------------------
# Training / evaluation loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: WeightedAdversarialLoss,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    running_tox_loss = 0.0
    running_adv_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tox_labels = batch["toxicity_label"].to(device)
        dia_labels = batch["dialect_label"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, loss_dict = loss_fn(
            outputs["toxicity_logits"],
            outputs["dialect_logits"],
            tox_labels,
            dia_labels,
            sample_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_tox_loss += loss_dict["tox_loss"]
        running_adv_loss += loss_dict["adv_loss"]

    n = max(1, len(dataloader))
    return {
        "loss": running_loss / n,
        "tox_loss": running_tox_loss / n,
        "adv_loss": running_adv_loss / n,
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate toxicity performance, fairness metrics, and adversary quality."""
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
    fair_metrics = fairness_metrics(tox_true_arr, tox_prob_arr, group_arr, threshold=threshold)
    dia_metrics = compute_binary_metrics(np.array(dia_true), np.array(dia_prob), threshold=0.5)

    return {
        **tox_metrics,
        **fair_metrics,
        "dialect_accuracy": dia_metrics["accuracy"],
        "dialect_f1": dia_metrics["f1"],
    }


# -----------------------------
# Full training pipeline
# -----------------------------

def fit_model(train_df: pd.DataFrame, val_df: pd.DataFrame, config: Config) -> Tuple[nn.Module, Dict[str, float]]:
    """Train the model and keep the best checkpoint by validation toxicity F1."""
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds = TextDataset(train_df, tokenizer, max_length=config.max_length)
    val_ds = TextDataset(val_df, tokenizer, max_length=config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = BERTAdversarialToxicityModel(
        model_name=config.model_name,
        hidden_dropout=config.hidden_dropout,
        adv_lambda=config.adv_lambda,
    ).to(config.device)

    loss_fn = WeightedAdversarialLoss(adv_weight=1.0)
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
            f"val_dialect_acc={val_stats['dialect_accuracy']:.4f}"
        )

        # Keep the model with the best validation toxicity F1.
        if val_stats["f1"] > best_val_f1:
            best_val_f1 = val_stats["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_stats

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, config.save_path)
        print(f"Saved best checkpoint to: {config.save_path}")

    return model, best_metrics


def evaluate_split(model: nn.Module, df: pd.DataFrame, config: Config) -> Dict[str, float]:
    """Evaluate a DataFrame split."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ds = TextDataset(df, tokenizer, max_length=config.max_length)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return evaluate(model, loader, config.device)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT toxicity classifier with adversarial dialect debiasing and reweighting.")
    parser.add_argument("--train_csv", type=str, default="../data/processed/train.csv", help="Path to training CSV file")
    parser.add_argument("--val_csv", type=str, default="../data/processed/val.csv", help="Path to validation CSV file")
    parser.add_argument("--test_csv", type=str, default="../data/processed/test.csv", help="Path to test CSV file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adv_lambda", type=float, default=1.0)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--aae_non_toxic_weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="best_bert_adversarial_weighted.pt")
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
        adv_lambda=args.adv_lambda,
        hidden_dropout=args.hidden_dropout,
        seed=args.seed,
        save_path=args.save_path,
        aae_non_toxic_weight=args.aae_non_toxic_weight,
    )

    print("Loading data...")
    train_df = load_csv(args.train_csv, aae_non_toxic_weight=config.aae_non_toxic_weight)
    val_df = load_csv(args.val_csv, aae_non_toxic_weight=config.aae_non_toxic_weight)
    test_df = load_csv(args.test_csv, aae_non_toxic_weight=config.aae_non_toxic_weight)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    print("\nTraining label/group counts:")
    print(train_df[["toxicity_label", "dialect_label"]].value_counts(dropna=False))

    print("\nAverage sample weights in train split:")
    print(train_df["sample_weight"].value_counts().sort_index())

    print("\nTraining model...")
    model, best_val_metrics = fit_model(train_df, val_df, config)

    print("\nBest validation metrics:")
    for k, v in best_val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_split(model, test_df, config)

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics for your report.
    metrics_df = pd.DataFrame([
        {"split": "val", **best_val_metrics},
        {"split": "test", **test_metrics},
    ])
    metrics_df.to_csv("bert_adversarial_weighted_metrics.csv", index=False)
    print("\nSaved metrics to bert_adversarial_weighted_metrics.csv")


if __name__ == "__main__":
    main()
