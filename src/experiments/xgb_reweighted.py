"""XGBoost reweighting sweep for fairness analysis.

This script:
1. Loads embeddings for train/val/test.
2. Loads processed CSVs with labels and dialect labels.
3. Trains XGBoost multiple times with different sample-weight values.
4. Evaluates each model on the test set.
5. Saves predictions for each weight setting.
6. Produces two line plots:
      - FPR vs weight
      - FNR vs weight

Expected files:
    ../data/embeddings/train_emb.npy
    ../data/embeddings/val_emb.npy
    ../data/embeddings/test_emb.npy
    ../data/processed/train.csv
    ../data/processed/val.csv
    ../data/processed/test.csv

Expected columns in CSVs:
    - label
    - dialect_strict   (values: AAE or SAE)

Weighting rule:
    Increase weight for AAE non-toxic samples (dialect_strict == AAE and label == 0).
    This targets the group most often over-flagged as toxic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
EMB_DIR = Path("../../data/embeddings")
DATA_DIR = Path("../../data/processed/twitterAAE")
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/xgb")
MODELS_DIR = Path("../../models")

TRAIN_EMB = EMB_DIR / "train_emb.npy"
VAL_EMB = EMB_DIR / "val_emb.npy"
TEST_EMB = EMB_DIR / "test_emb.npy"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WEIGHT_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 5.0]

# XGBoost hyperparameters (keep the same across sweep for fairness comparison)
XGB_PARAMS = dict(
    max_depth=5,
    n_estimators=100,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# =========================
# HELPERS
# =========================

def load_embeddings_and_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load embeddings and processed CSVs."""
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def normalize_dialect(series: pd.Series) -> np.ndarray:
    """Map AAE -> 1, SAE -> 0."""
    mapped = series.astype(str).str.strip().str.upper().map({"AAE": 1, "SAE": 0})
    if mapped.isna().any():
        bad_vals = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values found: {bad_vals}")
    return mapped.values.astype(int)


def compute_sample_weights(dialect: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    """Weight AAE non-toxic examples more heavily.

    Args:
        dialect: 1 for AAE, 0 for SAE
        labels: 1 for toxic, 0 for non-toxic
        alpha: weight assigned to AAE non-toxic samples
    """
    weights = np.ones(len(labels), dtype=float)
    mask = (dialect == 1) & (labels == 0)
    weights[mask] = alpha
    return weights


def fpr_fnr_by_group(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    """Compute FPR/FNR for AAE and SAE."""
    out: Dict[str, float] = {}

    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = group == g
        yt = y_true[idx]
        yp = y_pred[idx]

        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))
        tp = np.sum((yt == 1) & (yp == 1))

        out[f"fpr_{name}"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        out[f"fnr_{name}"] = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


def train_and_eval_one_alpha(
    alpha: float,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, XGBClassifier]:
    """Train one XGBoost model for a given alpha and return metrics + predictions."""
    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    train_dialect = normalize_dialect(train_df["dialect_strict"])
    val_dialect = normalize_dialect(val_df["dialect_strict"])
    test_dialect = normalize_dialect(test_df["dialect_strict"])

    train_weights = compute_sample_weights(train_dialect, y_train, alpha=alpha)
    val_weights = compute_sample_weights(val_dialect, y_val, alpha=alpha)

    model = XGBClassifier(**XGB_PARAMS)

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[val_weights],
        verbose=False,
    )

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    fairness = fpr_fnr_by_group(y_test, test_pred, test_dialect)

    metrics = {
        "alpha": alpha,
        "accuracy": acc,
        "f1": f1,
        **fairness,
    }

    return metrics, test_prob, test_pred, model


# =========================
# MAIN
# =========================

def main() -> None:
    X_train, X_val, X_test, train_df, val_df, test_df = load_embeddings_and_data()

    # Sanity checks
    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nJoint distribution (train):")
    print(pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    all_metrics: List[Dict[str, float]] = []

    for alpha in WEIGHT_VALUES:
        print(f"\n=== Training XGBoost with alpha={alpha} ===")
        metrics, test_prob, test_pred, model = train_and_eval_one_alpha(
            alpha=alpha,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        print(
            f"alpha={alpha} | "
            f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | "
            f"FPR_AAE={metrics['fpr_AAE']:.4f} | FPR_SAE={metrics['fpr_SAE']:.4f} | "
            f"FNR_AAE={metrics['fnr_AAE']:.4f} | FNR_SAE={metrics['fnr_SAE']:.4f}"
        )

        # Save model for this alpha
        model_path = MODELS_DIR / f"xgb_reweighted_alpha_{str(alpha).replace('.', '_')}.joblib"
        joblib.dump(model, model_path)

        # Save test predictions for this alpha
        pred_df = test_df.copy()
        pred_df["xgb_prob"] = test_prob
        pred_df["xgb_pred"] = test_pred
        pred_df.to_csv(RESULTS_DIR / f"xgb_predictions_alpha_{str(alpha).replace('.', '_')}.csv", index=False)

        all_metrics.append(metrics)

    # Save summary table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("alpha").reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "xgb_reweight_sweep_metrics.csv", index=False)
    print("\nSaved metrics table to:", RESULTS_DIR / "xgb_reweight_sweep_metrics.csv")

    # =========================
    # PLOTS
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["alpha"], metrics_df["fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(metrics_df["alpha"], metrics_df["fpr_SAE"], marker="o", label="FPR SAE")
    plt.xlabel("AAE non-toxic weight (alpha)")
    plt.ylabel("FPR")
    plt.title("FPR vs Re-weighting Strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_fpr_vs_alpha.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["alpha"], metrics_df["fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(metrics_df["alpha"], metrics_df["fnr_SAE"], marker="o", label="FNR SAE")
    plt.xlabel("AAE non-toxic weight (alpha)")
    plt.ylabel("FNR")
    plt.title("FNR vs Re-weighting Strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_fnr_vs_alpha.png", dpi=300)
    plt.close()

    print("Saved plots to:")
    print(" -", RESULTS_DIR / "xgb_fpr_vs_alpha.png")
    print(" -", RESULTS_DIR / "xgb_fnr_vs_alpha.png")

    # Print best alpha according to smallest FPR gap, then highest F1 as a tie-breaker.
    best = metrics_df.sort_values(["fpr_gap", "-f1"], ascending=[True, False]).iloc[0]
    print("\nBest setting by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()
