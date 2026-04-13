"""XGBoost + group-aware threshold sweep using a composite score.

Goal
----
Pick separate thresholds for AAE and SAE that balance:
    - fairness: FPR gap between AAE and SAE
    - performance: validation F1

Composite score
---------------
    score = lambda_fairness * FPR_gap + (1 - lambda_fairness) * (1 - F1)

Lower score is better.

This script:
1. Loads train/val/test embeddings and CSVs.
2. Trains one XGBoost model on embeddings.
3. Sweeps multiple lambda_fairness values.
4. For each lambda, searches thresholds (t_AAE, t_SAE) on validation data.
5. Applies the selected thresholds to test data.
6. Saves predictions for each lambda.
7. Saves a summary metrics table.
8. Produces two line plots:
      - FPR metrics vs lambda
      - FNR metrics vs lambda

Expected files
--------------
../data/embeddings/train_emb.npy
../data/embeddings/val_emb.npy
../data/embeddings/test_emb.npy
../data/processed/train.csv
../data/processed/val.csv
../data/processed/test.csv

Expected CSV columns
--------------------
- label
- dialect_strict  (AAE or SAE)

Notes
-----
- This is group-aware thresholding, so you must know dialect at prediction time.
- The same trained XGBoost model is used for all lambda values.
- Threshold search is done only on validation data.
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

# Threshold grid for (t_AAE, t_SAE)
THRESHOLDS = np.linspace(0.05, 0.95, 37)  # 0.05, 0.075, ... , 0.95

# Sweep fairness-performance tradeoff weight.
#   lambda_fairness = 1.0 -> focus only on FPR gap
#   lambda_fairness = 0.0 -> focus only on F1
LAMBDA_VALUES = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0]

# Optional minimum validation F1 floor.
# Set to None to disable. Example: 0.70.
MIN_VAL_F1 = None

# XGBoost hyperparameters (fixed across the sweep)
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

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load embeddings and processed CSV files."""
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def get_group(df: pd.DataFrame) -> np.ndarray:
    """Map dialect_strict values to group labels.

    AAE -> 1
    SAE -> 0
    """
    group = df["dialect_strict"].astype(str).str.strip().str.upper().map({"AAE": 1, "SAE": 0})
    if group.isna().any():
        bad_vals = df.loc[group.isna(), "dialect_strict"].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values: {bad_vals}")
    return group.values.astype(int)


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False positive rate."""
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False negative rate."""
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def apply_group_thresholds(probs: np.ndarray, group: np.ndarray, t_aae: float, t_sae: float) -> np.ndarray:
    """Apply different thresholds for AAE and SAE."""
    preds = np.zeros_like(probs, dtype=int)
    preds[(group == 1) & (probs >= t_aae)] = 1
    preds[(group == 0) & (probs >= t_sae)] = 1
    return preds


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    """Compute FPR/FNR by group and their gaps."""
    out: Dict[str, float] = {}

    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = group == g
        out[f"fpr_{name}"] = fpr(y_true[idx], y_pred[idx])
        out[f"fnr_{name}"] = fnr(y_true[idx], y_pred[idx])

    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


def composite_score(lambda_fairness: float, fpr_gap: float, f1: float) -> float:
    """Composite objective to minimize.

    Lower is better.
    """
    return lambda_fairness * fpr_gap + (1.0 - lambda_fairness) * (1.0 - f1)


def search_thresholds(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    val_group: np.ndarray,
    lambda_fairness: float,
    thresholds: np.ndarray,
    min_val_f1: float | None = None,
) -> Dict[str, float]:
    """Search over threshold pairs on validation data."""
    best: Dict[str, float] | None = None

    for t_aae in thresholds:
        for t_sae in thresholds:
            val_pred = apply_group_thresholds(val_prob, val_group, t_aae, t_sae)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

            if min_val_f1 is not None and val_f1 < min_val_f1:
                continue

            metrics = compute_group_metrics(y_val, val_pred, val_group)
            score = composite_score(lambda_fairness, metrics["fpr_gap"], val_f1)

            candidate = {
                "t_aae": float(t_aae),
                "t_sae": float(t_sae),
                "val_f1": float(val_f1),
                "score": float(score),
                **metrics,
            }

            if best is None:
                best = candidate
                continue

            # Lower score is better. Tie-breakers: lower FPR gap, then higher F1.
            if (
                candidate["score"] < best["score"]
                or (np.isclose(candidate["score"], best["score"]) and candidate["fpr_gap"] < best["fpr_gap"])
                or (
                    np.isclose(candidate["score"], best["score"])
                    and np.isclose(candidate["fpr_gap"], best["fpr_gap"])
                    and candidate["val_f1"] > best["val_f1"]
                )
            ):
                best = candidate

    if best is None:
        raise RuntimeError(
            "No valid threshold pair found. Try lowering MIN_VAL_F1 or expanding the threshold grid."
        )

    return best


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> XGBClassifier:
    """Train the XGBoost baseline once."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# =========================
# MAIN EXPERIMENT
# =========================

def main() -> None:
    X_train, X_val, X_test, train_df, val_df, test_df = load_data()

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    train_group = get_group(train_df)
    val_group = get_group(val_df)
    test_group = get_group(test_df)

    # Basic sanity checks for the report.
    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nTrain joint distribution:\n", pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    # Train one base model.
    print("\nTraining baseline XGBoost...")
    model = train_xgb(X_train, y_train, X_val, y_val)

    # Save baseline model.
    joblib.dump(model, MODELS_DIR / "xgb_baseline.joblib")

    # Probabilities from the trained model.
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    base_val_pred = (val_prob >= 0.5).astype(int)
    base_test_pred = (test_prob >= 0.5).astype(int)
    base_test_f1 = f1_score(y_test, base_test_pred, zero_division=0)

    print(f"Baseline test F1 @0.5 threshold: {base_test_f1:.4f}")

    results: List[Dict[str, float]] = []

    # Sweep over lambda values.
    for lam in LAMBDA_VALUES:
        print(f"\n=== Searching thresholds for lambda_fairness={lam} ===")
        best = search_thresholds(
            y_val=y_val,
            val_prob=val_prob,
            val_group=val_group,
            lambda_fairness=lam,
            thresholds=THRESHOLDS,
            min_val_f1=MIN_VAL_F1,
        )

        # Evaluate selected thresholds on the test set.
        test_pred = apply_group_thresholds(test_prob, test_group, best["t_aae"], best["t_sae"])
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        test_acc = accuracy_score(y_test, test_pred)
        test_metrics = compute_group_metrics(y_test, test_pred, test_group)

        row = {
            "lambda_fairness": lam,
            "t_aae": best["t_aae"],
            "t_sae": best["t_sae"],
            "val_f1": best["val_f1"],
            "val_score": best["score"],
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results.append(row)

        print(
            f"lambda={lam} | t_aae={best['t_aae']:.2f} | t_sae={best['t_sae']:.2f} | "
            f"val_f1={best['val_f1']:.4f} | test_f1={test_f1:.4f} | "
            f"test_fpr_gap={test_metrics['fpr_gap']:.4f} | test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        # Save predictions for this lambda.
        pred_df = test_df.copy()
        pred_df["xgb_prob"] = test_prob
        pred_df["xgb_pred"] = test_pred
        pred_df.to_csv(
            RESULTS_DIR / f"xgb_threshold_predictions_lambda_{str(lam).replace('.', '_')}.csv",
            index=False,
        )

    results_df = pd.DataFrame(results).sort_values("lambda_fairness").reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "xgb_threshold_sweep_summary.csv", index=False)
    print("\nSaved summary table to:", RESULTS_DIR / "xgb_threshold_sweep_summary.csv")

    # =========================
    # PLOTS
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["lambda_fairness"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(results_df["lambda_fairness"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE")
    plt.xlabel("Lambda (fairness weight)")
    plt.ylabel("False Positive Rate")
    plt.title("FPR vs Composite-Score Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_threshold_fpr_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["lambda_fairness"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(results_df["lambda_fairness"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE")
    plt.xlabel("Lambda (fairness weight)")
    plt.ylabel("False Negative Rate")
    plt.title("FNR vs Composite-Score Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_threshold_fnr_plot.png", dpi=300)
    plt.close()

    print("Saved plots to:")
    print(" -", RESULTS_DIR / "xgb_threshold_fpr_plot.png")
    print(" -", RESULTS_DIR / "xgb_threshold_fnr_plot.png")

    # Print a concise best row for convenience.
    best_row = results_df.sort_values(["test_fpr_gap", "test_f1"], ascending=[True, False]).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best_row.to_dict())


if __name__ == "__main__":
    main()
