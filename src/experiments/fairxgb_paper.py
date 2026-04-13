"""FairXGBoost-style objective for dialect bias mitigation in toxicity detection.

This script implements the paper-style regularizer used by FairXGBoost:

    L = BCE(y, p) - gamma * BCE(s, p)

where:
    - y is the toxicity label (0/1)
    - s is the sensitive attribute label (0/1)
    - p = sigmoid(raw_margin)

For this project, we encode the protected dialect group as 0 and the other
(group / majority) dialect as 1. That choice makes the regularizer push the
model to reduce toxic scores on the protected dialect group.

Why this works as a custom XGBoost objective:
    - XGBoost custom objectives are trained from gradients and Hessians.
    - The FairXGBoost paper derives a gradient/hessian form that fits the
      existing XGBoost training loop with minimal modification.

Inputs expected:
    ../data/embeddings/train_emb.npy
    ../data/embeddings/val_emb.npy
    ../data/embeddings/test_emb.npy
    ../data/processed/train.csv
    ../data/processed/val.csv
    ../data/processed/test.csv

Expected CSV columns:
    - label
    - dialect_strict   (AAE or SAE)

Outputs:
    - one model JSON per gamma
    - one predictions CSV per gamma
    - a summary CSV
    - plots for FPR / FNR / accuracy / F1 vs gamma

Notes:
    - Gamma should stay in [0, 1). At gamma=1 the Hessian becomes zero.
    - This reproduces the paper's objective form, adapted to your dialect task.
    - Save models as JSON rather than joblib because the custom objective is a
      Python callable and should not be pickled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


# =========================
# Configuration
# =========================
EMB_DIR = Path("../../data/embeddings")
DATA_DIR = Path("../../data/processed/twitterAAE")
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/fairxgb_paper")
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
NUM_BOOST_ROUND = 150

XGB_PARAMS = {
    "max_depth": 50,
    "eta": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "seed": SEED,
    "objective": "binary:logistic",  # evaluation remains binary classification
    "eval_metric": "logloss",
}


# =========================
# Helpers
# =========================

def sigmoid(x):
    out = np.zeros_like(x)
    
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)

    return out


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def encode_sensitive(df: pd.DataFrame) -> np.ndarray:
    """Encode dialect_strict as a binary sensitive label.

    We use:
        AAE -> 0
        SAE -> 1

    This follows the FairXGBoost-style regularizer form, while making the
    adaptation useful for reducing AAE false positives.
    """
    mapped = df["dialect_strict"].astype(str).str.strip().str.upper().map({"AAE": 0, "SAE": 1})
    if mapped.isna().any():
        bad = df.loc[mapped.isna(), "dialect_strict"].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values: {bad}")
    return mapped.values.astype(int)


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, g in [("AAE", 0), ("SAE", 1)]:
        idx = group == g
        out[f"fpr_{name}"] = fpr(y_true[idx], y_pred[idx])
        out[f"fnr_{name}"] = fnr(y_true[idx], y_pred[idx])
    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# =========================
# FairXGBoost-style objective
# =========================

def make_fairxgb_objective(sensitive_train: np.ndarray, gamma: float) -> Callable:
    """Create the custom objective.

    Paper-style form:
        L = BCE(y, p) - gamma * BCE(s, p)

    With p = sigmoid(raw_margin), the gradients are:
        grad = (p - y) + gamma * (s - p)
        hess = (1 - gamma) * p * (1 - p)

    This matches the paper's derivation:
        g_bar = g + mu (s - sigmoid(z))
        h_bar = h (1 - mu)
    """

    sensitive_train = sensitive_train.astype(float)

    def objective(preds: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label().astype(float)
        p = sigmoid(preds)

        # Standard BCE gradients / Hessians.
        grad_task = p - y_true
        hess_task = p * (1.0 - p)

        # Fairness regularizer gradients / Hessians.
        # BCE(s, p) gradient is (p - s), so the negative regularizer contributes (s - p).
        grad_fair = sensitive_train - p
        hess_fair = -p * (1.0 - p)

        grad = grad_task + gamma * grad_fair
        hess = hess_task + gamma * hess_fair

        # The paper's closed-form simplifies hess to (1 - gamma) * hess_task.
        # Clipping keeps training numerically safe for values close to 1.
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return objective


# =========================
# Training / evaluation
# =========================

def train_model(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    gamma: float,
    sensitive_train: np.ndarray,
) -> xgb.Booster:
    obj = make_fairxgb_objective(sensitive_train=sensitive_train, gamma=gamma)

    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        obj=obj,
        evals=[(dval, "val")],
        verbose_eval=False,
    )
    return booster


def predict_prob(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat, output_margin=True)
    return sigmoid(raw)


def evaluate_split(booster: xgb.Booster, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    prob = predict_prob(booster, X)
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        **group_metrics(y, pred, group),
    }
    return metrics, prob, pred


# =========================
# Main sweep
# =========================

def main() -> None:
    X_train, X_val, X_test, train_df, val_df, test_df = load_data()

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    s_train = encode_sensitive(train_df)
    s_val = encode_sensitive(val_df)
    s_test = encode_sensitive(test_df)

    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nTrain joint distribution:\n", pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    results: List[Dict[str, float]] = []

    for gamma in np.arange(0, 1, 1/50):
        print(f"\n=== Training gamma={gamma:.2f} ===")
        booster = train_model(dtrain=dtrain, dval=dval, gamma=gamma, sensitive_train=s_train)

        # Save booster and metadata.
        booster_path = MODELS_DIR / f"fairxgb_gamma_{str(gamma).replace('.', '_')}.json"
        booster.save_model(str(booster_path))

        meta_path = MODELS_DIR / f"fairxgb_gamma_{str(gamma).replace('.', '_')}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gamma": gamma,
                    "objective": "BCE(y,p) - gamma * BCE(s,p)",
                    "sensitive_encoding": {"AAE": 0, "SAE": 1},
                    "xgb_params": XGB_PARAMS,
                    "num_boost_round": NUM_BOOST_ROUND,
                },
                f,
                indent=2,
            )

        val_metrics, val_prob, val_pred = evaluate_split(booster, X_val, y_val, s_val)
        test_metrics, test_prob, test_pred = evaluate_split(booster, X_test, y_test, s_test)

        print(
            f"gamma={gamma:.2f} | "
            f"val_f1={val_metrics['f1']:.4f} | test_f1={test_metrics['f1']:.4f} | "
            f"test_fpr_gap={test_metrics['fpr_gap']:.4f} | test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        # Save test predictions.
        pred_df = test_df.copy()
        pred_df["xgb_prob"] = test_prob
        pred_df["xgb_pred"] = test_pred
        pred_df.to_csv(RESULTS_DIR / f"fairxgb_gamma_{str(gamma).replace('.', '_')}_predictions.csv", index=False)

        results.append(
            {
                "gamma": gamma,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_fpr_AAE": val_metrics["fpr_AAE"],
                "val_fpr_SAE": val_metrics["fpr_SAE"],
                "val_fpr_gap": val_metrics["fpr_gap"],
                "val_fnr_AAE": val_metrics["fnr_AAE"],
                "val_fnr_SAE": val_metrics["fnr_SAE"],
                "val_fnr_gap": val_metrics["fnr_gap"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_fpr_AAE": test_metrics["fpr_AAE"],
                "test_fpr_SAE": test_metrics["fpr_SAE"],
                "test_fpr_gap": test_metrics["fpr_gap"],
                "test_fnr_AAE": test_metrics["fnr_AAE"],
                "test_fnr_SAE": test_metrics["fnr_SAE"],
                "test_fnr_gap": test_metrics["fnr_gap"],
            }
        )

    results_df = pd.DataFrame(results).sort_values("gamma").reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "fairxgb_gamma_sweep_summary.csv", index=False)
    print("\nSaved summary to:", RESULTS_DIR / "fairxgb_gamma_sweep_summary.csv")

    # Plots.
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_gap"], marker="o", label="FPR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Positive Rate")
    plt.title("FPR vs Gamma (FairXGBoost-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairxgb_gamma_fpr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_gap"], marker="o", label="FNR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Negative Rate")
    plt.title("FNR vs Gamma (FairXGBoost-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairxgb_gamma_fnr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_f1"], marker="o", label="Test F1")
    plt.plot(results_df["gamma"], results_df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Performance vs Gamma (FairXGBoost-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fairxgb_gamma_performance.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print(" -", RESULTS_DIR / "fairxgb_gamma_fpr.png")
    print(" -", RESULTS_DIR / "fairxgb_gamma_fnr.png")
    print(" -", RESULTS_DIR / "fairxgb_gamma_performance.png")

    best = results_df.sort_values(["test_fpr_gap", "test_f1"], ascending=[True, False]).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()
