"""XGBoost fairness mitigation with a covariance-based custom objective.

This script trains XGBoost multiple times using:

    L = (1 - gamma) * L_BCE + gamma * L_fair

where the fairness term is a smooth covariance surrogate targeted at reducing
AAE false positives:

    L_fair = Cov( (1 - y) * p , a_centered )^2

with:
    - p = sigmoid(raw_margin)
    - y = toxicity label (0/1)
    - a = protected group indicator (AAE=1, SAE=0)

The script:
1. Loads embeddings (.npy) and CSV metadata.
2. Trains a baseline XGBoost model for each gamma value.
3. Evaluates each model on the test set.
4. Saves predictions and a summary metrics table.
5. Saves plots for FPR, FNR, and performance vs gamma.

Expected files:
    ../data/embeddings/train_emb.npy
    ../data/embeddings/val_emb.npy
    ../data/embeddings/test_emb.npy
    ../data/processed/train.csv
    ../data/processed/val.csv
    ../data/processed/test.csv

Expected CSV columns:
    - label
    - dialect_strict   (AAE or SAE)

Important:
    - This is a smooth surrogate fairness objective, not exact FPR-gap optimization.
    - The Hessian for the fairness term is a stable approximation (Gauss-Newton style).
    - Models are saved as booster JSON files to avoid pickling custom objective callables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# =========================
# Configuration
# =========================
EMB_DIR = Path("../../data/embeddings")
DATA_DIR = Path("../../data/processed/twitterAAE")
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/fair_xgb/CE_FPR_COV_loss")
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

# If you want faster runs, reduce n_estimators or gamma grid.
XGB_PARAMS = dict(
    max_depth=50,
    n_estimators=150,
    learning_rate=0.005,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",  # overridden by callable custom objective
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# =========================
# Utilities
# =========================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load embeddings and CSV metadata."""
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def get_group(df: pd.DataFrame) -> np.ndarray:
    """Map dialect_strict to group labels.

    AAE -> 1
    SAE -> 0
    """
    group = df["dialect_strict"].astype(str).str.strip().str.upper().map({"AAE": 1, "SAE": 0})
    if group.isna().any():
        bad_vals = df.loc[group.isna(), "dialect_strict"].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values found: {bad_vals}")
    return group.values.astype(int)


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    """Compute FPR/FNR by group and gaps."""
    out: Dict[str, float] = {}
    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = group == g
        out[f"fpr_{name}"] = fpr(y_true[idx], y_pred[idx])
        out[f"fnr_{name}"] = fnr(y_true[idx], y_pred[idx])
    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# =========================
# Covariance fairness objective
# =========================

@dataclass
class CovFairObjective:
    """Custom objective implementing:

        L = (1 - gamma) * BCE + gamma * Cov((1-y) * p, a_centered)^2

    Here:
        - y_true is the toxicity label vector.
        - group_train must be aligned with the training rows in the same order.
        - p is sigmoid(raw_margin).
    """

    gamma: float
    group_train: np.ndarray

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        # Raw margins from XGBoost.
        p = sigmoid(y_pred)

        # -----------------------------
        # Task loss: binary cross entropy
        #   grad = p - y
        #   hess = p * (1 - p)
        # -----------------------------
        grad_task = p - y_true
        hess_task = p * (1.0 - p)

        # -----------------------------
        # Fairness surrogate:
        #   cov = mean( ((1-y) * p) * a_centered )
        #   L_fair = cov^2
        #
        # This specifically targets non-toxic examples, because the observed
        # problem is elevated false positives on AAE.
        # -----------------------------
        a = self.group_train.astype(float)
        a_centered = a - a.mean()

        neg_mask = (y_true == 0).astype(float)
        n = len(y_true)

        # x_i = (1 - y_i) * p_i
        cov = np.mean(neg_mask * p * a_centered)

        # d cov / d z_i = (1/n) * (1-y_i) * a_centered_i * dp/dz_i
        dp_dz = p * (1.0 - p)
        d_cov = (neg_mask * a_centered * dp_dz) / n

        # Gradient of squared covariance:
        #   d L_fair / d z_i = 2 * cov * d_cov_i
        grad_fair = 2.0 * cov * d_cov

        # Stable Hessian approximation (Gauss-Newton style):
        #   hess ~ 2 * (d_cov^2)
        hess_fair = 2.0 * (d_cov ** 2)

        # Combine task + fairness components.
        grad = (1.0 - self.gamma) * grad_task + self.gamma * grad_fair
        hess = (1.0 - self.gamma) * hess_task + self.gamma * hess_fair

        # Numerical safety.
        hess = np.clip(hess, 1e-6, None)
        return grad, hess


# =========================
# Training / Evaluation
# =========================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    objective,
) -> XGBClassifier:
    model = XGBClassifier(**XGB_PARAMS)
    model.set_params(objective=objective)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_model(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    prob = model.predict_proba(X)[:, 1]
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

    g_train = get_group(train_df)
    g_val = get_group(val_df)
    g_test = get_group(test_df)

    # Quick sanity checks.
    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nTrain joint distribution:\n", pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    results: List[Dict[str, float]] = []

    for gamma in np.arange(0, 1, 1/50):
        print(f"\n=== Training gamma={gamma:.2f} ===")

        objective = CovFairObjective(gamma=gamma, group_train=g_train)
        model = train_model(X_train, y_train, X_val, y_val, objective)

        # Save the trained booster as JSON. Avoid joblib/pickle because the
        # custom objective is a local callable and is not pickle-friendly.
        booster_path = MODELS_DIR / f"xgb_cov_fair_gamma_{str(gamma).replace('.', '_')}.json"
        model.get_booster().save_model(str(booster_path))

        # Save metadata for reproducibility.
        meta_path = MODELS_DIR / f"xgb_cov_fair_gamma_{str(gamma).replace('.', '_')}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gamma": gamma,
                    "xgb_params": XGB_PARAMS,
                    "objective": "(1-gamma)*BCE + gamma*Cov((1-y)p, a_centered)^2",
                },
                f,
                indent=2,
            )

        # Evaluate on validation and test.
        val_metrics, val_prob, val_pred = evaluate_model(model, X_val, y_val, g_val)
        test_metrics, test_prob, test_pred = evaluate_model(model, X_test, y_test, g_test)

        print(
            f"gamma={gamma:.2f} | "
            f"val_f1={val_metrics['f1']:.4f} | test_f1={test_metrics['f1']:.4f} | "
            f"test_fpr_gap={test_metrics['fpr_gap']:.4f} | test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        # Save test predictions for this gamma.
        pred_df = test_df.copy()
        pred_df["xgb_prob"] = test_prob
        pred_df["xgb_pred"] = test_pred
        pred_df.to_csv(
            RESULTS_DIR / f"xgb_cov_fair_gamma_{str(gamma).replace('.', '_')}_predictions.csv",
            index=False,
        )

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
    results_df.to_csv(RESULTS_DIR / "xgb_cov_fair_gamma_sweep_summary.csv", index=False)
    print("\nSaved summary to:", RESULTS_DIR / "xgb_cov_fair_gamma_sweep_summary.csv")

    # =========================
    # Plots
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_gap"], marker="o", label="FPR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Positive Rate")
    plt.title("FPR vs Gamma (Covariance objective)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_cov_fair_gamma_fpr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_gap"], marker="o", label="FNR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Negative Rate")
    plt.title("FNR vs Gamma (Covariance objective)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_cov_fair_gamma_fnr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_f1"], marker="o", label="Test F1")
    plt.plot(results_df["gamma"], results_df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Performance vs Gamma (Covariance objective)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_cov_fair_gamma_performance.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print(" -", RESULTS_DIR / "xgb_cov_fair_gamma_fpr.png")
    print(" -", RESULTS_DIR / "xgb_cov_fair_gamma_fnr.png")
    print(" -", RESULTS_DIR / "xgb_cov_fair_gamma_performance.png")

    # Helpful summary row for reporting.
    best = results_df.sort_values(["test_fpr_gap", "test_f1"], ascending=[True, False]).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()
