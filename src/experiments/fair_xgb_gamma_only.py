"""XGBoost fairness mitigation with gamma-only squared fairness loss.

Objective:
    L = L_BCE + gamma * L_fair

Fairness term:
    L_fair = (mean_p_AAE_neg - mean_p_SAE_neg)^2

Where:
    - mean_p_AAE_neg = average predicted toxic probability on non-toxic AAE examples
    - mean_p_SAE_neg = average predicted toxic probability on non-toxic SAE examples

This objective is designed to reduce AAE false positives directly.

The script:
1. Loads train/val/test embeddings and metadata.
2. Trains XGBoost for multiple gamma values.
3. Evaluates each model on the test set.
4. Saves predictions, models, and summary metrics.
5. Produces plots for FPR, FNR, and performance vs gamma.

Expected inputs:
    - ../data/embeddings/train_emb.npy
    - ../data/embeddings/val_emb.npy
    - ../data/embeddings/test_emb.npy
    - ../data/processed/train.csv
    - ../data/processed/val.csv
    - ../data/processed/test.csv

Expected CSV columns:
    - label
    - dialect_strict   (AAE or SAE)

Important implementation notes:
    - This is a smooth surrogate fairness objective; it is not raw FPR-gap optimization.
    - The Hessian for the fairness term is an approximation for stability.
    - Models are saved as booster JSON to avoid pickling custom objective callables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/fair_xgb/CE_FPR_squared_loss_gamma")
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
    learning_rate=0.05,
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
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def get_group(df: pd.DataFrame) -> np.ndarray:
    """Map dialect_strict to group labels: AAE=1, SAE=0."""
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
    out: Dict[str, float] = {}
    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = group == g
        out[f"fpr_{name}"] = fpr(y_true[idx], y_pred[idx])
        out[f"fnr_{name}"] = fnr(y_true[idx], y_pred[idx])
    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# =========================
# Gamma-only squared fairness objective
# =========================

@dataclass
class GammaOnlySquaredFairObjective:
    """Objective: L = BCE + gamma * (mean_p_AAE_neg - mean_p_SAE_neg)^2."""

    gamma: float
    group_train: np.ndarray

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        # Raw margins from XGBoost
        p = sigmoid(y_pred)

        # Task loss: BCE
        grad_task = p - y_true
        hess_task = p * (1.0 - p)

        # Fairness term acts only on negative examples.
        group = self.group_train.astype(int)
        neg_mask = (y_true == 0)
        aae_neg = neg_mask & (group == 1)
        sae_neg = neg_mask & (group == 0)

        grad_fair = np.zeros_like(p)
        hess_fair = np.zeros_like(p)

        n_aae = int(np.sum(aae_neg))
        n_sae = int(np.sum(sae_neg))

        if n_aae > 0 and n_sae > 0:
            mean_aae = np.mean(p[aae_neg])
            mean_sae = np.mean(p[sae_neg])
            diff = mean_aae - mean_sae

            dp_dz = p * (1.0 - p)

            # Gradient of (mean_aae - mean_sae)^2
            grad_fair[aae_neg] = 2.0 * diff * dp_dz[aae_neg] / n_aae
            grad_fair[sae_neg] = -2.0 * diff * dp_dz[sae_neg] / n_sae

            # Stable Hessian approximation
            hess_fair[aae_neg] = 2.0 * (dp_dz[aae_neg] / n_aae) ** 2
            hess_fair[sae_neg] = 2.0 * (dp_dz[sae_neg] / n_sae) ** 2

        grad = grad_task + self.gamma * grad_fair
        hess = hess_task + self.gamma * hess_fair
        hess = np.clip(hess, 1e-6, None)

        return grad, hess


# =========================
# Training / Evaluation
# =========================

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, objective) -> XGBClassifier:
    model = XGBClassifier(**XGB_PARAMS)
    model.set_params(objective=objective)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, group: np.ndarray):
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

    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nTrain joint distribution:\n", pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    results: List[Dict[str, float]] = []

    for gamma in np.arange(0, 1, 1/50):
        print(f"\n=== Training gamma={gamma:.2f} ===")

        objective = GammaOnlySquaredFairObjective(gamma=gamma, group_train=g_train)
        model = train_model(X_train, y_train, X_val, y_val, objective)

        # Save booster JSON instead of pickling the sklearn object.
        booster_path = MODELS_DIR / f"xgb_gamma_only_squared_fair_{str(gamma).replace('.', '_')}.json"
        model.get_booster().save_model(str(booster_path))

        meta_path = MODELS_DIR / f"xgb_gamma_only_squared_fair_{str(gamma).replace('.', '_')}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gamma": gamma,
                    "objective": "BCE + gamma * (mean_p_AAE_neg - mean_p_SAE_neg)^2",
                    "xgb_params": XGB_PARAMS,
                },
                f,
                indent=2,
            )

        val_metrics, val_prob, val_pred = evaluate_model(model, X_val, y_val, g_val)
        test_metrics, test_prob, test_pred = evaluate_model(model, X_test, y_test, g_test)

        print(
            f"gamma={gamma:.2f} | val_f1={val_metrics['f1']:.4f} | "
            f"test_f1={test_metrics['f1']:.4f} | test_fpr_gap={test_metrics['fpr_gap']:.4f} | "
            f"test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        pred_df = test_df.copy()
        pred_df["xgb_prob"] = test_prob
        pred_df["xgb_pred"] = test_pred
        pred_df.to_csv(
            RESULTS_DIR / f"xgb_gamma_only_squared_fair_{str(gamma).replace('.', '_')}_predictions.csv",
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
    results_df.to_csv(RESULTS_DIR / "xgb_gamma_only_squared_fair_sweep_summary.csv", index=False)
    print("\nSaved summary to:", RESULTS_DIR / "xgb_gamma_only_squared_fair_sweep_summary.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_gap"], marker="o", label="FPR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Positive Rate")
    plt.title("FPR vs Gamma (Squared fairness loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_gamma_only_squared_fair_fpr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_gap"], marker="o", label="FNR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Negative Rate")
    plt.title("FNR vs Gamma (Squared fairness loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_gamma_only_squared_fair_fnr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_f1"], marker="o", label="Test F1")
    plt.plot(results_df["gamma"], results_df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Performance vs Gamma (Squared fairness loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_gamma_only_squared_fair_performance.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print(" -", RESULTS_DIR / "xgb_gamma_only_squared_fair_fpr.png")
    print(" -", RESULTS_DIR / "xgb_gamma_only_squared_fair_fnr.png")
    print(" -", RESULTS_DIR / "xgb_gamma_only_squared_fair_performance.png")

    best = results_df.sort_values(["test_fpr_gap", "test_f1"], ascending=[True, False]).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()
