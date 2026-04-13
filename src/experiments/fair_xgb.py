"""XGBoost custom objective for dialect fairness mitigation.

Objective:
    L = (1 - gamma) * L_BCE + gamma * L_fair

Where:
    - L_BCE is standard binary cross-entropy over toxicity labels.
    - L_fair is a differentiable surrogate that penalizes higher average predicted
      toxicity on non-toxic AAE examples compared with non-toxic SAE examples.

Why this surrogate?
    - Your issue is high false positives on AAE.
    - FPR itself is not differentiable because it depends on thresholded outputs.
    - This proxy pushes down the model's toxic score on negative AAE examples,
      which is a good training-time signal for reducing FPR_AAE.

Inputs:
    - Embeddings saved as .npy files for train/val/test.
    - CSVs with columns:
        label (0/1)
        dialect_strict (AAE/SAE)

Outputs:
    - A metrics table across gamma values.
    - Prediction CSV for each gamma.
    - Line plots for FPR/FNR/F1 tradeoffs.

Notes:
    - This uses XGBoost's sklearn API with a custom objective callable.
    - If your installed xgboost version does not accept a callable objective in
      XGBClassifier, you can adapt the same formulas to xgb.train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# =========================
# Config
# =========================
EMB_DIR = Path("../../data/embeddings")
DATA_DIR = Path("../../data/processed/twitterAAE")
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/fair_xgb/CE_FPR_squared_loss")
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

XGB_PARAMS = dict(
    max_depth=50,
    n_estimators=150,
    learning_rate=0.05,
    objective="binary:logistic",  # overridden by callable objective below
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# =========================
# Data helpers
# =========================


def load_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def get_group(df: pd.DataFrame) -> np.ndarray:
    """Map dialect_strict to group ids: AAE=1, SAE=0."""
    group = (
        df["dialect_strict"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"AAE": 1, "SAE": 0})
    )
    if group.isna().any():
        bad = df.loc[group.isna(), "dialect_strict"].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values: {bad}")
    return group.values.astype(int)


# =========================
# Metrics
# =========================


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, g in [("AAE", 1), ("SAE", 0)]:
        idx = group == g
        out[f"fpr_{name}"] = fpr(y_true[idx], y_pred[idx])
        out[f"fnr_{name}"] = fnr(y_true[idx], y_pred[idx])
    out["fpr_gap"] = abs(out["fpr_AAE"] - out["fpr_SAE"])
    out["fnr_gap"] = abs(out["fnr_AAE"] - out["fnr_SAE"])
    return out


# =========================
# Fair objective factory
# =========================


def make_fair_objective(
    train_y: np.ndarray, train_group: np.ndarray, gamma: float
) -> Callable:
    """Create a custom objective for XGBClassifier.

    Objective:
        L = (1-gamma) * BCE + gamma * fair_penalty

    Fair penalty:
        (mean(p | y=0, AAE) - mean(p | y=0, SAE))^2

    This is a smooth proxy for reducing false positive rate disparity.
    """

    train_y = np.asarray(train_y).astype(int)
    train_group = np.asarray(train_group).astype(int)

    def objective(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # y_pred are raw margins (logits)
        p = sigmoid(y_pred)

        # Standard binary cross-entropy gradient / hessian.
        grad_task = p - y_true
        hess_task = p * (1.0 - p)

        # Fairness surrogate uses only negative examples because the observed issue
        # is AAE false positives.
        neg = y_true == 0
        neg_aae = neg & (train_group == 1)
        neg_sae = neg & (train_group == 0)

        grad_fair = np.zeros_like(p)
        hess_fair = np.zeros_like(p)

        n_aae = int(np.sum(neg_aae))
        n_sae = int(np.sum(neg_sae))

        if n_aae > 0 and n_sae > 0:
            mean_aae = float(np.mean(p[neg_aae]))
            mean_sae = float(np.mean(p[neg_sae]))
            diff = mean_aae - mean_sae

            # dp/dz for sigmoid
            dp_dz = p * (1.0 - p)

            # Gradient of the group-mean penalty.
            grad_fair[neg_aae] = 2.0 * diff * dp_dz[neg_aae] / n_aae
            grad_fair[neg_sae] = -2.0 * diff * dp_dz[neg_sae] / n_sae

            # Positive semi-definite Hessian approximation (Gauss-Newton style).
            hess_fair[neg_aae] = 2.0 * (dp_dz[neg_aae] / n_aae) ** 2
            hess_fair[neg_sae] = 2.0 * (dp_dz[neg_sae] / n_sae) ** 2

        grad = (1.0 - gamma) * grad_task + gamma * grad_fair
        hess = (1.0 - gamma) * hess_task + gamma * hess_fair

        # Keep hessians numerically safe.
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return objective


# =========================
# Training / evaluation
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
    model: XGBClassifier, X: np.ndarray, y: np.ndarray, group: np.ndarray
) -> Dict[str, float]:
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
    print(
        "\nTrain joint distribution:\n",
        pd.crosstab(train_df["dialect_strict"], train_df["label"]),
    )

    results: List[Dict[str, float]] = []

    for gamma in np.arange(0, 1, 1/50):
        print(f"\n=== Training gamma={gamma:.2f} ===")

        objective = make_fair_objective(y_train, g_train, gamma)
        model = train_model(X_train, y_train, X_val, y_val, objective)

        # Save the model for this gamma.
        # model_path = (
        #     MODELS_DIR / f"xgb_fair_gamma_{str(gamma).replace('.', '_')}.joblib"
        # )
        # joblib.dump(model, model_path)

        # Evaluate on validation and test.
        val_metrics, val_prob, val_pred = evaluate_model(model, X_val, y_val, g_val)
        test_metrics, test_prob, test_pred = evaluate_model(
            model, X_test, y_test, g_test
        )

        print(
            f"gamma={gamma:.2f} | "
            f"val_f1={val_metrics['f1']:.4f} | test_f1={test_metrics['f1']:.4f} | "
            f"test_fpr_gap={test_metrics['fpr_gap']:.4f} | test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        # Save predictions for the test split.
        out_df = test_df.copy()
        out_df["xgb_prob"] = test_prob
        out_df["xgb_pred"] = test_pred
        out_df.to_csv(
            RESULTS_DIR
            / f"xgb_fair_gamma_{str(gamma).replace('.', '_')}_predictions.csv",
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
    results_df.to_csv(RESULTS_DIR / "xgb_fair_gamma_sweep_summary.csv", index=False)
    print("\nSaved summary to:", RESULTS_DIR / "xgb_fair_gamma_sweep_summary.csv")

    # Plots: one for FPR, one for FNR, plus F1 for context.
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["gamma"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE"
    )
    plt.plot(
        results_df["gamma"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE"
    )
    plt.plot(
        results_df["gamma"], results_df["test_fpr_gap"], marker="o", label="FPR gap"
    )
    plt.xlabel("Gamma")
    plt.ylabel("False Positive Rate")
    plt.title("FPR vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_fair_gamma_fpr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["gamma"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE"
    )
    plt.plot(
        results_df["gamma"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE"
    )
    plt.plot(
        results_df["gamma"], results_df["test_fnr_gap"], marker="o", label="FNR gap"
    )
    plt.xlabel("Gamma")
    plt.ylabel("False Negative Rate")
    plt.title("FNR vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_fair_gamma_fnr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_f1"], marker="o", label="Test F1")
    plt.plot(
        results_df["gamma"],
        results_df["test_accuracy"],
        marker="o",
        label="Test Accuracy",
    )
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Performance vs Gamma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgb_fair_gamma_performance.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print(" -", RESULTS_DIR / "xgb_fair_gamma_fpr.png")
    print(" -", RESULTS_DIR / "xgb_fair_gamma_fnr.png")
    print(" -", RESULTS_DIR / "xgb_fair_gamma_performance.png")

    # Helpful summary row.
    best = results_df.sort_values(
        ["test_fpr_gap", "test_f1"], ascending=[True, False]
    ).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()
