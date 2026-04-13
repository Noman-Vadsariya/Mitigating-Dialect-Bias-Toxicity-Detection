"""Fair Adversarial Gradient Tree Boosting (FAGTB-NN) for dialect-bias mitigation.

This script follows the training recipe described in:
    Grari et al., "Fair Adversarial Gradient Tree Boosting"

Core idea from the paper:
    - Predictor F: gradient tree boosting (GTB)
    - Adversary A: neural network that predicts the sensitive attribute from
      the predictor output
    - During boosting, add the adversary gradient to the predictor gradient
    - Train the adversary multiple steps per predictor step so it does not get
      dominated by the GTB
    - Start from a biased predictor (warm start), then initialize/train the
      adversary on those biased predictions

For our task:
    - Main label: toxicity label (0/1)
    - Sensitive attribute: dialect_strict (AAE vs SAE)
    - Fairness target: reduce over-flagging of AAE as toxic

Implementation notes:
    - Predictor is trained with xgboost.train in an explicit loop.
    - Each outer iteration adds exactly one new tree.
    - Adversary is a PyTorch MLP trained multiple epochs per boosting step.
    - Predictor gradient uses exact task BCE gradient.
    - Adversary gradient is computed exactly by autograd w.r.t. predictor raw
      margins (through sigmoid(predictor_output) as in the paper).
    - To stay close to the paper, the adversary input is sigmoid(F(x)) for
      demographic parity, and [sigmoid(F(x)), y] for equalized odds.
      Since the present task is false-positive disparity, equalized odds is
      the default because it conditions on the label.

Expected inputs:
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
    - summary CSV
    - plots for FPR / FNR / F1 vs gamma

References from the paper used in this implementation:
    - The objective is a min-max game between predictor and adversary.
    - The predictor uses pseudoresiduals (negative gradient of the task loss).
    - The adversary gradient is added to the predictor gradient.
    - For FAGTB-NN, multiple adversary training iterations per boosting step
      are recommended so the predictor does not dominate the adversary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb


# =========================
# Configuration
# =========================
EMB_DIR = Path("../../data/embeddings")
DATA_DIR = Path("../../data/processed/twitterAAE")
RESULTS_DIR = Path("../../data/results/twitterAAE_experiments/fagbt/")
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Predictor training.
TASK_WARMUP_ROUNDS = 20   # initial biased GTB before fairness starts
FAIR_ROUNDS = 100          # one tree per outer fairness iteration
ETA = 0.05                # predictor learning rate (XGBoost eta)
MAX_DEPTH = 50
MIN_CHILD_WEIGHT = 1.0
SUBSAMPLE = 1.0
COLSAMPLE_BYTREE = 1.0
LAMBDA_L2 = 1.0

# Adversary training.
ADV_PRETRAIN_EPOCHS = 10   # initialize from biased predictor outputs
ADV_EPOCHS_PER_ROUND = 5   # several adversary updates per predictor update
ADV_BATCH_SIZE = 64
ADV_LR = 1e-3
ADV_HIDDEN = [16, 8]

# Adversary mode:
#   "eo" = equalized odds  -> input [sigmoid(F(x)), y]
#   "dp" = demographic parity -> input [sigmoid(F(x))]
ADV_MODE: Literal["eo", "dp"] = "eo"

# Whether to standardize the predictor probability before sending it to adversary.
# The paper uses sigmoid(F(x)). We keep that as default.
STANDARDIZE_ADVERSARY_INPUT = False

# XGBoost parameters. objective is supplied by our custom gradients.
XGB_PARAMS = {
    "max_depth": MAX_DEPTH,
    "eta": ETA,
    "min_child_weight": MIN_CHILD_WEIGHT,
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,
    "lambda": LAMBDA_L2,
    "objective": "binary:logistic",
    "seed": SEED,
    "verbosity": 0,
}


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = np.load(TRAIN_EMB)
    X_val = np.load(VAL_EMB)
    X_test = np.load(TEST_EMB)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    return X_train, X_val, X_test, train_df, val_df, test_df


def encode_dialect(df: pd.DataFrame) -> np.ndarray:
    """AAE -> 1, SAE -> 0.

    We keep AAE as the positive sensitive class for reporting, but the adversary
    can learn either encoding. Here AAE=1 and SAE=0 is convenient for metrics.
    """
    g = df["dialect_strict"].astype(str).str.strip().str.upper().map({"AAE": 1, "SAE": 0})
    if g.isna().any():
        bad = df.loc[g.isna(), "dialect_strict"].unique().tolist()
        raise ValueError(f"Unexpected dialect_strict values: {bad}")
    return g.values.astype(int)


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


def to_torch(x: np.ndarray, device: str = DEVICE, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


# =========================
# Adversary network
# =========================

class AdversaryNet(nn.Module):
    """Small MLP adversary, matching the paper's recommendation to use a NN.

    For demographic parity:
        input dim = 1  (sigmoid(F(x)))
    For equalized odds:
        input dim = 2  ([sigmoid(F(x)), y])
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [16, 8]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary sensitive attribute logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =========================
# Predictor objective helpers
# =========================

def task_bce_grad_hess(raw_margin: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact gradient and hessian for binary cross-entropy wrt raw margin z.

    If p = sigmoid(z), BCE(y, p) has:
        grad = p - y
        hess = p * (1 - p)
    """
    p = stable_sigmoid_np(raw_margin)
    grad = p - y_true
    hess = p * (1.0 - p)
    return p, grad, hess


@dataclass
class FAGTBState:
    """Holds the current adversary and the aligned training arrays."""

    adversary: AdversaryNet
    gamma: float
    y_train: np.ndarray
    s_train: np.ndarray
    adv_mode: Literal["eo", "dp"] = "eo"
    standardize_adv_input: bool = False
    input_mean: Optional[float] = None
    input_std: Optional[float] = None


def make_adversary_features(
    predictor_prob: np.ndarray,
    y_true: np.ndarray,
    mode: Literal["eo", "dp"] = "eo",
    standardize: bool = False,
    fit_stats: bool = False,
    input_mean: Optional[float] = None,
    input_std: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Create adversary input features.

    - DP: [p]
    - EO: [p, y]

    The paper uses sigmoid(F(x)) as predictor output input; for equalized odds,
    the label is concatenated to the predictor output.
    """
    p = predictor_prob.reshape(-1, 1).astype(np.float32)
    if mode == "dp":
        feats = p
    elif mode == "eo":
        feats = np.concatenate([p, y_true.reshape(-1, 1).astype(np.float32)], axis=1)
    else:
        raise ValueError(f"Unknown adversary mode: {mode}")

    if standardize:
        if fit_stats:
            input_mean = float(feats.mean())
            input_std = float(feats.std() + 1e-8)
        assert input_mean is not None and input_std is not None
        feats = (feats - input_mean) / input_std
    return feats.astype(np.float32), input_mean, input_std


# =========================
# Adversary training
# =========================

def train_adversary(
    adversary: AdversaryNet,
    predictor_prob: np.ndarray,
    y_true: np.ndarray,
    s_true: np.ndarray,
    mode: Literal["eo", "dp"],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = DEVICE,
    standardize: bool = False,
    input_mean: Optional[float] = None,
    input_std: Optional[float] = None,
) -> Tuple[AdversaryNet, Optional[float], Optional[float]]:
    """Train adversary for multiple epochs on frozen predictor outputs.

    This is the key practical insight from the paper:
    several adversary training iterations per predictor step, so the GTB does
    not dominate the adversary.
    """
    feats, input_mean, input_std = make_adversary_features(
        predictor_prob=predictor_prob,
        y_true=y_true,
        mode=mode,
        standardize=standardize,
        fit_stats=input_mean is None and input_std is None,
        input_mean=input_mean,
        input_std=input_std,
    )

    ds = torch.utils.data.TensorDataset(
        to_torch(feats, device=device),
        to_torch(s_true, device=device),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    adversary = adversary.to(device)
    adversary.train()
    opt = torch.optim.Adam(adversary.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for xb, sb in loader:
            opt.zero_grad(set_to_none=True)
            logits = adversary(xb)
            loss = bce(logits, sb)
            loss.backward()
            opt.step()

    return adversary, input_mean, input_std


# =========================
# Adversarial gradient wrt predictor raw margin
# =========================

def adversary_grad_wrt_margin(
    raw_margin: np.ndarray,
    y_true: np.ndarray,
    s_true: np.ndarray,
    adversary: AdversaryNet,
    mode: Literal["eo", "dp"] = "eo",
    standardize: bool = False,
    input_mean: Optional[float] = None,
    input_std: Optional[float] = None,
    device: str = DEVICE,
) -> np.ndarray:
    """Exact gradient of adversary loss wrt predictor raw margin.

    We compute the gradient through the chain:
        z -> sigmoid(z) -> adversary input -> adversary output -> BCE(s, adv)

    The predictor should maximize adversary loss, so the final predictor update
    uses -gamma * grad_adv.
    """
    adversary.eval()
    z = torch.tensor(raw_margin, dtype=torch.float32, device=device, requires_grad=True)
    p = torch.sigmoid(z).unsqueeze(1)  # paper: sigmoid(F(x)) as adversary input

    if mode == "dp":
        adv_in = p
    else:
        y_col = torch.tensor(y_true, dtype=torch.float32, device=device).unsqueeze(1)
        adv_in = torch.cat([p, y_col], dim=1)

    if standardize:
        assert input_mean is not None and input_std is not None
        adv_in = (adv_in - input_mean) / input_std

    s_t = torch.tensor(s_true, dtype=torch.float32, device=device)
    logits = adversary(adv_in)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, s_t)

    grad_z = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
    return grad_z.detach().cpu().numpy().astype(np.float32)


# =========================
# One boosting step objective
# =========================

def make_fagtb_objective(
    state: FAGTBState,
) -> callable:
    """Create an XGBoost custom objective for one boosting step.

    The objective uses:
        g = g_task - gamma * g_adv
        h = h_task

    Why not exact hessian for the adversary term?
        The paper injects the adversary gradient into the boosting residuals.
        In gradient boosting/tree growth, the first-order correction is the key
        fairness signal. Using the task hessian preserves stable tree growth.
    """

    def objective(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label().astype(np.float32)
        p, g_task, h_task = task_bce_grad_hess(preds, y)

        # Exact adversarial gradient wrt predictor raw margin z.
        g_adv = adversary_grad_wrt_margin(
            raw_margin=preds,
            y_true=state.y_train,
            s_true=state.s_train,
            adversary=state.adversary,
            mode=state.adv_mode,
            standardize=state.standardize_adv_input,
            input_mean=state.input_mean,
            input_std=state.input_std,
            device=DEVICE,
        )

        # Predictor wants to maximize adversary loss, so subtract its gradient.
        grad = g_task - state.gamma * g_adv

        # Keep the second-order term from the task loss for stable boosting.
        hess = np.clip(h_task, 1e-6, None)
        return grad, hess

    return objective


# =========================
# Evaluation
# =========================

def predict_prob(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat, output_margin=True)
    return stable_sigmoid_np(raw)


def evaluate_model(
    booster: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    prob = predict_prob(booster, X)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        **group_metrics(y, pred, group),
    }
    return metrics, prob, pred


# =========================
# Training loop
# =========================

def train_fagtb_one_gamma(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    gamma: float,
) -> Tuple[xgb.Booster, Dict[str, float], Dict[str, float]]:
    """Train a single FAGTB-NN model for one gamma value."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 1) Warm-start: train a biased GTB first (task only).
    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=TASK_WARMUP_ROUNDS,
        evals=[(dval, "val")],
        verbose_eval=False,
    )

    # 2) Initialize the adversary on the biased GTB predictions.
    train_prob = predict_prob(booster, X_train)
    adversary = AdversaryNet(input_dim=(2 if ADV_MODE == "eo" else 1), hidden_dims=ADV_HIDDEN)
    adversary, input_mean, input_std = train_adversary(
        adversary=adversary,
        predictor_prob=train_prob,
        y_true=y_train,
        s_true=s_train,
        mode=ADV_MODE,
        epochs=ADV_PRETRAIN_EPOCHS,
        batch_size=ADV_BATCH_SIZE,
        lr=ADV_LR,
        device=DEVICE,
        standardize=STANDARDIZE_ADVERSARY_INPUT,
    )

    state = FAGTBState(
        adversary=adversary,
        gamma=gamma,
        y_train=y_train,
        s_train=s_train,
        adv_mode=ADV_MODE,
        standardize_adv_input=STANDARDIZE_ADVERSARY_INPUT,
        input_mean=input_mean,
        input_std=input_std,
    )

    # 3) Iterative fairness boosting: for each predictor step, train the adversary
    #    several times, then add one tree using the modified residuals.
    for round_idx in range(FAIR_ROUNDS):
        # Re-train the adversary multiple steps on the current predictor outputs.
        current_train_prob = predict_prob(booster, X_train)
        state.adversary, state.input_mean, state.input_std = train_adversary(
            adversary=state.adversary,
            predictor_prob=current_train_prob,
            y_true=y_train,
            s_true=s_train,
            mode=ADV_MODE,
            epochs=ADV_EPOCHS_PER_ROUND,
            batch_size=ADV_BATCH_SIZE,
            lr=ADV_LR,
            device=DEVICE,
            standardize=STANDARDIZE_ADVERSARY_INPUT,
            input_mean=state.input_mean,
            input_std=state.input_std,
        )

        # Custom objective for exactly one tree.
        obj = make_fagtb_objective(state)

        booster = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=1,
            evals=[(dval, "val")],
            obj=obj,
            xgb_model=booster,
            verbose_eval=False,
        )

    # Final metrics.
    val_metrics, _, _ = evaluate_model(booster, X_val, y_val, s_val)
    train_metrics, _, _ = evaluate_model(booster, X_train, y_train, s_train)
    return booster, train_metrics, val_metrics


# =========================
# Main
# =========================

def main() -> None:
    set_seed(SEED)

    X_train, X_val, X_test, train_df, val_df, test_df = load_data()
    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    s_train = encode_dialect(train_df)
    s_val = encode_dialect(val_df)
    s_test = encode_dialect(test_df)

    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Train dialect distribution:\n", train_df["dialect_strict"].value_counts())
    print("\nTrain joint distribution:\n", pd.crosstab(train_df["dialect_strict"], train_df["label"]))

    all_rows: List[Dict[str, float]] = []

    for gamma in np.arange(0, 1, 1/50):
        print(f"\n=== Training FAGTB-NN with gamma={gamma:.3f} ===")
        booster, train_metrics, val_metrics = train_fagtb_one_gamma(
            X_train=X_train,
            y_train=y_train,
            s_train=s_train,
            X_val=X_val,
            y_val=y_val,
            s_val=s_val,
            gamma=gamma,
        )

        test_metrics, test_prob, test_pred = evaluate_model(booster, X_test, y_test, s_test)

        # Save model and predictions.
        model_path = MODELS_DIR / f"fagtb_nn_gamma_{str(gamma).replace('.', '_')}.json"
        booster.save_model(str(model_path))

        pred_df = test_df.copy()
        pred_df["fagtb_prob"] = test_prob
        pred_df["fagtb_pred"] = test_pred
        pred_df.to_csv(RESULTS_DIR / f"fagtb_nn_gamma_{str(gamma).replace('.', '_')}_predictions.csv", index=False)

        print(
            f"gamma={gamma:.3f} | "
            f"val_f1={val_metrics['f1']:.4f} | test_f1={test_metrics['f1']:.4f} | "
            f"test_fpr_gap={test_metrics['fpr_gap']:.4f} | test_fnr_gap={test_metrics['fnr_gap']:.4f}"
        )

        all_rows.append(
            {
                "gamma": gamma,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "train_fpr_AAE": train_metrics["fpr_AAE"],
                "train_fpr_SAE": train_metrics["fpr_SAE"],
                "train_fpr_gap": train_metrics["fpr_gap"],
                "train_fnr_AAE": train_metrics["fnr_AAE"],
                "train_fnr_SAE": train_metrics["fnr_SAE"],
                "train_fnr_gap": train_metrics["fnr_gap"],
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

    results_df = pd.DataFrame(all_rows).sort_values("gamma").reset_index(drop=True)
    summary_path = RESULTS_DIR / "fagtb_nn_gamma_sweep_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Plots.
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fpr_AAE"], marker="o", label="FPR AAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_SAE"], marker="o", label="FPR SAE")
    plt.plot(results_df["gamma"], results_df["test_fpr_gap"], marker="o", label="FPR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Positive Rate")
    plt.title(f"FPR vs Gamma ({ADV_MODE.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fagtb_nn_fpr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_fnr_AAE"], marker="o", label="FNR AAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_SAE"], marker="o", label="FNR SAE")
    plt.plot(results_df["gamma"], results_df["test_fnr_gap"], marker="o", label="FNR gap")
    plt.xlabel("Gamma")
    plt.ylabel("False Negative Rate")
    plt.title(f"FNR vs Gamma ({ADV_MODE.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fagtb_nn_fnr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["gamma"], results_df["test_f1"], marker="o", label="Test F1")
    plt.plot(results_df["gamma"], results_df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title(f"Performance vs Gamma ({ADV_MODE.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fagtb_nn_performance.png", dpi=300)
    plt.close()

    print("Saved plots:")
    print(" -", RESULTS_DIR / "fagtb_nn_fpr.png")
    print(" -", RESULTS_DIR / "fagtb_nn_fnr.png")
    print(" -", RESULTS_DIR / "fagtb_nn_performance.png")

    best = results_df.sort_values(["test_fpr_gap", "test_f1"], ascending=[True, False]).iloc[0]
    print("\nBest test result by low FPR gap (tie-breaker F1):")
    print(best.to_dict())


if __name__ == "__main__":
    main()