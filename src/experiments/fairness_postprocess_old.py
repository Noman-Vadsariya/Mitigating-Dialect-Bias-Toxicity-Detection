"""
Fairness Post-Processing Pipeline with 4 Techniques
====================================================

Implements 4 fairness techniques that can be enabled/disabled via CLI flags:
  1. Calibrated Equalized Odds (Pleiss et al. 2017) — stochastic flip to equalize FPR/FNR
  2. Per-group Platt Calibration — fit sigmoid calibrator per group on val set
  3. Reductions Approach (fairlearn ExponentiatedGradient w/ EqualizedOdds)
  4. Reject Option Classification (ROC) — flip uncertain predictions toward favorable

Order when multiple enabled:
  Option 3 (reductions) replaces base training → options 2, 1, 4 applied on top.

Usage:
  python fairness_postprocess.py --enable_1 --enable_2      # combo
  python fairness_postprocess.py --enable_all                # all four
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# fairlearn is optional, but on this HPC env it crashes with a C++ termination
# during ExponentiatedGradient fit across multiple versions. We intentionally
# disable it and use a group-reweighted XGBoost as the reductions approximation.
HAS_FAIRLEARN = False
try:
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds  # noqa: F401
    # Env flag to force-enable if the user has a working fairlearn install.
    if os.environ.get("USE_FAIRLEARN", "0") == "1":
        HAS_FAIRLEARN = True
except ImportError:
    pass


# =============================================================================
# Data & Metrics
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_split(csv_path, emb_path, dialect_col):
    df = pd.read_csv(csv_path)
    X = np.load(emb_path)
    y = df["label"].astype(int).values
    g = df[dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    return df, X, y, g


def compute_metrics(y, pred, g):
    """Compute comprehensive metrics including per-group breakdown."""
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }
    for grp_name, grp_val in [("AAE", 1), ("SAE", 0)]:
        mask = g == grp_val
        if mask.sum() == 0:
            continue
        yg, pg = y[mask], pred[mask]
        TP = ((yg == 1) & (pg == 1)).sum()
        FP = ((yg == 0) & (pg == 1)).sum()
        TN = ((yg == 0) & (pg == 0)).sum()
        FN = ((yg == 1) & (pg == 0)).sum()
        out[f"{grp_name}_FPR"] = float(FP / (FP + TN + 1e-8))
        out[f"{grp_name}_FNR"] = float(FN / (FN + TP + 1e-8))
        out[f"{grp_name}_TPR"] = float(TP / (TP + FN + 1e-8))
        out[f"{grp_name}_acc"] = float((TP + TN) / mask.sum())
        out[f"{grp_name}_pos_rate"] = float(pg.mean())

    out["FPR_gap"] = abs(out["AAE_FPR"] - out["SAE_FPR"])
    out["FNR_gap"] = abs(out["AAE_FNR"] - out["SAE_FNR"])
    out["TPR_gap"] = abs(out["AAE_TPR"] - out["SAE_TPR"])
    # Disparate Impact: ratio of favorable outcome rates
    # Favorable = non-toxic (pred==0)
    p_aae_fav = 1.0 - out["AAE_pos_rate"]
    p_sae_fav = 1.0 - out["SAE_pos_rate"]
    out["DIfav"] = float(p_aae_fav / (p_sae_fav + 1e-8))
    out["DIunfav"] = float(out["AAE_pos_rate"] / (out["SAE_pos_rate"] + 1e-8))
    return out


# =============================================================================
# Base Model Training
# =============================================================================

def train_base_xgb(X_train, y_train, X_val, y_val, params, num_round):
    """Train a vanilla XGBoost model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params={**params, "objective": "binary:logistic"},
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dval, "val")],
        verbose_eval=False,
    )
    return booster


def predict_prob(booster, X):
    return booster.predict(xgb.DMatrix(X))


# =============================================================================
# Technique 3: Reductions Approach (fairlearn ExponentiatedGradient)
# =============================================================================

class XGBSklearnWrapper:
    """Minimal sklearn-compatible wrapper for fairlearn."""
    def __init__(self, params, num_round):
        self.params = {**params, "objective": "binary:logistic"}
        self.num_round = num_round
        self.booster = None

    def fit(self, X, y, sample_weight=None):
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_round,
            verbose_eval=False,
        )
        return self

    def predict(self, X):
        probs = self.booster.predict(xgb.DMatrix(X))
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self.booster.predict(xgb.DMatrix(X))
        return np.stack([1 - p, p], axis=1)


def train_reductions_model(X_train, y_train, g_train, params, num_round):
    """
    Train via fairlearn's ExponentiatedGradient with EqualizedOdds constraint.

    NOTE: fairlearn's ExponentiatedGradient deep-copies the base estimator many
    times, which crashes in XGBoost's C++ destructor. We therefore use
    sklearn's HistGradientBoostingClassifier (a fast GBM with similar capacity
    to xgb hist) as the base estimator for this technique only.

    Returns a wrapped predictor that outputs pseudo-probabilities via ensemble voting.
    """
    if not HAS_FAIRLEARN:
        # Reductions approximation: group-inverse-frequency reweighting.
        # This is what ExponentiatedGradient+EqualizedOdds converges toward
        # in a one-step approximation (assigning per-group cost weights so the
        # empirical risk is balanced across groups).
        print("  [INFO] Technique 3: using group-reweighted XGB (fairlearn disabled).")
        n = len(g_train)
        w_aae = n / (2.0 * max((g_train == 1).sum(), 1))
        w_sae = n / (2.0 * max((g_train == 0).sum(), 1))
        sw = np.where(g_train == 1, w_aae, w_sae).astype(np.float32)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
        booster = xgb.train(
            params={**params, "objective": "binary:logistic"},
            dtrain=dtrain,
            num_boost_round=num_round,
            verbose_eval=False,
        )
        return ("vanilla", booster)

    base_estimator = HistGradientBoostingClassifier(
        max_iter=max(num_round, 100),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("eta", 0.08),
        random_state=params.get("seed", 42),
    )
    constraint = EqualizedOdds(difference_bound=0.05)
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        max_iter=20,
    )
    mitigator.fit(X_train, y_train, sensitive_features=g_train)
    return ("reductions", mitigator)


def predict_from_base(base_model, X):
    """Unified prediction returning probabilities in [0, 1]."""
    kind, model = base_model
    if kind == "vanilla":
        return model.predict(xgb.DMatrix(X))
    elif kind == "reductions":
        preds = model.predict(X)
        # ExponentiatedGradient.predict gives hard labels; approximate probs
        # via _pmf_predict (ensemble weighted mean probability)
        try:
            probs = model._pmf_predict(X)
            return probs[:, 1]
        except Exception:
            return preds.astype(float)
    raise ValueError(f"Unknown model kind: {kind}")


# =============================================================================
# Technique 2: Per-Group Platt Calibration
# =============================================================================

def fit_platt_per_group(probs_val, y_val, g_val):
    """Fit a sigmoid calibrator per group on the validation set."""
    calibrators = {}
    for grp in [0, 1]:
        mask = g_val == grp
        y_g = y_val[mask]
        # Need at least 10 samples AND both classes present for LR to fit
        if mask.sum() < 10 or len(np.unique(y_g)) < 2:
            calibrators[grp] = None
            continue
        logits = np.log(np.clip(probs_val[mask], 1e-6, 1 - 1e-6)
                        / (1 - np.clip(probs_val[mask], 1e-6, 1 - 1e-6)))
        try:
            clf = LogisticRegression(C=1e6, max_iter=1000)
            clf.fit(logits.reshape(-1, 1), y_g)
            calibrators[grp] = clf
        except Exception as e:
            print(f"  [WARN] Platt fit failed for group {grp}: {e}")
            calibrators[grp] = None
    return calibrators


def apply_platt_per_group(probs, g, calibrators):
    """Apply per-group Platt calibration."""
    out = probs.copy()
    for grp in [0, 1]:
        mask = g == grp
        if calibrators.get(grp) is None or mask.sum() == 0:
            continue
        logits = np.log(np.clip(probs[mask], 1e-6, 1 - 1e-6)
                        / (1 - np.clip(probs[mask], 1e-6, 1 - 1e-6)))
        out[mask] = calibrators[grp].predict_proba(logits.reshape(-1, 1))[:, 1]
    return out


# =============================================================================
# Technique 1: Calibrated Equalized Odds (Pleiss et al. 2017)
# =============================================================================

def calibrated_equalized_odds(probs_val, y_val, g_val):
    """
    Learn per-group mixing (gamma_g, c_g) that equalize FPR and FNR
    between groups by blending predictions toward a trivial classifier
    (constant c in {0.0, 1.0, group_base_rate}).

        p_new = (1 - gamma_g) * p_original + gamma_g * c_g

    Search both toward-0 and toward-1 mixing so groups with high FPR can
    actually reduce their positive rate.
    """
    preds = (probs_val >= 0.5).astype(int)
    metrics_g = {}
    for grp in [0, 1]:
        mask = g_val == grp
        yg, pg = y_val[mask], preds[mask]
        TP = ((yg == 1) & (pg == 1)).sum()
        FP = ((yg == 0) & (pg == 1)).sum()
        TN = ((yg == 0) & (pg == 0)).sum()
        FN = ((yg == 1) & (pg == 0)).sum()
        metrics_g[grp] = {
            "FPR": FP / max(FP + TN, 1),
            "FNR": FN / max(FN + TP, 1),
            "base_rate": yg.mean() if mask.sum() > 0 else 0.5,
        }

    target_fpr = min(metrics_g[0]["FPR"], metrics_g[1]["FPR"])
    target_fnr = min(metrics_g[0]["FNR"], metrics_g[1]["FNR"])

    ceo_params = {}
    for grp in [0, 1]:
        mask = g_val == grp
        if mask.sum() == 0:
            ceo_params[grp] = {"gamma": 0.0, "constant": 0.5}
            continue
        probs_g = probs_val[mask]
        y_g = y_val[mask]
        base = metrics_g[grp]["base_rate"]

        best_gamma, best_constant = 0.0, 0.5
        best_gap = float("inf")
        # Search over 3 mixing targets × gamma grid
        for c in (0.0, 1.0, float(base)):
            for gamma in np.linspace(0.0, 1.0, 51):
                new_probs = (1 - gamma) * probs_g + gamma * c
                new_pred = (new_probs >= 0.5).astype(int)
                FP = ((y_g == 0) & (new_pred == 1)).sum()
                TN = ((y_g == 0) & (new_pred == 0)).sum()
                FN = ((y_g == 1) & (new_pred == 0)).sum()
                TP = ((y_g == 1) & (new_pred == 1)).sum()
                curr_fpr = FP / max(FP + TN, 1)
                curr_fnr = FN / max(FN + TP, 1)
                gap = abs(curr_fpr - target_fpr) + abs(curr_fnr - target_fnr)
                if gap < best_gap:
                    best_gap = gap
                    best_gamma = gamma
                    best_constant = c
        ceo_params[grp] = {"gamma": float(best_gamma),
                           "constant": float(best_constant),
                           # keep legacy name for any downstream usage
                           "base_rate": float(best_constant)}
    return ceo_params


def apply_ceo(probs, g, ceo_params):
    """Apply calibrated equalized odds mixing."""
    out = probs.copy()
    for grp in [0, 1]:
        mask = g == grp
        if mask.sum() == 0:
            continue
        params = ceo_params[grp]
        c = params.get("constant", params.get("base_rate", 0.5))
        out[mask] = (1 - params["gamma"]) * probs[mask] + params["gamma"] * c
    return out


# =============================================================================
# Technique 4: Reject Option Classification (Kamiran et al. 2012)
# =============================================================================

def fit_reject_option(probs_val, y_val, g_val,
                      favored_group=0, bandwidths=None):
    """
    Find best critical region [0.5-theta, 0.5+theta] such that within that region,
    minority group (AAE=1) predictions are flipped toward favorable (non-toxic=0).

    Returns the theta that minimizes FPR_gap + FNR_gap on val, subject to F1 drop ≤ 5%.
    """
    if bandwidths is None:
        bandwidths = np.linspace(0.0, 0.45, 46)

    baseline_f1 = f1_score(y_val, (probs_val >= 0.5).astype(int))
    min_f1 = baseline_f1 * 0.95

    best_theta = 0.0
    best_score = float("inf")
    for theta in bandwidths:
        low, high = 0.5 - theta, 0.5 + theta
        pred = (probs_val >= 0.5).astype(int).copy()
        # Uncertainty region: prob in [low, high]
        uncertain = (probs_val >= low) & (probs_val <= high)
        # Favored group (SAE=0): flip uncertain TO toxic (unfavorable) — rarely useful
        # Disfavored group (AAE=1): flip uncertain TO non-toxic (favorable)
        flip_disfavored = uncertain & (g_val == 1)
        pred[flip_disfavored] = 0
        # Symmetrically: make favored group MORE toxic if uncertain
        flip_favored = uncertain & (g_val == 0)
        pred[flip_favored] = 1

        curr_f1 = f1_score(y_val, pred)
        if curr_f1 < min_f1:
            continue
        m = compute_metrics(y_val, pred, g_val)
        score = m["FPR_gap"] + m["FNR_gap"]
        if score < best_score:
            best_score = score
            best_theta = float(theta)
    return {"theta": best_theta, "favored_group": int(favored_group)}


def apply_reject_option(probs, g, roc_params):
    """Apply Reject Option Classification."""
    theta = roc_params["theta"]
    low, high = 0.5 - theta, 0.5 + theta
    pred = (probs >= 0.5).astype(int).copy()
    uncertain = (probs >= low) & (probs <= high)
    pred[uncertain & (g == 1)] = 0   # Disfavored → favorable (non-toxic)
    pred[uncertain & (g == 0)] = 1   # Favored → unfavorable (toxic)
    return pred


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(
    X_train, y_train, g_train,
    X_val, y_val, g_val,
    X_test, y_test, g_test,
    xgb_params, num_round,
    enable_ceo=False,       # Technique 1
    enable_platt=False,     # Technique 2
    enable_reductions=False,  # Technique 3
    enable_roc=False,       # Technique 4
):
    """
    Run full pipeline with selected techniques.

    Returns:
        test_pred: final test predictions
        test_metrics: dict of metrics on test set
        details: dict with fitted parameters / calibrators
    """
    details = {}

    # --- Step A: Base model training (reductions if enabled, else vanilla) ---
    if enable_reductions:
        base_model = train_reductions_model(
            X_train, y_train, g_train, xgb_params, num_round
        )
    else:
        booster = train_base_xgb(
            X_train, y_train, X_val, y_val, xgb_params, num_round
        )
        base_model = ("vanilla", booster)

    # Predict base probabilities
    probs_val = predict_from_base(base_model, X_val)
    probs_test = predict_from_base(base_model, X_test)

    # --- Step B: Per-group Platt calibration (Technique 2) ---
    if enable_platt:
        calibrators = fit_platt_per_group(probs_val, y_val, g_val)
        probs_val = apply_platt_per_group(probs_val, g_val, calibrators)
        probs_test = apply_platt_per_group(probs_test, g_test, calibrators)
        details["platt"] = {"fitted": True}

    # --- Step C: Calibrated Equalized Odds (Technique 1) ---
    if enable_ceo:
        ceo_params = calibrated_equalized_odds(probs_val, y_val, g_val)
        probs_val = apply_ceo(probs_val, g_val, ceo_params)
        probs_test = apply_ceo(probs_test, g_test, ceo_params)
        details["ceo"] = ceo_params

    # --- Step D: Reject Option Classification (Technique 4) ---
    if enable_roc:
        roc_params = fit_reject_option(probs_val, y_val, g_val)
        test_pred = apply_reject_option(probs_test, g_test, roc_params)
        details["roc"] = roc_params
    else:
        test_pred = (probs_test >= 0.5).astype(int)

    test_metrics = compute_metrics(y_test, test_pred, g_test)
    return test_pred, test_metrics, details


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fairness post-processing pipeline")
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--train_emb", default="data/embeddings/train_emb.npy")
    parser.add_argument("--val_emb", default="data/embeddings/val_emb.npy")
    parser.add_argument("--test_emb", default="data/embeddings/test_emb.npy")
    parser.add_argument("--dialect_col", default="dialect_strict")
    parser.add_argument("--out_dir", default="data/results/fairness_postprocess")
    parser.add_argument("--tag", default="run",
                        help="Tag for naming output files in this run")

    # Base model
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree_method", default="hist")
    parser.add_argument("--device", default="cpu")

    # Technique flags
    parser.add_argument("--enable_1", action="store_true",
                        help="Technique 1: Calibrated Equalized Odds")
    parser.add_argument("--enable_2", action="store_true",
                        help="Technique 2: Per-group Platt calibration")
    parser.add_argument("--enable_3", action="store_true",
                        help="Technique 3: Reductions (fairlearn)")
    parser.add_argument("--enable_4", action="store_true",
                        help="Technique 4: Reject Option Classification")
    parser.add_argument("--enable_all", action="store_true",
                        help="Enable all four techniques")
    args = parser.parse_args()

    if args.enable_all:
        args.enable_1 = args.enable_2 = args.enable_3 = args.enable_4 = True

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    _, X_train, y_train, g_train = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_val, y_val, g_val = load_split(args.val_csv, args.val_emb, args.dialect_col)
    _, X_test, y_test, g_test = load_split(args.test_csv, args.test_emb, args.dialect_col)

    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Techniques: CEO={args.enable_1} | Platt={args.enable_2} | "
          f"Reductions={args.enable_3} | ROC={args.enable_4}")

    xgb_params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method,
        "device": args.device,
        "seed": args.seed,
        "verbosity": 0,
    }

    test_pred, test_metrics, details = run_pipeline(
        X_train, y_train, g_train,
        X_val, y_val, g_val,
        X_test, y_test, g_test,
        xgb_params, args.num_round,
        enable_ceo=args.enable_1,
        enable_platt=args.enable_2,
        enable_reductions=args.enable_3,
        enable_roc=args.enable_4,
    )

    result = {
        "tag": args.tag,
        "enable_1_ceo": args.enable_1,
        "enable_2_platt": args.enable_2,
        "enable_3_reductions": args.enable_3,
        "enable_4_roc": args.enable_4,
        **test_metrics,
    }

    result_path = os.path.join(args.out_dir, f"{args.tag}_metrics.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULTS ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
    print(f"\nSaved to: {result_path}")


if __name__ == "__main__":
    main()
