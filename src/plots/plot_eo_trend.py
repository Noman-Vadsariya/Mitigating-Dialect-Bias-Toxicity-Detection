"""
Plot per-round trends (FPR, FNR, F1) for the best EO adversarial-debiasing
XGBoost configuration.

Loads the best hyperparameters from
  data/results/adv_xgb_eo_grid/best_by_val_score.json
retrains the model while recording per-round test metrics, then saves
  data/results/plots/eo_fpr_trend.png
  data/results/plots/eo_fnr_trend.png
  data/results/plots/eo_f1_trend.png
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "experiments"))

from train_adv_xgb import load_split, sigmoid, apply_thresholds  # noqa: E402
from train_adv_xgb_eo import (  # noqa: E402
    _adversary_features,
    _project_out,
    adversary_grad_hess,
    fit_eo_adversary,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402


def group_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg = y_true == 0
    pos = y_true == 1
    fpr = float(y_pred[neg].mean()) if neg.any() else float("nan")
    fnr = float(1.0 - y_pred[pos].mean()) if pos.any() else float("nan")
    return fpr, fnr


def train_and_track(
    X_tr, y_tr, g_tr,
    X_te, y_te, g_te,
    params, num_round, lambda_adv, adv_c,
    t_aae, t_sae,
    warmup_rounds=5,
):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)

    booster = None
    adv_clf: LogisticRegression | None = None

    aae_fpr, aae_fnr, sae_fpr, sae_fnr, f1_hist = [], [], [], [], []

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True)
            adv_clf = fit_eo_adversary(
                margin_tr, y_tr, g_tr, adv_c=adv_c, sample_weight=None,
            )

        def objective(preds, dmat, _adv_clf=adv_clf):
            y = dmat.get_label()
            p_tox = sigmoid(preds)
            grad_tox = p_tox - y
            hess_tox = p_tox * (1.0 - p_tox)

            if _adv_clf is None or lambda_adv == 0.0:
                hess = np.clip(hess_tox, 1e-6, None)
                return grad_tox, hess

            grad_adv, hess_adv = adversary_grad_hess(_adv_clf, preds, y, g_tr)
            grad_pred = _project_out(grad_tox, grad_adv)
            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            return grad, np.clip(hess, 1e-6, None)

        booster = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=1,
            obj=objective, xgb_model=booster, verbose_eval=False,
        )

        probs = sigmoid(booster.predict(dtest, output_margin=True))
        pred = apply_thresholds(probs, g_te, t_aae, t_sae)

        aae_mask = g_te == 1
        sae_mask = g_te == 0
        fpr_a, fnr_a = group_fpr_fnr(y_te[aae_mask], pred[aae_mask])
        fpr_s, fnr_s = group_fpr_fnr(y_te[sae_mask], pred[sae_mask])

        aae_fpr.append(fpr_a); aae_fnr.append(fnr_a)
        sae_fpr.append(fpr_s); sae_fnr.append(fnr_s)
        f1_hist.append(float(f1_score(y_te, pred)))

    return {
        "round": list(range(1, num_round + 1)),
        "aae_fpr": aae_fpr, "aae_fnr": aae_fnr,
        "sae_fpr": sae_fpr, "sae_fnr": sae_fnr,
        "f1": f1_hist,
    }


def plot_group_metric(rounds, aae, sae, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, aae, label="AAE", color="#d62728", linewidth=2)
    ax.plot(rounds, sae, label="SAE", color="#1f77b4", linewidth=2)
    gap = np.abs(np.array(aae) - np.array(sae))
    ax.plot(rounds, gap, label="|gap|", color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_f1(rounds, f1, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, f1, color="#9467bd", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Test F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    best_path = os.path.join(REPO_ROOT, "data", "results",
                             "adv_xgb_eo_grid", "best_by_val_score.json")
    with open(best_path) as f:
        best = json.load(f)
    print(f"Best EO config: lambda_adv={best['lambda_adv']}, adv_c={best['adv_c']}, "
          f"t_aae={best['t_aae']:.4f}, t_sae={best['t_sae']:.4f}")

    tr_csv = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
    va_csv = os.path.join(REPO_ROOT, "data", "processed", "val.csv")
    te_csv = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
    tr_emb = os.path.join(REPO_ROOT, "data", "embeddings", "train_emb.npy")
    va_emb = os.path.join(REPO_ROOT, "data", "embeddings", "val_emb.npy")
    te_emb = os.path.join(REPO_ROOT, "data", "embeddings", "test_emb.npy")

    _, X_tr, y_tr, g_tr = load_split(tr_csv, tr_emb, "dialect_strict")
    _, X_va, y_va, g_va = load_split(va_csv, va_emb, "dialect_strict")
    _, X_te, y_te, g_te = load_split(te_csv, te_emb, "dialect_strict")
    print(f"Loaded train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    params = {
        "max_depth": 5, "eta": 0.08,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "tree_method": "hist", "seed": 42, "verbosity": 0,
    }

    print("Retraining best EO model while tracking per-round metrics...")
    hist = train_and_track(
        X_tr, y_tr, g_tr,
        X_te, y_te, g_te,
        params=params, num_round=100,
        lambda_adv=float(best["lambda_adv"]),
        adv_c=float(best["adv_c"]),
        t_aae=float(best["t_aae"]),
        t_sae=float(best["t_sae"]),
        warmup_rounds=5,
    )

    out_dir = os.path.join(REPO_ROOT, "data", "results", "plots")
    os.makedirs(out_dir, exist_ok=True)

    subtitle = (f"EO-adversarial XGBoost "
                f"(λ={best['lambda_adv']}, C={best['adv_c']}, "
                f"t_AAE={best['t_aae']:.3f}, t_SAE={best['t_sae']:.3f})")

    plot_group_metric(hist["round"], hist["aae_fpr"], hist["sae_fpr"],
                      ylabel="FPR", title=f"Test FPR per round\n{subtitle}",
                      out_path=os.path.join(out_dir, "eo_fpr_trend.png"))
    plot_group_metric(hist["round"], hist["aae_fnr"], hist["sae_fnr"],
                      ylabel="FNR", title=f"Test FNR per round\n{subtitle}",
                      out_path=os.path.join(out_dir, "eo_fnr_trend.png"))
    plot_f1(hist["round"], hist["f1"],
            out_path=os.path.join(out_dir, "eo_f1_trend.png"),
            title=f"Test F1 per round\n{subtitle}")

    hist_path = os.path.join(out_dir, "eo_trend_history.json")
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"  saved {hist_path}")


if __name__ == "__main__":
    main()
