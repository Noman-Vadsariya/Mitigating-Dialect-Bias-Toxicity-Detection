"""Plot per-round trends (per-class FPR, FNR, macro-F1) for the best
EO adversarial-debiasing ternary XGBoost configuration.

Loads the best hyperparameters and per-group/per-class logit offsets from
  data/results/adv_xgb_eo_grid_ternary/best_by_val_score.json
retrains the ternary EO model while recording per-round per-class
group-conditional metrics, then saves PNG plots and a history JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "experiments"))

from train_adv_xgb_ternary import (  # noqa: E402
    NUM_CLASSES, apply_logit_offsets, load_split, softmax,
)
from train_adv_xgb_eo_ternary import (  # noqa: E402
    _adversary_features, _project_out, _softmax_ce_grad_hess,
    adversary_grad_hess, fit_eo_adversary,
)


def per_class_group_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray, K: int):
    """One-vs-rest FPR/FNR per class for a single group slice."""
    fpr = np.full(K, np.nan, dtype=np.float64)
    fnr = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        pos = (y_true == k)
        neg = ~pos
        if neg.any():
            fpr[k] = float((y_pred[neg] == k).mean())
        if pos.any():
            fnr[k] = float(1.0 - (y_pred[pos] == k).mean())
    return fpr, fnr


def train_and_track(
    X_tr, y_tr, g_tr,
    X_te, y_te, g_te,
    params, num_round,
    lambda_adv, adv_c,
    delta_aae, delta_sae,
    warmup_rounds=5,
):
    K = NUM_CLASSES
    params = {**params, "objective": "multi:softprob", "num_class": K,
              "disable_default_eval_metric": 1}
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)
    N_tr = X_tr.shape[0]
    N_te = X_te.shape[0]

    booster: xgb.Booster | None = None
    adv_clf: LogisticRegression | None = None

    aae_fpr = np.zeros((num_round, K))
    sae_fpr = np.zeros((num_round, K))
    aae_fnr = np.zeros((num_round, K))
    sae_fnr = np.zeros((num_round, K))
    f1_macro_hist = []

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True).reshape(N_tr, K)
            P_tr = softmax(margin_tr, axis=1)
            adv_clf = fit_eo_adversary(P_tr, y_tr, g_tr, adv_c=adv_c)

        def objective(preds, dmat, _adv_clf=adv_clf):
            y = dmat.get_label().astype(int)
            n = y.shape[0]
            margins = np.asarray(preds).reshape(n, K)
            grad_tox, hess_tox = _softmax_ce_grad_hess(margins, y)
            if _adv_clf is None or lambda_adv == 0.0:
                return grad_tox.reshape(-1), np.clip(hess_tox, 1e-6, None).reshape(-1)
            grad_adv, hess_adv = adversary_grad_hess(_adv_clf, margins, y, g_tr)
            grad_pred = _project_out(grad_tox, grad_adv)
            grad = grad_pred - lambda_adv * grad_adv
            hess = np.clip(hess_tox + lambda_adv * hess_adv, 1e-6, None)
            return grad.reshape(-1), hess.reshape(-1)

        booster = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=1,
            obj=objective, xgb_model=booster, verbose_eval=False,
        )

        margin_te = booster.predict(dtest, output_margin=True).reshape(N_te, K)
        pred = apply_logit_offsets(margin_te, g_te, delta_aae, delta_sae)

        aae_mask = (g_te == 1)
        sae_mask = ~aae_mask
        fpr_a, fnr_a = per_class_group_fpr_fnr(y_te[aae_mask], pred[aae_mask], K)
        fpr_s, fnr_s = per_class_group_fpr_fnr(y_te[sae_mask], pred[sae_mask], K)
        aae_fpr[r] = fpr_a; aae_fnr[r] = fnr_a
        sae_fpr[r] = fpr_s; sae_fnr[r] = fnr_s
        f1_macro_hist.append(float(f1_score(y_te, pred, average="macro")))

    return {
        "round": list(range(1, num_round + 1)),
        "aae_fpr": aae_fpr.tolist(), "sae_fpr": sae_fpr.tolist(),
        "aae_fnr": aae_fnr.tolist(), "sae_fnr": sae_fnr.tolist(),
        "f1_macro": f1_macro_hist,
    }


def plot_class_group_metric(rounds, aae_col, sae_col, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, aae_col, label="AAE", color="#d62728", linewidth=2)
    ax.plot(rounds, sae_col, label="SAE", color="#1f77b4", linewidth=2)
    gap = np.abs(np.array(aae_col) - np.array(sae_col))
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


def plot_macro_f1(rounds, f1, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, f1, color="#9467bd", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Test macro-F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", default=os.path.join(
        REPO_ROOT, "data", "results", "adv_xgb_eo_grid_ternary",
        "best_by_val_score.json"))
    base_data = os.path.join(REPO_ROOT, "data", "processed", "twitterAAE", "unbalanced_ternary")
    base_emb = os.path.join(REPO_ROOT, "data", "embeddings", "twitterAAE", "unbalanced_ternary")
    p.add_argument("--train_csv", default=os.path.join(base_data, "train.csv"))
    p.add_argument("--val_csv",   default=os.path.join(base_data, "val.csv"))
    p.add_argument("--test_csv",  default=os.path.join(base_data, "test.csv"))
    p.add_argument("--train_emb", default=os.path.join(base_emb,  "train_emb.npy"))
    p.add_argument("--val_emb",   default=os.path.join(base_emb,  "val_emb.npy"))
    p.add_argument("--test_emb",  default=os.path.join(base_emb,  "test_emb.npy"))
    p.add_argument("--out_dir",   default=os.path.join(
        REPO_ROOT, "data", "results", "plots", "eo_trend_ternary"))
    p.add_argument("--prefix",    default="eo_ternary")
    p.add_argument("--title_suffix", default="(ternary unbalanced TwitterAAE)")
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--tree_method", default="hist")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    K = NUM_CLASSES
    with open(args.best_json) as f:
        best = json.load(f)
    delta_aae = np.array([float(best[f"d_aae_{k}"]) for k in range(K)])
    delta_sae = np.array([float(best[f"d_sae_{k}"]) for k in range(K)])
    print(f"Best EO ternary config: lambda_adv={best['lambda_adv']}, "
          f"adv_c={best['adv_c']}, ratio={best['min_f1_ratio']}")
    print(f"  delta_AAE = {delta_aae.tolist()}")
    print(f"  delta_SAE = {delta_sae.tolist()}")

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, "dialect_strict")
    _, X_va, y_va, g_va = load_split(args.val_csv,   args.val_emb,   "dialect_strict")
    _, X_te, y_te, g_te = load_split(args.test_csv,  args.test_emb,  "dialect_strict")
    print(f"Loaded train={len(y_tr)} val={len(y_va)} test={len(y_te)}")

    params = {
        "max_depth": 5, "eta": 0.08,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "tree_method": args.tree_method, "device": args.device,
        "seed": 42, "verbosity": 0,
    }

    print("Retraining best EO ternary model while tracking per-round metrics...")
    hist = train_and_track(
        X_tr, y_tr, g_tr, X_te, y_te, g_te,
        params=params, num_round=args.num_round,
        lambda_adv=float(best["lambda_adv"]), adv_c=float(best["adv_c"]),
        delta_aae=delta_aae, delta_sae=delta_sae,
        warmup_rounds=args.warmup_rounds,
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    suffix = f" {args.title_suffix}" if args.title_suffix else ""
    subtitle = (f"EO-adversarial XGBoost (ternary){suffix} "
                f"(λ={best['lambda_adv']}, C={best['adv_c']}, "
                f"ratio={best['min_f1_ratio']})")

    pre = args.prefix
    rounds = hist["round"]
    aae_fpr = np.array(hist["aae_fpr"]); sae_fpr = np.array(hist["sae_fpr"])
    aae_fnr = np.array(hist["aae_fnr"]); sae_fnr = np.array(hist["sae_fnr"])
    for k in range(K):
        plot_class_group_metric(
            rounds, aae_fpr[:, k].tolist(), sae_fpr[:, k].tolist(),
            ylabel=f"FPR (class {k} vs rest)",
            title=f"Test FPR (class {k}) per round\n{subtitle}",
            out_path=os.path.join(out_dir, f"{pre}_fpr_class{k}_trend.png"))
        plot_class_group_metric(
            rounds, aae_fnr[:, k].tolist(), sae_fnr[:, k].tolist(),
            ylabel=f"FNR (class {k} vs rest)",
            title=f"Test FNR (class {k}) per round\n{subtitle}",
            out_path=os.path.join(out_dir, f"{pre}_fnr_class{k}_trend.png"))

    plot_macro_f1(
        rounds, hist["f1_macro"],
        out_path=os.path.join(out_dir, f"{pre}_f1_trend.png"),
        title=f"Test macro-F1 per round\n{subtitle}")

    hist_path = os.path.join(out_dir, f"{pre}_trend_history.json")
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"  saved {hist_path}")


if __name__ == "__main__":
    main()
