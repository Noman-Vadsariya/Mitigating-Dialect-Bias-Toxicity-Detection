"""
Per-round trend plots for the best abstain (toxic/nontoxic/unsure) config.

Loads:  data/results/adv_xgb_eo_grid_abstain_<tag>/best_by_val_score.json
Writes: data/results/plots/abstain_<prefix>_*.png
"""

from __future__ import annotations

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

from train_adv_xgb import load_split, sigmoid  # noqa: E402
from train_adv_xgb_eo import (  # noqa: E402
    _project_out, adversary_grad_hess, fit_eo_adversary,
)
from train_adv_xgb_eo_abstain import (  # noqa: E402
    apply_thresholds_abstain, compute_metrics_abstain,
)


def train_and_track(X_tr, y_tr, g_tr, X_te, y_te, g_te,
                    params, num_round, lambda_adv, adv_c,
                    t_aae, d_aae, t_sae, d_sae, warmup_rounds=5):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)
    booster = None
    adv_clf: LogisticRegression | None = None

    hist = {k: [] for k in ["aae_fpr","sae_fpr","aae_fnr","sae_fnr",
                            "aae_abst","sae_abst","sel_f1","coverage"]}

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True)
            adv_clf = fit_eo_adversary(margin_tr, y_tr, g_tr, adv_c=adv_c)

        def objective(preds, dmat, _adv_clf=adv_clf):
            y = dmat.get_label()
            p = sigmoid(preds)
            grad_tox = p - y
            hess_tox = p * (1.0 - p)
            if _adv_clf is None or lambda_adv == 0.0:
                return grad_tox, np.clip(hess_tox, 1e-6, None)
            grad_adv, hess_adv = adversary_grad_hess(_adv_clf, preds, y, g_tr)
            grad_pred = _project_out(grad_tox, grad_adv)
            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            return grad, np.clip(hess, 1e-6, None)

        booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=1,
                            obj=objective, xgb_model=booster, verbose_eval=False)

        probs = sigmoid(booster.predict(dtest, output_margin=True))
        pred = apply_thresholds_abstain(probs, g_te, t_aae, d_aae, t_sae, d_sae)
        m = compute_metrics_abstain(y_te, pred, g_te)
        hist["aae_fpr"].append(m["AAE_FPR"])
        hist["sae_fpr"].append(m["SAE_FPR"])
        hist["aae_fnr"].append(m["AAE_FNR"])
        hist["sae_fnr"].append(m["SAE_FNR"])
        hist["aae_abst"].append(m["AAE_abstain_rate"])
        hist["sae_abst"].append(m["SAE_abstain_rate"])
        hist["sel_f1"].append(m["sel_f1"])
        hist["coverage"].append(m["coverage"])

    hist["round"] = list(range(1, num_round + 1))
    return hist


def plot_group_metric(rounds, aae, sae, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, aae, label="AAE", color="#d62728", linewidth=2)
    ax.plot(rounds, sae, label="SAE", color="#1f77b4", linewidth=2)
    gap = np.abs(np.array(aae, dtype=float) - np.array(sae, dtype=float))
    ax.plot(rounds, gap, label="|gap|", color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Boosting round"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def plot_scalar(rounds, vals, ylabel, title, out_path, color="#9467bd"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, vals, color=color, linewidth=2)
    ax.set_xlabel("Boosting round"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  saved {out_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv",   required=True)
    p.add_argument("--test_csv",  required=True)
    p.add_argument("--train_emb", required=True)
    p.add_argument("--val_emb",   required=True)
    p.add_argument("--test_emb",  required=True)
    p.add_argument("--out_dir",   default=os.path.join(REPO_ROOT, "data", "results", "plots"))
    p.add_argument("--prefix",    default="abstain_unbalanced")
    p.add_argument("--title_suffix", default="")
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--tree_method", default="hist")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dialect_col", default="dialect_strict")
    args = p.parse_args()

    with open(args.best_json) as f:
        best = json.load(f)
    print(f"Best: lam={best['lambda_adv']} c={best['adv_c']}  "
          f"AAE(t={float(best['t_aae']):.3f}, d={float(best['delta_aae']):.3f})  "
          f"SAE(t={float(best['t_sae']):.3f}, d={float(best['delta_sae']):.3f})")

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_te, y_te, g_te = load_split(args.test_csv,  args.test_emb,  args.dialect_col)
    print(f"train={len(y_tr)} test={len(y_te)}")

    params = {"max_depth": 5, "eta": 0.08, "subsample": 0.9,
              "colsample_bytree": 0.9, "tree_method": args.tree_method,
              "device": args.device, "seed": 42, "verbosity": 0}

    hist = train_and_track(
        X_tr, y_tr, g_tr, X_te, y_te, g_te,
        params=params, num_round=args.num_round,
        lambda_adv=float(best["lambda_adv"]), adv_c=float(best["adv_c"]),
        t_aae=float(best["t_aae"]), d_aae=float(best["delta_aae"]),
        t_sae=float(best["t_sae"]), d_sae=float(best["delta_sae"]),
        warmup_rounds=args.warmup_rounds,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f" {args.title_suffix}" if args.title_suffix else ""
    sub = (f"Abstain XGBoost{suffix} "
           f"(λ={best['lambda_adv']}, C={best['adv_c']}, "
           f"AAE t={float(best['t_aae']):.2f}±{float(best['delta_aae']):.2f}, "
           f"SAE t={float(best['t_sae']):.2f}±{float(best['delta_sae']):.2f})")
    pre = args.prefix
    od = args.out_dir

    plot_group_metric(hist["round"], hist["aae_fpr"], hist["sae_fpr"],
                      "Selective FPR", f"Test selective FPR per round\n{sub}",
                      os.path.join(od, f"{pre}_fpr_trend.png"))
    plot_group_metric(hist["round"], hist["aae_fnr"], hist["sae_fnr"],
                      "Selective FNR", f"Test selective FNR per round\n{sub}",
                      os.path.join(od, f"{pre}_fnr_trend.png"))
    plot_group_metric(hist["round"], hist["aae_abst"], hist["sae_abst"],
                      "Abstain rate", f"Test abstain rate per round\n{sub}",
                      os.path.join(od, f"{pre}_abstain_trend.png"))
    plot_scalar(hist["round"], hist["sel_f1"], "Selective F1",
                f"Test selective F1 per round\n{sub}",
                os.path.join(od, f"{pre}_self1_trend.png"))
    plot_scalar(hist["round"], hist["coverage"], "Coverage",
                f"Test coverage per round\n{sub}",
                os.path.join(od, f"{pre}_coverage_trend.png"), color="#17becf")

    with open(os.path.join(od, f"{pre}_trend_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"  saved {od}/{pre}_trend_history.json")


if __name__ == "__main__":
    main()
