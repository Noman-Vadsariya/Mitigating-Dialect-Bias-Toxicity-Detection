"""Per-round trend plots for the best BERT-MLP-adversary EO config.

Loads:  data/results/adv_xgb_eo_grid_bert_<tag>/best_by_val_score.json
Writes: data/results/plots/bert_<prefix>_{fpr,fnr,f1}_trend.png
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import f1_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "experiments"))

from train_adv_xgb import load_split, sigmoid, apply_thresholds  # noqa: E402
from train_adv_xgb_eo import _project_out  # noqa: E402
from train_adv_xgb_eo_bert import BertAdversaryState, _device  # noqa: E402


def group_fpr_fnr(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    neg = y_true == 0; pos = y_true == 1
    fpr = float(y_pred[neg].mean()) if neg.any() else float("nan")
    fnr = float(1.0 - y_pred[pos].mean()) if pos.any() else float("nan")
    return fpr, fnr


def train_and_track(X_tr, y_tr, g_tr, X_te, y_te, g_te,
                    params, num_round, lambda_adv,
                    adv_hidden, adv_lr, adv_dropout, adv_weight_decay,
                    adv_epochs_per_round, adv_batch_size,
                    t_aae, t_sae, warmup_rounds, device):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)

    booster = None
    adv_state: Optional[BertAdversaryState] = None
    torch_device = _device(device)

    aae_fpr, aae_fnr, sae_fpr, sae_fnr, f1_hist = [], [], [], [], []

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True)
            if adv_state is None:
                adv_state = BertAdversaryState(
                    emb=X_tr, hidden=adv_hidden, dropout=adv_dropout,
                    lr=adv_lr, weight_decay=adv_weight_decay, device=torch_device,
                )
            adv_state.fit(margin_tr, y_tr, g_tr,
                          epochs=adv_epochs_per_round, batch_size=adv_batch_size)

        def objective(preds, dmat, _adv=adv_state):
            y = dmat.get_label()
            p = sigmoid(preds)
            grad_tox = p - y; hess_tox = p * (1.0 - p)
            if _adv is None or lambda_adv == 0.0:
                return grad_tox, np.clip(hess_tox, 1e-6, None)
            grad_adv, hess_adv = _adv.grad_hess(preds, y, g_tr)
            grad_pred = _project_out(grad_tox, grad_adv)
            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            return grad, np.clip(hess, 1e-6, None)

        booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=1,
                            obj=objective, xgb_model=booster, verbose_eval=False)

        probs = sigmoid(booster.predict(dtest, output_margin=True))
        pred = apply_thresholds(probs, g_te, t_aae, t_sae)
        aae = g_te == 1; sae = g_te == 0
        fa, na = group_fpr_fnr(y_te[aae], pred[aae])
        fs, ns = group_fpr_fnr(y_te[sae], pred[sae])
        aae_fpr.append(fa); aae_fnr.append(na)
        sae_fpr.append(fs); sae_fnr.append(ns)
        f1_hist.append(float(f1_score(y_te, pred)))

    return {"round": list(range(1, num_round + 1)),
            "aae_fpr": aae_fpr, "aae_fnr": aae_fnr,
            "sae_fpr": sae_fpr, "sae_fnr": sae_fnr, "f1": f1_hist}


def plot_group(rounds, aae, sae, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, aae, label="AAE", color="#d62728", linewidth=2)
    ax.plot(rounds, sae, label="SAE", color="#1f77b4", linewidth=2)
    gap = np.abs(np.array(aae) - np.array(sae))
    ax.plot(rounds, gap, label="|gap|", color="#2ca02c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Boosting round"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def plot_scalar(rounds, vals, ylabel, title, out, color="#9467bd"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, vals, color=color, linewidth=2)
    ax.set_xlabel("Boosting round"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--train_emb", required=True)
    p.add_argument("--val_emb", required=True)
    p.add_argument("--test_emb", required=True)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data", "results", "plots"))
    p.add_argument("--prefix", default="bert_unbalanced")
    p.add_argument("--title_suffix", default="")
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--tree_method", default="hist")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dialect_col", default="dialect_strict")
    p.add_argument("--adv_dropout", type=float, default=0.2)
    p.add_argument("--adv_weight_decay", type=float, default=1e-4)
    p.add_argument("--adv_epochs_per_round", type=int, default=2)
    p.add_argument("--adv_batch_size", type=int, default=1024)
    args = p.parse_args()

    with open(args.best_json) as f:
        best = json.load(f)
    print(f"Best BERT-adv: lam={best['lambda_adv']} hidden={best['adv_hidden']} "
          f"lr={best['adv_lr']}  t_AAE={float(best['t_aae']):.4f} t_SAE={float(best['t_sae']):.4f}")

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_te, y_te, g_te = load_split(args.test_csv, args.test_emb, args.dialect_col)
    print(f"train={len(y_tr)} test={len(y_te)}")

    params = {"max_depth": 5, "eta": 0.08, "subsample": 0.9, "colsample_bytree": 0.9,
              "tree_method": args.tree_method, "device": args.device,
              "seed": 42, "verbosity": 0}

    hist = train_and_track(
        X_tr, y_tr, g_tr, X_te, y_te, g_te,
        params=params, num_round=args.num_round,
        lambda_adv=float(best["lambda_adv"]),
        adv_hidden=int(best["adv_hidden"]),
        adv_lr=float(best["adv_lr"]),
        adv_dropout=args.adv_dropout,
        adv_weight_decay=args.adv_weight_decay,
        adv_epochs_per_round=args.adv_epochs_per_round,
        adv_batch_size=args.adv_batch_size,
        t_aae=float(best["t_aae"]), t_sae=float(best["t_sae"]),
        warmup_rounds=args.warmup_rounds,
        device=args.device,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f" {args.title_suffix}" if args.title_suffix else ""
    sub = (f"BERT-MLP adv XGBoost{suffix} "
           f"(λ={best['lambda_adv']}, h={best['adv_hidden']}, lr={best['adv_lr']}, "
           f"t_AAE={float(best['t_aae']):.3f}, t_SAE={float(best['t_sae']):.3f})")
    pre = args.prefix; od = args.out_dir
    plot_group(hist["round"], hist["aae_fpr"], hist["sae_fpr"],
               "FPR", f"Test FPR per round\n{sub}",
               os.path.join(od, f"{pre}_fpr_trend.png"))
    plot_group(hist["round"], hist["aae_fnr"], hist["sae_fnr"],
               "FNR", f"Test FNR per round\n{sub}",
               os.path.join(od, f"{pre}_fnr_trend.png"))
    plot_scalar(hist["round"], hist["f1"], "Test F1",
                f"Test F1 per round\n{sub}",
                os.path.join(od, f"{pre}_f1_trend.png"))

    with open(os.path.join(od, f"{pre}_trend_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"  saved {od}/{pre}_trend_history.json")


if __name__ == "__main__":
    main()
