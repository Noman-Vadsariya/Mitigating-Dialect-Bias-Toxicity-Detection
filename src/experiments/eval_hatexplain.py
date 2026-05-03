"""Evaluate generalization of best EO-debiased XGBoost on HateXplain.

Loads the best EO config selected on the unbalanced TwitterAAE val set,
retrains on the unbalanced TwitterAAE train split, then evaluates on
the HateXplain test set using the same dialect-specific thresholds.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from train_adv_xgb import load_split, sigmoid, apply_thresholds
from train_adv_xgb_eo import train_one_adv_model_eo
from fairness_postprocess import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_json",
                   default="data/results/adv_xgb_eo_grid_unbalanced/best_by_val_score.json",
                   help="Path to best_by_*_score.json from EO grid")
    p.add_argument("--train_csv", default="data/processed/twitterAAE/unbalanced/train.csv")
    p.add_argument("--val_csv",   default="data/processed/twitterAAE/unbalanced/val.csv")
    p.add_argument("--src_test_csv", default="data/processed/twitterAAE/unbalanced/test.csv",
                   help="In-domain test set (for sanity-check metrics)")
    p.add_argument("--gen_test_csv", default="data/processed/hatexplain/test.csv",
                   help="Out-of-domain generalization test set")
    p.add_argument("--train_emb", default="data/embeddings/twitterAAE/unbalanced/train_emb.npy")
    p.add_argument("--val_emb",   default="data/embeddings/twitterAAE/unbalanced/val_emb.npy")
    p.add_argument("--src_test_emb", default="data/embeddings/twitterAAE/unbalanced/test_emb.npy")
    p.add_argument("--gen_test_emb", default="data/embeddings/hatexplain/test_emb.npy")
    p.add_argument("--dialect_col", default="dialect_strict")
    p.add_argument("--out_dir", default="data/results/hatexplain_generalization")
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.08)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tree_method", default="hist")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--no_projection", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.best_json) as f:
        best = json.load(f)
    lam = float(best["lambda_adv"])
    adv_c = float(best["adv_c"])
    t_aae = float(best["t_aae"])
    t_sae = float(best["t_sae"])
    ratio = best.get("min_f1_ratio")
    print(f"Best EO config from {args.best_json}:")
    print(f"  lambda_adv={lam}  adv_c={adv_c}  min_f1_ratio={ratio}")
    print(f"  thresholds: t_AAE={t_aae:.4f}  t_SAE={t_sae:.4f}")

    print("\nLoading splits...")
    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_va, y_va, g_va = load_split(args.val_csv,   args.val_emb,   args.dialect_col)
    _, X_st, y_st, g_st = load_split(args.src_test_csv, args.src_test_emb, args.dialect_col)
    _, X_gt, y_gt, g_gt = load_split(args.gen_test_csv, args.gen_test_emb, args.dialect_col)
    print(f"  Train={len(y_tr)}  Val={len(y_va)}")
    print(f"  TwitterAAE test={len(y_st)}  HateXplain test={len(y_gt)}")
    print(f"  HateXplain group counts: AAE={int((g_gt==1).sum())} SAE={int((g_gt==0).sum())}")
    print(f"  HateXplain label distribution: pos={int((y_gt==1).sum())} neg={int((y_gt==0).sum())}")

    base_params = {
        "max_depth": args.max_depth, "eta": args.eta,
        "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method, "device": args.device,
        "seed": args.seed, "verbosity": 0,
    }

    print("\nRetraining best EO model on unbalanced TwitterAAE train...")
    t0 = time.time()
    booster, _ = train_one_adv_model_eo(
        X_train=X_tr, y_train=y_tr, g_train=g_tr,
        X_val=X_va,   y_val=y_va,   g_val=g_va,
        params=base_params,
        lambda_adv=lam, adv_c=adv_c,
        num_round=args.num_round,
        use_reweighting=False,
        use_projection=not args.no_projection,
        warmup_rounds=args.warmup_rounds,
    )
    print(f"  trained in {time.time()-t0:.1f}s")

    s_prob = sigmoid(booster.predict(xgb.DMatrix(X_st), output_margin=True))
    g_prob = sigmoid(booster.predict(xgb.DMatrix(X_gt), output_margin=True))

    s_pred = apply_thresholds(s_prob, g_st, t_aae, t_sae)
    g_pred = apply_thresholds(g_prob, g_gt, t_aae, t_sae)

    src_metrics = compute_metrics(y_st, s_pred, g_st)
    gen_metrics = compute_metrics(y_gt, g_pred, g_gt)

    def fmt(m):
        return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in m.items()}

    print("\n=== In-domain (TwitterAAE unbalanced test) ===")
    for k, v in src_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"  {k}: {v}")

    print("\n=== Out-of-domain (HateXplain test) ===")
    for k, v in gen_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"  {k}: {v}")

    # Also report metrics with default 0.5 threshold for context
    s_pred_05 = (s_prob >= 0.5).astype(int)
    g_pred_05 = (g_prob >= 0.5).astype(int)
    src_metrics_05 = compute_metrics(y_st, s_pred_05, g_st)
    gen_metrics_05 = compute_metrics(y_gt, g_pred_05, g_gt)

    print("\n=== HateXplain (default threshold 0.5/0.5) ===")
    for k, v in gen_metrics_05.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"  {k}: {v}")

    out = {
        "best_config": {
            "lambda_adv": lam, "adv_c": adv_c,
            "min_f1_ratio": ratio, "t_aae": t_aae, "t_sae": t_sae,
        },
        "in_domain_test_with_tuned_thresholds": fmt(src_metrics),
        "out_of_domain_test_with_tuned_thresholds": fmt(gen_metrics),
        "in_domain_test_threshold_0.5": fmt(src_metrics_05),
        "out_of_domain_test_threshold_0.5": fmt(gen_metrics_05),
    }
    out_path = os.path.join(args.out_dir, "hatexplain_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
