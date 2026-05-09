"""Grid runner for binary EO XGBoost with abstention output.

Trains the same binary EO model used by run_adv_eo_grid.py (no changes to
training), then post-processes with per-group (threshold, deadband) tuning so
that predictions become {0=nontoxic, 1=toxic, 2=unsure}.
"""
from __future__ import annotations

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

from train_adv_xgb import load_split, sigmoid              
from train_adv_xgb_eo import train_one_adv_model_eo              
from train_adv_xgb_eo_abstain import (              
    apply_thresholds_abstain,
    compute_metrics_abstain,
    tune_thresholds_with_abstain,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/twitterAAE/unbalanced/train.csv")
    p.add_argument("--val_csv",   default="data/processed/twitterAAE/unbalanced/val.csv")
    p.add_argument("--test_csv",  default="data/processed/twitterAAE/unbalanced/test.csv")
    p.add_argument("--train_emb", default="data/embeddings/twitterAAE/unbalanced/train_emb.npy")
    p.add_argument("--val_emb",   default="data/embeddings/twitterAAE/unbalanced/val_emb.npy")
    p.add_argument("--test_emb",  default="data/embeddings/twitterAAE/unbalanced/test_emb.npy")
    p.add_argument("--dialect_col", default="dialect_strict")
    p.add_argument("--out_dir", default="data/results/adv_xgb_eo_grid_abstain")
    p.add_argument("--tag", default="run")
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.08)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tree_method", default="hist")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lambda_grid", required=True)
    p.add_argument("--adv_c_grid", required=True)
    p.add_argument("--min_f1_ratios", default="0.85,0.90,0.95,0.99")
    p.add_argument("--min_coverages", default="0.70,0.85,0.95")
    p.add_argument("--delta_max", type=float, default=0.30)
    p.add_argument("--t_grid_size", type=int, default=21)
    p.add_argument("--delta_grid_size", type=int, default=11)
    p.add_argument("--no_projection", action="store_true")
    p.add_argument("--warmup_rounds", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_va, y_va, g_va = load_split(args.val_csv,   args.val_emb,   args.dialect_col)
    _, X_te, y_te, g_te = load_split(args.test_csv,  args.test_emb,  args.dialect_col)

    print(f"Train={len(y_tr)} Val={len(y_va)} Test={len(y_te)}  Tag={args.tag}")
    print("Mode: binary EO training + abstain post-processing "
          "(predictions in {0=nontoxic, 1=toxic, 2=unsure})")

    base_params = {
        "max_depth": args.max_depth, "eta": args.eta,
        "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method, "device": args.device,
        "seed": args.seed, "verbosity": 0,
    }

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    adv_c_grid = [float(x) for x in args.adv_c_grid.split(",")]
    cells = [(l, c) for l in lambda_grid for c in adv_c_grid]

    f1_ratios = [float(s) for s in args.min_f1_ratios.split(",") if s.strip()]
    cov_floors = [float(s) for s in args.min_coverages.split(",") if s.strip()]
    print(f"Grid: {len(cells)} cells x {len(f1_ratios)} f1-ratios "
          f"x {len(cov_floors)} cov-floors = "
          f"{len(cells)*len(f1_ratios)*len(cov_floors)} rows")

    rows = []
    for i, (lam, c) in enumerate(cells, 1):
        t0 = time.time()
        try:
            booster, _ = train_one_adv_model_eo(
                X_train=X_tr, y_train=y_tr, g_train=g_tr,
                X_val=X_va, y_val=y_va, g_val=g_va,
                params=base_params,
                lambda_adv=lam, adv_c=c,
                num_round=args.num_round,
                use_reweighting=False,
                use_projection=not args.no_projection,
                warmup_rounds=args.warmup_rounds,
            )
            v_prob = sigmoid(booster.predict(xgb.DMatrix(X_va), output_margin=True))
            t_prob = sigmoid(booster.predict(xgb.DMatrix(X_te), output_margin=True))
        except Exception as e:
            print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f}  "
                  f"TRAIN FAILED: {type(e).__name__}: {e}")
            continue
        train_elapsed = round(time.time() - t0, 2)

        for ratio in f1_ratios:
            for cov in cov_floors:
                t_aae, d_aae, t_sae, d_sae, _ = tune_thresholds_with_abstain(
                    probs=v_prob, y=y_va, g=g_va,
                    t_grid_size=args.t_grid_size,
                    delta_grid_size=args.delta_grid_size,
                    delta_max=args.delta_max,
                    min_f1_ratio=ratio,
                    min_coverage=cov,
                )
                v_pred = apply_thresholds_abstain(v_prob, g_va, t_aae, d_aae, t_sae, d_sae)
                t_pred = apply_thresholds_abstain(t_prob, g_te, t_aae, d_aae, t_sae, d_sae)
                vm = compute_metrics_abstain(y_va, v_pred, g_va)
                tm = compute_metrics_abstain(y_te, t_pred, g_te)

                row = {
                    "lambda_adv": lam, "adv_c": c,
                    "min_f1_ratio": ratio, "min_coverage": cov,
                    "t_aae": float(t_aae), "delta_aae": float(d_aae),
                    "t_sae": float(t_sae), "delta_sae": float(d_sae),
                }
                for k, v in vm.items():
                    row[f"val_{k}"] = v
                for k, v in tm.items():
                    row[f"test_{k}"] = v
                row["train_elapsed_sec"] = train_elapsed
                rows.append(row)

                print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f} "
                      f"r={ratio:.2f} cov>={cov:.2f}  "
                      f"t_AAE=({t_aae:.2f},+/-{d_aae:.2f}) "
                      f"t_SAE=({t_sae:.2f},+/-{d_sae:.2f})  "
                      f"val_selF1={vm['sel_f1']:.4f} cov={vm['coverage']:.3f} "
                      f"gap={vm['FPR_gap']:.4f}  "
                      f"test_selF1={tm['sel_f1']:.4f} cov={tm['coverage']:.3f} "
                      f"gap={tm['FPR_gap']:.4f}")

        df_part = pd.DataFrame(rows)
        df_part.to_csv(os.path.join(args.out_dir, f"{args.tag}_grid.csv"), index=False)

    with open(os.path.join(args.out_dir, f"{args.tag}_grid.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved: {args.out_dir}/{args.tag}_grid.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
