
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
from train_adv_xgb import (              
    load_split,
    sigmoid,
    train_one_adv_model,
    tune_thresholds,
    apply_thresholds,
)
from fairness_postprocess import compute_metrics              


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--test_csv", default="data/processed/test.csv")
    p.add_argument("--train_emb", default="data/embeddings/train_emb.npy")
    p.add_argument("--val_emb", default="data/embeddings/val_emb.npy")
    p.add_argument("--test_emb", default="data/embeddings/test_emb.npy")
    p.add_argument("--dialect_col", default="dialect_strict")
    p.add_argument("--out_dir", default="data/results/adv_xgb_thresh_grid")
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
    p.add_argument(
        "--min_f1_ratios",
        default="none,0.85,0.90,0.93,0.95,0.97,0.99",
    )
    p.add_argument("--threshold_grid_size", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_va, y_va, g_va = load_split(args.val_csv, args.val_emb, args.dialect_col)
    _, X_te, y_te, g_te = load_split(args.test_csv, args.test_emb, args.dialect_col)

    print(f"Train={len(y_tr)} (AAE={(g_tr==1).sum()}, SAE={(g_tr==0).sum()}) "
          f"Val={len(y_va)} Test={len(y_te)}")
    print(f"Tag: {args.tag}")
    print("Config: leaf_adv=True, reweighting=FALSE, "
          "threshold_tuning=PER-GROUP, post_processing=NONE")

    base_params = {
        "max_depth": args.max_depth, "eta": args.eta,
        "subsample": args.subsample, "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method, "device": args.device,
        "seed": args.seed, "verbosity": 0,
    }

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    adv_c_grid = [float(x) for x in args.adv_c_grid.split(",")]
    cells = [(l, c) for l in lambda_grid for c in adv_c_grid]

    ratio_tokens = [s.strip() for s in args.min_f1_ratios.split(",") if s.strip()]
    ratios = []
    for tok in ratio_tokens:
        if tok.lower() == "none":
            ratios.append(None)
        else:
            ratios.append(float(tok))
    print(f"Grid: {len(cells)} cells x {len(ratios)} ratios = "
          f"{len(cells)*len(ratios)} rows  lambda={lambda_grid}  "
          f"adv_c={adv_c_grid}  ratios={ratios}")

    rows = []
    for i, (lam, c) in enumerate(cells, 1):
        t0 = time.time()
        try:
            booster, _ = train_one_adv_model(
                X_train=X_tr, y_train=y_tr, g_train=g_tr,
                X_val=X_va, y_val=y_va, g_val=g_va,
                params=base_params,
                lambda_adv=lam, adv_c=c,
                num_round=args.num_round,
                dialect_col=args.dialect_col,
                use_leaf_adv=True,
                use_reweighting=False,
            )
            v_prob = sigmoid(booster.predict(xgb.DMatrix(X_va), output_margin=True))
            t_prob = sigmoid(booster.predict(xgb.DMatrix(X_te), output_margin=True))
        except Exception as e:
            print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f}  "
                  f"TRAIN FAILED: {type(e).__name__}: {e}")
            continue
        train_elapsed = round(time.time() - t0, 2)

        for ratio in ratios:
            if ratio is None:
                t_aae, t_sae = 0.5, 0.5
                tag = "none"
            else:
                try:
                    t_aae, t_sae, _ = tune_thresholds(
                        probs=v_prob, y=y_va, g=g_va,
                        dialect_col=args.dialect_col,
                        grid_size=args.threshold_grid_size,
                        min_f1_ratio=ratio,
                    )
                except Exception as e:
                    print(f"    tune FAILED ratio={ratio}: {e}; using 0.5/0.5")
                    t_aae, t_sae = 0.5, 0.5
                tag = f"{ratio:.2f}"

            v_pred = apply_thresholds(v_prob, g_va, t_aae, t_sae)
            t_pred = apply_thresholds(t_prob, g_te, t_aae, t_sae)
            vm = compute_metrics(y_va, v_pred, g_va)
            tm = compute_metrics(y_te, t_pred, g_te)

            row = {
                "lambda_adv": lam,
                "adv_c": c,
                "min_f1_ratio": ratio if ratio is not None else float("nan"),
                "t_aae": float(t_aae),
                "t_sae": float(t_sae),
            }
            for k, v in vm.items():
                row[f"val_{k}"] = v
            for k, v in tm.items():
                row[f"test_{k}"] = v
            row["train_elapsed_sec"] = train_elapsed
            rows.append(row)
            print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f} "
                  f"ratio={tag:>5s}  t_aae={t_aae:.3f} t_sae={t_sae:.3f}  "
                  f"val_f1={vm['f1']:.4f} val_FPR_gap={vm['FPR_gap']:.4f}  "
                  f"test_f1={tm['f1']:.4f} test_FPR_gap={tm['FPR_gap']:.4f}")

        df_part = pd.DataFrame(rows)
        csv_path = os.path.join(args.out_dir, f"{args.tag}_grid.csv")
        df_part.to_csv(csv_path, index=False)

    csv_path = os.path.join(args.out_dir, f"{args.tag}_grid.csv")
    with open(os.path.join(args.out_dir, f"{args.tag}_grid.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
