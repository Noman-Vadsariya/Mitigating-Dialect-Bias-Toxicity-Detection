"""Grid runner for ternary EO adversarial XGBoost (clone of run_adv_eo_grid.py)."""
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
from train_adv_xgb_ternary import (
    load_split, softmax, tune_logit_offsets, apply_logit_offsets, NUM_CLASSES,
)
from train_adv_xgb_eo_ternary import train_one_adv_model_eo_ternary
from fairness_postprocess_ternary import compute_metrics_ternary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/twitterAAE/unbalanced_ternary/train.csv")
    p.add_argument("--val_csv",   default="data/processed/twitterAAE/unbalanced_ternary/val.csv")
    p.add_argument("--test_csv",  default="data/processed/twitterAAE/unbalanced_ternary/test.csv")
    p.add_argument("--train_emb", default="data/embeddings/twitterAAE/unbalanced_ternary/train_emb.npy")
    p.add_argument("--val_emb",   default="data/embeddings/twitterAAE/unbalanced_ternary/val_emb.npy")
    p.add_argument("--test_emb",  default="data/embeddings/twitterAAE/unbalanced_ternary/test_emb.npy")
    p.add_argument("--dialect_col", default="dialect_strict")
    p.add_argument("--out_dir", default="data/results/adv_xgb_eo_grid_ternary")
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
    p.add_argument("--min_f1_ratios",
                   default="none,0.85,0.90,0.93,0.95,0.97,0.99")
    p.add_argument("--offset_grid_size", type=int, default=7)
    p.add_argument("--offset_range", type=float, default=1.5)
    p.add_argument("--no_projection", action="store_true")
    p.add_argument("--warmup_rounds", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, X_tr, y_tr, g_tr = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_va, y_va, g_va = load_split(args.val_csv,   args.val_emb,   args.dialect_col)
    _, X_te, y_te, g_te = load_split(args.test_csv,  args.test_emb,  args.dialect_col)

    print(f"Train={len(y_tr)} Val={len(y_va)} Test={len(y_te)}  Tag={args.tag}")
    print(f"Class distribution train: {np.bincount(y_tr, minlength=NUM_CLASSES).tolist()}")
    print(f"Config: ternary EO adversary on softmax probs, projection={not args.no_projection}, "
          f"reweighting=FALSE, warmup={args.warmup_rounds}")

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
    ratios = [None if t.lower() == "none" else float(t) for t in ratio_tokens]
    print(f"Grid: {len(cells)} cells x {len(ratios)} ratios = {len(cells)*len(ratios)} rows")

    rows = []
    for i, (lam, c) in enumerate(cells, 1):
        t0 = time.time()
        try:
            booster, _ = train_one_adv_model_eo_ternary(
                X_train=X_tr, y_train=y_tr, g_train=g_tr,
                X_val=X_va, y_val=y_va, g_val=g_va,
                params=base_params,
                lambda_adv=lam, adv_c=c,
                num_round=args.num_round,
                use_reweighting=False,
                use_projection=not args.no_projection,
                warmup_rounds=args.warmup_rounds,
            )
            v_margins = booster.predict(xgb.DMatrix(X_va), output_margin=True).reshape(len(y_va), NUM_CLASSES)
            t_margins = booster.predict(xgb.DMatrix(X_te), output_margin=True).reshape(len(y_te), NUM_CLASSES)
        except Exception as e:
            print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f}  "
                  f"TRAIN FAILED: {type(e).__name__}: {e}")
            continue
        train_elapsed = round(time.time() - t0, 2)

        for ratio in ratios:
            if ratio is None:
                d_aae = np.zeros(NUM_CLASSES)
                d_sae = np.zeros(NUM_CLASSES)
                tag = "none"
            else:
                try:
                    d_aae, d_sae, _ = tune_logit_offsets(
                        margins=v_margins, y=y_va, g=g_va,
                        dialect_col=args.dialect_col,
                        grid_size=args.offset_grid_size,
                        offset_range=args.offset_range,
                        min_f1_ratio=ratio,
                    )
                except Exception as e:
                    print(f"    tune FAILED ratio={ratio}: {e}; using zeros")
                    d_aae = np.zeros(NUM_CLASSES); d_sae = np.zeros(NUM_CLASSES)
                tag = f"{ratio:.2f}"

            v_pred = apply_logit_offsets(v_margins, g_va, d_aae, d_sae)
            t_pred = apply_logit_offsets(t_margins, g_te, d_aae, d_sae)
            vm = compute_metrics_ternary(y_va, v_pred, g_va)
            tm = compute_metrics_ternary(y_te, t_pred, g_te)

            row = {
                "lambda_adv": lam, "adv_c": c,
                "min_f1_ratio": ratio if ratio is not None else float("nan"),
            }
            for k in range(NUM_CLASSES):
                row[f"d_aae_{k}"] = float(d_aae[k])
                row[f"d_sae_{k}"] = float(d_sae[k])
            for k, v in vm.items():
                row[f"val_{k}"] = v
            for k, v in tm.items():
                row[f"test_{k}"] = v
            row["train_elapsed_sec"] = train_elapsed
            rows.append(row)
            print(f"  [{i:2d}/{len(cells)}] lam={lam:.3f} c={c:.2f} "
                  f"ratio={tag:>5s}  "
                  f"val_f1m={vm['f1_macro']:.4f} val_FPRgap={vm['mean_FPR_gap']:.4f}  "
                  f"test_f1m={tm['f1_macro']:.4f} test_FPRgap={tm['mean_FPR_gap']:.4f}")

        df_part = pd.DataFrame(rows)
        csv_path = os.path.join(args.out_dir, f"{args.tag}_grid.csv")
        df_part.to_csv(csv_path, index=False)

    with open(os.path.join(args.out_dir, f"{args.tag}_grid.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved: {args.out_dir}/{args.tag}_grid.csv  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
