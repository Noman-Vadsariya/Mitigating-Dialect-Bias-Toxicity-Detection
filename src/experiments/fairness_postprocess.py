
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

                                                       
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train_adv_xgb import (              
    load_split,
    sigmoid,
    train_one_adv_model,
    compute_group_weights,
)


                                                                               
         
                                                                               

def compute_metrics(y, pred, g):
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }
    for grp_name, grp_val in [("AAE", 1), ("SAE", 0)]:
        mask = g == grp_val
        yg, pg = y[mask], pred[mask]
        TP = ((yg == 1) & (pg == 1)).sum()
        FP = ((yg == 0) & (pg == 1)).sum()
        TN = ((yg == 0) & (pg == 0)).sum()
        FN = ((yg == 1) & (pg == 0)).sum()
        out[f"{grp_name}_FPR"] = float(FP / max(FP + TN, 1))
        out[f"{grp_name}_FNR"] = float(FN / max(FN + TP, 1))
        out[f"{grp_name}_TPR"] = float(TP / max(TP + FN, 1))
        out[f"{grp_name}_acc"] = float((TP + TN) / max(mask.sum(), 1))
        out[f"{grp_name}_pos_rate"] = float(pg.mean()) if mask.sum() > 0 else 0.0
    out["FPR_gap"] = abs(out["AAE_FPR"] - out["SAE_FPR"])
    out["FNR_gap"] = abs(out["AAE_FNR"] - out["SAE_FNR"])
    out["TPR_gap"] = abs(out["AAE_TPR"] - out["SAE_TPR"])
    out["mean_FPR"] = 0.5 * (out["AAE_FPR"] + out["SAE_FPR"])
    out["mean_FNR"] = 0.5 * (out["AAE_FNR"] + out["SAE_FNR"])
    out["abs_balance"] = abs(out["mean_FPR"] - out["mean_FNR"])
    p_aae_fav = 1.0 - out["AAE_pos_rate"]
    p_sae_fav = 1.0 - out["SAE_pos_rate"]
    out["DIfav"] = float(p_aae_fav / (p_sae_fav + 1e-8))
    out["DIunfav"] = float(out["AAE_pos_rate"] / (out["SAE_pos_rate"] + 1e-8))
    return out


def composite_score(m, f1_floor=0.70):
    penalty = 10.0 if m["f1"] < f1_floor else 0.0
    return (
        1.5 * m["mean_FPR"]
        + 1.0 * m["abs_balance"]
        + 0.5 * (m["FPR_gap"] + m["FNR_gap"])
        - 0.5 * m["f1"]
        + penalty
    )


                                                                               
                                          
                                                                               

def fit_platt_per_group(probs_val, y_val, g_val):
    calibrators = {}
    for grp in [0, 1]:
        mask = g_val == grp
        y_g = y_val[mask]
        if mask.sum() < 10 or len(np.unique(y_g)) < 2:
            calibrators[grp] = None
            continue
        p = np.clip(probs_val[mask], 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        try:
            clf = LogisticRegression(C=1e6, max_iter=1000)
            clf.fit(logits.reshape(-1, 1), y_g)
            calibrators[grp] = clf
        except Exception:
            calibrators[grp] = None
    return calibrators


def apply_platt_per_group(probs, g, calibrators):
    out = probs.copy()
    for grp in [0, 1]:
        mask = g == grp
        if calibrators.get(grp) is None or mask.sum() == 0:
            continue
        p = np.clip(probs[mask], 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        out[mask] = calibrators[grp].predict_proba(logits.reshape(-1, 1))[:, 1]
    return out


                                                                               
                                        
                                                                               

def calibrated_equalized_odds(probs_val, y_val, g_val):
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
        best_gamma, best_c = 0.0, 0.5
        best_gap = float("inf")
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
                    best_c = c
        ceo_params[grp] = {"gamma": float(best_gamma), "constant": float(best_c)}
    return ceo_params


def apply_ceo(probs, g, ceo_params):
    out = probs.copy()
    for grp in [0, 1]:
        mask = g == grp
        if mask.sum() == 0:
            continue
        p = ceo_params[grp]
        out[mask] = (1 - p["gamma"]) * probs[mask] + p["gamma"] * p["constant"]
    return out


                                                                               
                                                                                 
                                                                               

def train_reweighted_xgb(X_train, y_train, g_train, params, num_round):
    sw = compute_group_weights(g_train).astype(np.float32)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw)
    return xgb.train(
        params={**params, "objective": "binary:logistic"},
        dtrain=dtrain,
        num_boost_round=num_round,
        verbose_eval=False,
    )


def ensemble_probs(probs_a, probs_b, alpha=0.5):
    return alpha * probs_a + (1 - alpha) * probs_b


                                                                               
                                           
                                                                               

def fit_reject_option(probs_val, y_val, g_val, bandwidths=None):
    if bandwidths is None:
        bandwidths = np.linspace(0.0, 0.45, 46)
    baseline_f1 = f1_score(y_val, (probs_val >= 0.5).astype(int))
    min_f1 = baseline_f1 * 0.95
    best_theta, best_score = 0.0, float("inf")
    for theta in bandwidths:
        low, high = 0.5 - theta, 0.5 + theta
        pred = (probs_val >= 0.5).astype(int).copy()
        uncertain = (probs_val >= low) & (probs_val <= high)
        pred[uncertain & (g_val == 1)] = 0                              
        pred[uncertain & (g_val == 0)] = 1                                     
        curr_f1 = f1_score(y_val, pred)
        if curr_f1 < min_f1:
            continue
        m = compute_metrics(y_val, pred, g_val)
        score = m["FPR_gap"] + m["FNR_gap"]
        if score < best_score:
            best_score = score
            best_theta = float(theta)
    return {"theta": best_theta}


def apply_reject_option(probs, g, roc_params):
    theta = roc_params["theta"]
    low, high = 0.5 - theta, 0.5 + theta
    pred = (probs >= 0.5).astype(int).copy()
    uncertain = (probs >= low) & (probs <= high)
    pred[uncertain & (g == 1)] = 0
    pred[uncertain & (g == 0)] = 1
    return pred


                                                                               
                                                
                                                                               

def run_one_cell(
    X_train, y_train, g_train,
    X_val, y_val, g_val,
    X_test, y_test, g_test,
    xgb_params, num_round,
    lambda_adv, adv_c,
    enable_ceo, enable_platt, enable_reductions, enable_roc,
    use_reweighting=True,
):
                      
    booster, _history = train_one_adv_model(
        X_train=X_train, y_train=y_train, g_train=g_train,
        X_val=X_val, y_val=y_val, g_val=g_val,
        params=xgb_params,
        lambda_adv=lambda_adv, adv_c=adv_c,
        num_round=num_round,
        dialect_col="dialect_strict",
        use_leaf_adv=True,
        use_reweighting=use_reweighting,
    )
                        
    dval, dtest = xgb.DMatrix(X_val), xgb.DMatrix(X_test)
    probs_val = sigmoid(booster.predict(dval, output_margin=True))
    probs_test = sigmoid(booster.predict(dtest, output_margin=True))

                                                                   
    if enable_reductions:
        rw_booster = train_reweighted_xgb(
            X_train, y_train, g_train, xgb_params, num_round
        )
        rw_val = rw_booster.predict(xgb.DMatrix(X_val))
        rw_test = rw_booster.predict(xgb.DMatrix(X_test))
        probs_val = ensemble_probs(probs_val, rw_val, alpha=0.5)
        probs_test = ensemble_probs(probs_test, rw_test, alpha=0.5)

                                  
    if enable_platt:
        cal = fit_platt_per_group(probs_val, y_val, g_val)
        probs_val = apply_platt_per_group(probs_val, g_val, cal)
        probs_test = apply_platt_per_group(probs_test, g_test, cal)

                                            
    if enable_ceo:
        ceo = calibrated_equalized_odds(probs_val, y_val, g_val)
        probs_val = apply_ceo(probs_val, g_val, ceo)
        probs_test = apply_ceo(probs_test, g_test, ceo)

                                               
    if enable_roc:
        roc = fit_reject_option(probs_val, y_val, g_val)
        val_pred = apply_reject_option(probs_val, g_val, roc)
        test_pred = apply_reject_option(probs_test, g_test, roc)
    else:
        val_pred = (probs_val >= 0.5).astype(int)
        test_pred = (probs_test >= 0.5).astype(int)

    val_metrics = compute_metrics(y_val, val_pred, g_val)
    test_metrics = compute_metrics(y_test, test_pred, g_test)
    return val_metrics, test_metrics


                                                                               
                    
                                                                               

def run_grid(args):
    _, X_train, y_train, g_train = load_split(args.train_csv, args.train_emb, args.dialect_col)
    _, X_val, y_val, g_val = load_split(args.val_csv, args.val_emb, args.dialect_col)
    _, X_test, y_test, g_test = load_split(args.test_csv, args.test_emb, args.dialect_col)

    print(f"Train: {len(y_train)} (AAE={(g_train==1).sum()}, SAE={(g_train==0).sum()})")
    print(f"Val:   {len(y_val)} | Test: {len(y_test)}")
    print(f"Tag: {args.tag}")
    print(f"Techniques: CEO={args.enable_1} Platt={args.enable_2} "
          f"Reductions={args.enable_3} ROC={args.enable_4}")

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

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    adv_c_grid = [float(x) for x in args.adv_c_grid.split(",")]
    grid = [(lam, c) for lam in lambda_grid for c in adv_c_grid]
    print(f"Grid size: {len(grid)} cells "
          f"(lambda={lambda_grid}, adv_c={adv_c_grid})")

    all_cells = []
    best_score = float("inf")
    best_cell = None

    for i, (lam, c) in enumerate(grid, 1):
        t0 = time.time()
        val_m, test_m = run_one_cell(
            X_train, y_train, g_train,
            X_val, y_val, g_val,
            X_test, y_test, g_test,
            xgb_params, args.num_round,
            lambda_adv=lam, adv_c=c,
            enable_ceo=args.enable_1,
            enable_platt=args.enable_2,
            enable_reductions=args.enable_3,
            enable_roc=args.enable_4,
            use_reweighting=True,
        )
        elapsed = time.time() - t0
        val_score = composite_score(val_m)
        print(f"  [{i:2d}/{len(grid)}] lam={lam:.3f} adv_c={c:.2f} "
              f"val_f1={val_m['f1']:.4f} val_FPR={val_m['mean_FPR']:.4f} "
              f"val_score={val_score:.4f}  ({elapsed:.1f}s)")
        cell = {
            "lambda_adv": lam,
            "adv_c": c,
            "val_score": float(val_score),
            "val_metrics": val_m,
            "test_metrics": test_m,
        }
        all_cells.append(cell)
        if val_score < best_score:
            best_score = val_score
            best_cell = cell

    return all_cells, best_cell


                                                                               
     
                                                                               

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial XGB + 4 fairness post-processing techniques"
    )
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--train_emb", default="data/embeddings/train_emb.npy")
    parser.add_argument("--val_emb", default="data/embeddings/val_emb.npy")
    parser.add_argument("--test_emb", default="data/embeddings/test_emb.npy")
    parser.add_argument("--dialect_col", default="dialect_strict")
    parser.add_argument("--out_dir", default="data/results/fairness_ablation")
    parser.add_argument("--tag", default="run")

    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree_method", default="hist")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--lambda_grid", default="0.0,0.05,0.1,0.25,0.5")
    parser.add_argument("--adv_c_grid", default="0.5,1.0,2.0")

    parser.add_argument("--enable_1", action="store_true", help="Calibrated Equalized Odds")
    parser.add_argument("--enable_2", action="store_true", help="Per-group Platt calibration")
    parser.add_argument("--enable_3", action="store_true", help="Reductions ensemble")
    parser.add_argument("--enable_4", action="store_true", help="Reject Option Classification")
    parser.add_argument("--enable_all", action="store_true")
    args = parser.parse_args()

    if args.enable_all:
        args.enable_1 = args.enable_2 = args.enable_3 = args.enable_4 = True

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    all_cells, best = run_grid(args)
    elapsed = time.time() - t0

                                                        
    out = {
        "tag": args.tag,
        "enable_1_ceo": args.enable_1,
        "enable_2_platt": args.enable_2,
        "enable_3_reductions": args.enable_3,
        "enable_4_roc": args.enable_4,
        "best_lambda_adv": best["lambda_adv"],
        "best_adv_c": best["adv_c"],
        "val_score": best["val_score"],
        "val_metrics": best["val_metrics"],
                                                               
        **best["test_metrics"],
        "elapsed_sec": elapsed,
        "all_cells": all_cells,
    }

    path = os.path.join(args.out_dir, f"{args.tag}_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== BEST CELL FOR TAG={args.tag} ===")
    print(f"  lambda_adv={best['lambda_adv']}  adv_c={best['adv_c']}")
    print(f"  val_score={best['val_score']:.4f}  F1={best['test_metrics']['f1']:.4f}")
    print(f"  test FPR (AAE/SAE)={best['test_metrics']['AAE_FPR']:.4f}/"
          f"{best['test_metrics']['SAE_FPR']:.4f}  "
          f"FNR (AAE/SAE)={best['test_metrics']['AAE_FNR']:.4f}/"
          f"{best['test_metrics']['SAE_FNR']:.4f}")
    print(f"  test mean_FPR={best['test_metrics']['mean_FPR']:.4f} "
          f"mean_FNR={best['test_metrics']['mean_FNR']:.4f}")
    print(f"  FPR_gap={best['test_metrics']['FPR_gap']:.4f}  "
          f"FNR_gap={best['test_metrics']['FNR_gap']:.4f}")
    print(f"\nSaved to: {path}  (elapsed {elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
