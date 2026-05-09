"""Binary-with-abstention ("toxic / nontoxic / unsure") classifier.

This is a copy of the binary EO setup with a third "unsure" output added at
post-processing time. Training is unchanged — we reuse the binary EO adversary
from train_adv_xgb_eo.py exactly. What changes:

  Predict:
    For group g with threshold t_g and deadband delta_g:
      p >= t_g + delta_g  -> toxic     (label 1)
      p <= t_g - delta_g  -> nontoxic  (label 0)
      otherwise           -> unsure    (label 2)

  Tune (on val):
    Search per-group (t_g, delta_g) jointly to minimize
        FPR_gap + FNR_gap   (over the *certain* subset)
    subject to:
        selective_F1   >= min_f1_ratio * baseline_F1   (binary @0.5 over all)
        coverage       >= min_coverage                  (default 0.7)

  Metrics:
    Selective metrics on the certain subset (predictions in {0, 1}),
    plus coverage and per-group abstention rates.

Output convention for predictions:
    0 = nontoxic, 1 = toxic, 2 = unsure / abstain
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


UNSURE = 2


def apply_thresholds_abstain(
    probs: np.ndarray, g: np.ndarray,
    t_aae: float, delta_aae: float,
    t_sae: float, delta_sae: float,
) -> np.ndarray:
    """Returns array of {0=nontoxic, 1=toxic, 2=unsure}."""
    pred = np.full(len(probs), UNSURE, dtype=int)
    for grp_val, t, d in [(1, t_aae, delta_aae), (0, t_sae, delta_sae)]:
        m = (g == grp_val)
        if not m.any():
            continue
        p = probs[m]
        toxic = p >= (t + d)
        nontox = p <= (t - d)
        local = np.full(p.shape, UNSURE, dtype=int)
        local[toxic] = 1
        local[nontox] = 0
        pred[m] = local
    return pred


def compute_metrics_abstain(y: np.ndarray, pred: np.ndarray, g: np.ndarray) -> dict:
    """Selective metrics on covered subset + abstention/coverage stats."""
    out = {}
    certain = (pred != UNSURE)
    out["coverage"] = float(certain.mean()) if len(pred) else 0.0
    out["abstain_rate"] = float(1.0 - out["coverage"])

    if certain.any():
        y_c = y[certain]
        p_c = pred[certain]
        out["sel_accuracy"] = float(accuracy_score(y_c, p_c))
        if len(np.unique(y_c)) > 1 and (p_c == 1).any():
            out["sel_f1"] = float(f1_score(y_c, p_c))
        else:
            out["sel_f1"] = float("nan")
    else:
        out["sel_accuracy"] = float("nan")
        out["sel_f1"] = float("nan")

    for grp_name, grp_val in [("AAE", 1), ("SAE", 0)]:
        gm = (g == grp_val)
        gc = gm & certain
        out[f"{grp_name}_coverage"] = float(gc.sum() / max(gm.sum(), 1))
        out[f"{grp_name}_abstain_rate"] = float(1.0 - out[f"{grp_name}_coverage"])
        if gc.any():
            yg, pg = y[gc], pred[gc]
            TP = int(((yg == 1) & (pg == 1)).sum())
            FP = int(((yg == 0) & (pg == 1)).sum())
            TN = int(((yg == 0) & (pg == 0)).sum())
            FN = int(((yg == 1) & (pg == 0)).sum())
            out[f"{grp_name}_FPR"] = float(FP / max(FP + TN, 1))
            out[f"{grp_name}_FNR"] = float(FN / max(FN + TP, 1))
            out[f"{grp_name}_TPR"] = float(TP / max(TP + FN, 1))
            out[f"{grp_name}_pos_rate"] = float((pg == 1).mean())
        else:
            for k in ["FPR", "FNR", "TPR", "pos_rate"]:
                out[f"{grp_name}_{k}"] = float("nan")

    a_fpr, s_fpr = out["AAE_FPR"], out["SAE_FPR"]
    a_fnr, s_fnr = out["AAE_FNR"], out["SAE_FNR"]
    out["FPR_gap"] = float(abs(a_fpr - s_fpr)) if not (np.isnan(a_fpr) or np.isnan(s_fpr)) else float("nan")
    out["FNR_gap"] = float(abs(a_fnr - s_fnr)) if not (np.isnan(a_fnr) or np.isnan(s_fnr)) else float("nan")
    out["abstain_gap"] = float(abs(out["AAE_abstain_rate"] - out["SAE_abstain_rate"]))

    if not (np.isnan(a_fpr) or np.isnan(s_fpr) or np.isnan(a_fnr) or np.isnan(s_fnr)):
        out["mean_FPR"] = 0.5 * (a_fpr + s_fpr)
        out["mean_FNR"] = 0.5 * (a_fnr + s_fnr)
        out["abs_balance"] = float(abs(out["mean_FPR"] - out["mean_FNR"]))
    else:
        out["mean_FPR"] = float("nan")
        out["mean_FNR"] = float("nan")
        out["abs_balance"] = float("nan")
    return out


def tune_thresholds_with_abstain(
    probs: np.ndarray, y: np.ndarray, g: np.ndarray,
    t_grid_size: int = 21,
    delta_grid_size: int = 11,
    delta_max: float = 0.30,
    min_f1_ratio: float = 0.95,
    min_coverage: float = 0.70,
    coord_descent_passes: int = 2,
) -> Tuple[float, float, float, float, dict]:
    """Joint per-group (threshold, deadband) search via coordinate descent.

    Strategy:
      Hold (t_sae, delta_sae) fixed -> grid-search (t_aae, delta_aae).
      Hold (t_aae, delta_aae) fixed -> grid-search (t_sae, delta_sae).
      Repeat.

    Returns (t_aae, delta_aae, t_sae, delta_sae, metrics).
    """
    baseline_pred = (probs >= 0.5).astype(int)
    baseline_f1 = f1_score(y, baseline_pred) if len(np.unique(y)) > 1 else 1.0
    min_f1 = baseline_f1 * min_f1_ratio

    t_grid = np.linspace(0.10, 0.90, t_grid_size)
    d_grid = np.linspace(0.0, delta_max, delta_grid_size)

    t_aae, t_sae = 0.5, 0.5
    d_aae, d_sae = 0.0, 0.0

    def evaluate(ta, da, ts, ds):
        pred = apply_thresholds_abstain(probs, g, ta, da, ts, ds)
        m = compute_metrics_abstain(y, pred, g)
        return pred, m

    def search_group(grp: str, ta_fix, da_fix, ts_fix, ds_fix):
        best = None
        best_score = float("inf")
        for t in t_grid:
            for d in d_grid:
                if grp == "AAE":
                    pred, m = evaluate(t, d, ts_fix, ds_fix)
                else:
                    pred, m = evaluate(ta_fix, da_fix, t, d)
                if np.isnan(m["sel_f1"]) or m["sel_f1"] < min_f1:
                    continue
                if m["coverage"] < min_coverage:
                    continue
                if np.isnan(m["FPR_gap"]) or np.isnan(m["FNR_gap"]):
                    continue
                score = (m["FPR_gap"] + m["FNR_gap"]
                         + 0.5 * m["abs_balance"]
                         + 0.3 * m["abstain_gap"])
                if score < best_score:
                    best_score = score
                    best = (float(t), float(d), m)
        return best

    metrics_final = None
    for _ in range(coord_descent_passes):
        cand = search_group("AAE", t_aae, d_aae, t_sae, d_sae)
        if cand is not None:
            t_aae, d_aae, metrics_final = cand
        cand = search_group("SAE", t_aae, d_aae, t_sae, d_sae)
        if cand is not None:
            t_sae, d_sae, metrics_final = cand

    if metrics_final is None:
        _, metrics_final = evaluate(0.5, 0.0, 0.5, 0.0)

    return t_aae, d_aae, t_sae, d_sae, metrics_final
