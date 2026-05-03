"""Per-class per-group fairness metrics for ternary classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


NUM_CLASSES = 3


def compute_metrics_ternary(y: np.ndarray, pred: np.ndarray, g: np.ndarray) -> dict:
    """y, pred: int arrays in [0,K). g: 1=AAE, 0=SAE.

    Returns a flat dict with:
      accuracy, f1_macro, f1_weighted, f1_class_{c},
      {AAE,SAE}_FPR_{c}, {AAE,SAE}_FNR_{c},
      FPR_gap_{c}, FNR_gap_{c},
      mean_FPR_gap, mean_FNR_gap,
      mean_FPR (macro avg of group/class FPRs), mean_FNR,
      abs_balance (|mean_FPR_AAE - mean_FPR_SAE|).
    """
    out = {}
    out["accuracy"] = float(accuracy_score(y, pred))
    out["f1_macro"] = float(f1_score(y, pred, average="macro"))
    out["f1_weighted"] = float(f1_score(y, pred, average="weighted"))
    f1_per_class = f1_score(y, pred, average=None, labels=list(range(NUM_CLASSES)))
    for c in range(NUM_CLASSES):
        out[f"f1_class_{c}"] = float(f1_per_class[c])

    group_class_fpr = {}
    group_class_fnr = {}
    for grp_id, grp_name in [(1, "AAE"), (0, "SAE")]:
        mask = (g == grp_id)
        y_g = y[mask]; p_g = pred[mask]
        for c in range(NUM_CLASSES):
            y_c = (y_g == c).astype(int)
            p_c = (p_g == c).astype(int)
            TP = int(((y_c == 1) & (p_c == 1)).sum())
            FP = int(((y_c == 0) & (p_c == 1)).sum())
            TN = int(((y_c == 0) & (p_c == 0)).sum())
            FN = int(((y_c == 1) & (p_c == 0)).sum())
            FPR = FP / (FP + TN + 1e-8)
            FNR = FN / (FN + TP + 1e-8)
            out[f"{grp_name}_FPR_{c}"] = float(FPR)
            out[f"{grp_name}_FNR_{c}"] = float(FNR)
            group_class_fpr[(grp_name, c)] = FPR
            group_class_fnr[(grp_name, c)] = FNR

    fpr_gaps, fnr_gaps = [], []
    for c in range(NUM_CLASSES):
        f_a = group_class_fpr[("AAE", c)]; f_s = group_class_fpr[("SAE", c)]
        n_a = group_class_fnr[("AAE", c)]; n_s = group_class_fnr[("SAE", c)]
        out[f"FPR_gap_{c}"] = float(abs(f_a - f_s))
        out[f"FNR_gap_{c}"] = float(abs(n_a - n_s))
        fpr_gaps.append(abs(f_a - f_s))
        fnr_gaps.append(abs(n_a - n_s))
    out["mean_FPR_gap"] = float(np.mean(fpr_gaps))
    out["mean_FNR_gap"] = float(np.mean(fnr_gaps))

    aae_fpr_mean = np.mean([group_class_fpr[("AAE", c)] for c in range(NUM_CLASSES)])
    sae_fpr_mean = np.mean([group_class_fpr[("SAE", c)] for c in range(NUM_CLASSES)])
    aae_fnr_mean = np.mean([group_class_fnr[("AAE", c)] for c in range(NUM_CLASSES)])
    sae_fnr_mean = np.mean([group_class_fnr[("SAE", c)] for c in range(NUM_CLASSES)])
    out["AAE_mean_FPR"] = float(aae_fpr_mean)
    out["SAE_mean_FPR"] = float(sae_fpr_mean)
    out["AAE_mean_FNR"] = float(aae_fnr_mean)
    out["SAE_mean_FNR"] = float(sae_fnr_mean)
    out["mean_FPR"] = float((aae_fpr_mean + sae_fpr_mean) / 2)
    out["mean_FNR"] = float((aae_fnr_mean + sae_fnr_mean) / 2)
    out["abs_balance"] = float(abs(aae_fpr_mean - sae_fpr_mean))
    return out
