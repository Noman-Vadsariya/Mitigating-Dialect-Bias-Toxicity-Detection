"""Ternary classification helpers (copy of binary infrastructure, adapted).

Keeps the binary `train_adv_xgb.py` untouched. Provides:
- load_split (3-class labels)
- softmax
- compute_group_weights (re-exported from binary)
- tune_logit_offsets: ternary analog of tune_thresholds
- apply_logit_offsets: ternary analog of apply_thresholds
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train_adv_xgb import compute_group_weights              


NUM_CLASSES = 3


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def load_split(csv_path: str, emb_path: str, dialect_col: str):
    df = pd.read_csv(csv_path)
    X = np.load(emb_path)

    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {csv_path}")
    if dialect_col not in df.columns:
        raise ValueError(f"Missing '{dialect_col}' column in {csv_path}")
    if len(df) != len(X):
        raise ValueError(f"Row mismatch: {csv_path} has {len(df)} rows, embeddings have {len(X)} rows")

    y = df["label"].astype(int).values
    if not np.all((y >= 0) & (y < NUM_CLASSES)):
        raise ValueError(f"Labels in {csv_path} outside [0,{NUM_CLASSES-1}]: {np.unique(y)}")
    g = df[dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    return df, X, y, g


def apply_logit_offsets(margins: np.ndarray, g: np.ndarray,
                        delta_aae: np.ndarray, delta_sae: np.ndarray) -> np.ndarray:
    """margins: (N, K). delta_*: (K,). g: (N,) with 1=AAE, 0=SAE. Returns argmax preds."""
    adj = margins.copy().astype(np.float64)
    aae_mask = (g == 1)
    sae_mask = ~aae_mask
    adj[aae_mask] += delta_aae[None, :]
    adj[sae_mask] += delta_sae[None, :]
    return np.argmax(adj, axis=1)


def _per_class_per_group_rates(y: np.ndarray, pred: np.ndarray, g: np.ndarray):
    """Return dict {(group, class): (FPR_c, FNR_c)} computed one-vs-rest per class."""
    out = {}
    for grp_id, grp_name in [(1, "AAE"), (0, "SAE")]:
        mask = (g == grp_id)
        if mask.sum() == 0:
            for c in range(NUM_CLASSES):
                out[(grp_name, c)] = (0.0, 0.0)
            continue
        y_g = y[mask]
        p_g = pred[mask]
        for c in range(NUM_CLASSES):
            y_c = (y_g == c).astype(int)
            p_c = (p_g == c).astype(int)
            TP = int(((y_c == 1) & (p_c == 1)).sum())
            FP = int(((y_c == 0) & (p_c == 1)).sum())
            TN = int(((y_c == 0) & (p_c == 0)).sum())
            FN = int(((y_c == 1) & (p_c == 0)).sum())
            FPR = FP / (FP + TN + 1e-8)
            FNR = FN / (FN + TP + 1e-8)
            out[(grp_name, c)] = (FPR, FNR)
    return out


def _mean_class_gaps(y: np.ndarray, pred: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    rates = _per_class_per_group_rates(y, pred, g)
    fpr_gaps, fnr_gaps = [], []
    for c in range(NUM_CLASSES):
        f_a, n_a = rates[("AAE", c)]
        f_s, n_s = rates[("SAE", c)]
        fpr_gaps.append(abs(f_a - f_s))
        fnr_gaps.append(abs(n_a - n_s))
    return float(np.mean(fpr_gaps)), float(np.mean(fnr_gaps))


def tune_logit_offsets(
    margins: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    dialect_col: str = "dialect_strict",
    grid_size: int = 7,
    offset_range: float = 1.5,
    min_f1_ratio: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Search per-group additive logit offsets minimizing mean(class FPR_gap + FNR_gap)
    on the validation set, subject to macro_F1 >= min_f1_ratio * baseline_macro_F1.

    Searches a per-group, per-class offset grid jointly. To keep the search tractable,
    we use a coordinate descent: fix SAE at zero, sweep AAE per-class offsets;
    then fix AAE at best, sweep SAE per-class offsets; one more pass.
    """
    baseline_pred = np.argmax(margins, axis=1)
    baseline_f1 = f1_score(y, baseline_pred, average="macro")
    f1_floor = min_f1_ratio * baseline_f1

    grid = np.linspace(-offset_range, offset_range, grid_size)
    delta_aae = np.zeros(NUM_CLASSES)
    delta_sae = np.zeros(NUM_CLASSES)

    def score(da, ds):
        pred = apply_logit_offsets(margins, g, da, ds)
        f1 = f1_score(y, pred, average="macro")
        if f1 < f1_floor:
            return float("inf"), f1, pred
        fpr_gap, fnr_gap = _mean_class_gaps(y, pred, g)
        return fpr_gap + fnr_gap, f1, pred

    best_obj, best_f1, _ = score(delta_aae, delta_sae)

    for _ in range(2):  # two coordinate-descent passes
        for grp in ("aae", "sae"):
            for c in range(NUM_CLASSES):
                cur = delta_aae if grp == "aae" else delta_sae
                local_best = (best_obj, cur[c])
                for v in grid:
                    cur[c] = v
                    obj, f1, _ = score(delta_aae, delta_sae)
                    if obj < local_best[0]:
                        local_best = (obj, v)
                cur[c] = local_best[1]
                best_obj = local_best[0]

    final_pred = apply_logit_offsets(margins, g, delta_aae, delta_sae)
    final_f1 = f1_score(y, final_pred, average="macro")
    info = {
        "baseline_macro_f1": float(baseline_f1),
        "f1_floor": float(f1_floor),
        "tuned_macro_f1": float(final_f1),
        "best_obj": float(best_obj) if np.isfinite(best_obj) else float("nan"),
        "min_f1_ratio": float(min_f1_ratio),
    }
    return delta_aae, delta_sae, info
