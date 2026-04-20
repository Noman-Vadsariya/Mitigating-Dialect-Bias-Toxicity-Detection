from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train_adv_xgb import (              
    load_split,
    sigmoid,
    compute_group_weights,
    tune_thresholds,
    apply_thresholds,
)


def _adversary_features(margin: np.ndarray, y: np.ndarray) -> np.ndarray:
    margin = margin.astype(np.float64)
    y = y.astype(np.float64)
    return np.stack([margin, y, margin * y], axis=1)


def fit_eo_adversary(
    margin: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    adv_c: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[LogisticRegression]:
    if len(np.unique(g)) < 2:
        return None
    X = _adversary_features(margin, y)
    clf = LogisticRegression(
        C=adv_c, solver="lbfgs", max_iter=1000, warm_start=True,
    )
    clf.fit(X, g, sample_weight=sample_weight)
    return clf


def adversary_grad_hess(
    clf: LogisticRegression,
    margin: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w = clf.coef_[0]
    b = clf.intercept_[0]
    w_f, w_y, w_fy = float(w[0]), float(w[1]), float(w[2])

    z = w_f * margin + w_y * y + w_fy * (margin * y) + b
    p = sigmoid(z)
    dzdm = w_f + w_fy * y.astype(np.float64)
    grad = (p - g.astype(np.float64)) * dzdm
    hess = p * (1.0 - p) * (dzdm ** 2)
    return grad, hess


def _project_out(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm_sq = float(np.dot(v, v))
    if norm_sq < eps:
        return u
    coef = float(np.dot(u, v)) / norm_sq
    return u - coef * v


def train_one_adv_model_eo(
    X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, g_val: np.ndarray,
    params: dict,
    lambda_adv: float = 0.1,
    adv_c: float = 1.0,
    num_round: int = 100,
    use_reweighting: bool = False,
    use_projection: bool = True,
    warmup_rounds: int = 5,
) -> Tuple[xgb.Booster, dict]:
    sample_weight = compute_group_weights(g_train) if use_reweighting else None
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster: Optional[xgb.Booster] = None
    adv_clf: Optional[LogisticRegression] = None

    history = {"adv_acc_train": [], "grad_align_cos": [], "val_f1": []}

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True)
            adv_clf = fit_eo_adversary(
                margin_tr, y_train, g_train,
                adv_c=adv_c, sample_weight=sample_weight,
            )

        def objective(preds, dmat, _adv_clf=adv_clf):
            y = dmat.get_label()
            p_tox = sigmoid(preds)
            grad_tox = p_tox - y
            hess_tox = p_tox * (1.0 - p_tox)

            if _adv_clf is None or lambda_adv == 0.0:
                hess = np.clip(hess_tox, 1e-6, None)
                return grad_tox, hess

            grad_adv, hess_adv = adversary_grad_hess(_adv_clf, preds, y, g_train)

            if use_projection:
                grad_pred = _project_out(grad_tox, grad_adv)
            else:
                grad_pred = grad_tox

            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            hess = np.clip(hess, 1e-6, None)
            return grad, hess

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            obj=objective,
            xgb_model=booster,
            verbose_eval=False,
        )

        if adv_clf is not None:
            try:
                mtr = booster.predict(dtrain, output_margin=True)
                X_adv = _adversary_features(mtr, y_train)
                history["adv_acc_train"].append(float(adv_clf.score(X_adv, g_train)))
            except Exception:
                pass

    return booster, history
