"""Ternary EO adversarial XGBoost (copy of binary EO, adapted to multiclass softmax).

Adversary input: [softmax_probs (K), y_onehot (K)]  -> predicts group g (binary).
We use a binary logistic regression for the adversary (group AAE vs SAE).
Adversary gradient w.r.t. each margin column is computed via softmax Jacobian
chain rule: d adv_loss / d margin_k = sum_j (d adv_loss / d p_j) * (d p_j / d margin_k).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train_adv_xgb_ternary import softmax, NUM_CLASSES              
from train_adv_xgb import compute_group_weights              


def _adversary_features(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """probs: (N,K) softmax probs.  y: (N,) ints in [0,K).  Returns (N, 2K)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return np.concatenate([probs.astype(np.float64), onehot], axis=1)


def fit_eo_adversary(
    probs: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    adv_c: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[LogisticRegression]:
    if len(np.unique(g)) < 2:
        return None
    X = _adversary_features(probs, y)
    clf = LogisticRegression(C=adv_c, solver="lbfgs", max_iter=1000)
    clf.fit(X, g, sample_weight=sample_weight)
    return clf


def adversary_grad_hess(
    clf: LogisticRegression,
    margins: np.ndarray,         # (N, K)
    y: np.ndarray,               # (N,)
    g: np.ndarray,               # (N,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (grad, hess) each with shape (N, K).
    Loss = -log p(g | features).  Features depend on margins through softmax(margins).
    Adversary is binary logistic; let q = sigmoid(w·feat + b) = P(g=1).
    dL/d feat_j = (q - g) * w_j  for j in [0, K) (the prob features).
                  the y-onehot features have zero margin-gradient.
    Then df/dmargin_k via softmax Jacobian: dp_j/dmargin_k = p_j*(delta_jk - p_k).
    """
    w = clf.coef_[0]
    b = clf.intercept_[0]
    K = NUM_CLASSES

    P = softmax(margins, axis=1)               # (N, K)
    feat = _adversary_features(P, y)           # (N, 2K)
    z = feat @ w + b
    # numerically stable sigmoid
    q = 1.0 / (1.0 + np.exp(-z))               # (N,)
    g_f = g.astype(np.float64)

    # gradient of binary CE w.r.t. probability features only (first K of w)
    w_p = w[:K]                                 # (K,)
    coef = (q - g_f)                            # (N,)

    # chain through softmax Jacobian:
    # grad_margin_k = sum_j coef * w_p[j] * P[:,j] * (delta_jk - P[:,k])
    #               = coef * (w_p[k]*P[:,k] - P[:,k] * sum_j w_p[j]*P[:,j])
    wpP_sum = (P * w_p[None, :]).sum(axis=1)    # (N,)
    grad = coef[:, None] * P * (w_p[None, :] - wpP_sum[:, None])   # (N, K)

    # diagonal Hessian approximation: d2L/dz2 = q*(1-q); chain rule with
    # |dz/dmargin_k|^2.  Use diagonal-only approx for stability.
    qq = q * (1.0 - q)                          # (N,)
    dzdm = P * (w_p[None, :] - wpP_sum[:, None])  # (N, K), == grad / coef
    hess = qq[:, None] * (dzdm ** 2)
    return grad, hess


def _project_out(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project tensor u onto the orthogonal complement of v, treating both as
    flat vectors over (N, K)."""
    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)
    norm_sq = float(np.dot(v_flat, v_flat))
    if norm_sq < eps:
        return u
    coef = float(np.dot(u_flat, v_flat)) / norm_sq
    return (u_flat - coef * v_flat).reshape(u.shape)


def _softmax_ce_grad_hess(margins: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = softmax(margins, axis=1)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    grad = (P - Y)                       # (N, K)
    hess = P * (1.0 - P)                 # diagonal approx, (N, K)
    return grad, hess


def train_one_adv_model_eo_ternary(
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
    """XGBoost ternary classifier with EO adversary.

    Uses multi:softprob output. XGBoost passes ``preds`` to the custom obj as
    a flat array of length N*K (row-major); we reshape to (N, K). The returned
    grad/hess must be flat in the same layout.
    """
    K = NUM_CLASSES
    sample_weight = compute_group_weights(g_train) if use_reweighting else None
    params = {**params, "objective": "multi:softprob", "num_class": K,
              "disable_default_eval_metric": 1}

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)

    booster: Optional[xgb.Booster] = None
    adv_clf: Optional[LogisticRegression] = None
    history = {"adv_acc_train": []}

    N_tr = X_train.shape[0]

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True).reshape(N_tr, K)
            P_tr = softmax(margin_tr, axis=1)
            adv_clf = fit_eo_adversary(P_tr, y_train, g_train,
                                       adv_c=adv_c, sample_weight=sample_weight)

        def objective(preds, dmat, _adv_clf=adv_clf):
            y = dmat.get_label().astype(int)
            n = y.shape[0]
            margins = np.asarray(preds).reshape(n, K)

            grad_tox, hess_tox = _softmax_ce_grad_hess(margins, y)

            if _adv_clf is None or lambda_adv == 0.0:
                hess = np.clip(hess_tox, 1e-6, None)
                return grad_tox.reshape(-1), hess.reshape(-1)

            grad_adv, hess_adv = adversary_grad_hess(_adv_clf, margins, y, g_train)

            if use_projection:
                grad_pred = _project_out(grad_tox, grad_adv)
            else:
                grad_pred = grad_tox

            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            hess = np.clip(hess, 1e-6, None)
            return grad.reshape(-1), hess.reshape(-1)

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
                m = booster.predict(dtrain, output_margin=True).reshape(N_tr, K)
                P = softmax(m, axis=1)
                Xa = _adversary_features(P, y_train)
                history["adv_acc_train"].append(float(adv_clf.score(Xa, g_train)))
            except Exception:
                pass

    return booster, history
