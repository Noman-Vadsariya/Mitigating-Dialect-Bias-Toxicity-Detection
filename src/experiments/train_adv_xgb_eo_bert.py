"""Adversarial XGBoost (EO) with a BERT-features-based MLP adversary.

Drop-in alternative to ``train_adv_xgb_eo.py``. The original module uses a
sklearn LogisticRegression as the EO adversary on ``[margin, y, margin*y]``.
This module replaces that adversary with an MLP that consumes the precomputed
BERT embedding (768-d) PLUS the same ``[margin, y, margin*y]`` triple, giving
the adversary much more capacity to detect group leakage in the predictor
margin while remaining EO (still conditioned on ``y``).

The XGBoost predictor side and the closed-form objective contract are
identical to ``train_adv_xgb_eo.py``: per-sample (grad, hess) of the adversary
log-loss with respect to the predictor margin. Gradients are obtained via
torch autograd; the Hessian is approximated by the BCE curvature
``p*(1-p)`` (the chain-rule factor through the MLP is folded into the step
size on the predictor side, which only needs a positive Hessian).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from train_adv_xgb import (  # noqa: E402
    compute_group_weights,
    sigmoid,
)
from train_adv_xgb_eo import _project_out  # noqa: E402


def _device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BertMLPAdversary(nn.Module):
    """MLP that predicts group from [BERT_emb, margin, y, margin*y]."""

    def __init__(self, emb_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, emb: torch.Tensor, margin: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([margin, y, margin * y], dim=1)
        x = torch.cat([emb, feats], dim=1)
        return self.net(x).squeeze(-1)


class BertAdversaryState:
    """Persistent state across boosting rounds (warm-started)."""

    def __init__(self, emb: np.ndarray, hidden: int, dropout: float,
                 lr: float, weight_decay: float, device: torch.device):
        self.device = device
        self.emb_t = torch.as_tensor(emb, dtype=torch.float32, device=device)
        self.model = BertMLPAdversary(
            emb_dim=emb.shape[1], hidden=hidden, dropout=dropout,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

    def fit(self, margin: np.ndarray, y: np.ndarray, g: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            epochs: int = 3, batch_size: int = 1024) -> None:
        if len(np.unique(g)) < 2:
            return
        device = self.device
        margin_t = torch.as_tensor(margin, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        g_t = torch.as_tensor(g, dtype=torch.float32, device=device)
        if sample_weight is not None:
            w_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
        else:
            w_t = None
        n = self.emb_t.shape[0]
        self.model.train()
        for _ in range(epochs):
            idx = torch.randperm(n, device=device)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                logit = self.model(self.emb_t[b], margin_t[b], y_t[b])
                loss = F.binary_cross_entropy_with_logits(
                    logit, g_t[b], reduction="none",
                )
                loss = (loss * w_t[b]).mean() if w_t is not None else loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def grad_hess(self, margin: np.ndarray, y: np.ndarray, g: np.ndarray,
                  batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        device = self.device
        n = self.emb_t.shape[0]
        grad_out = np.zeros(n, dtype=np.float64)
        hess_out = np.zeros(n, dtype=np.float64)
        margin_t_full = torch.as_tensor(margin, dtype=torch.float32, device=device)
        y_t_full = torch.as_tensor(y, dtype=torch.float32, device=device)
        g_t_full = torch.as_tensor(g, dtype=torch.float32, device=device)
        self.model.eval()
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            m = margin_t_full[s:e].clone().detach().requires_grad_(True)
            logit = self.model(self.emb_t[s:e], m, y_t_full[s:e])
            loss = F.binary_cross_entropy_with_logits(
                logit, g_t_full[s:e], reduction="sum",
            )
            gm = torch.autograd.grad(loss, m, create_graph=False)[0]
            with torch.no_grad():
                p = torch.sigmoid(logit)
                h = p * (1.0 - p)
            grad_out[s:e] = gm.detach().cpu().numpy().astype(np.float64)
            hess_out[s:e] = h.detach().cpu().numpy().astype(np.float64)
        return grad_out, hess_out


def train_one_adv_model_eo_bert(
    X_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, g_val: np.ndarray,
    params: dict,
    lambda_adv: float = 0.1,
    adv_lr: float = 1e-3,
    adv_weight_decay: float = 1e-4,
    adv_hidden: int = 256,
    adv_dropout: float = 0.2,
    adv_epochs_per_round: int = 2,
    adv_batch_size: int = 1024,
    num_round: int = 100,
    use_reweighting: bool = False,
    use_projection: bool = True,
    warmup_rounds: int = 5,
    device: str = "cuda",
) -> Tuple[xgb.Booster, dict]:
    sample_weight = compute_group_weights(g_train) if use_reweighting else None
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    _ = xgb.DMatrix(X_val, label=y_val)

    booster: Optional[xgb.Booster] = None
    adv_state: Optional[BertAdversaryState] = None
    torch_device = _device(device)

    history = {"adv_loss_train": []}

    for r in range(num_round):
        if booster is not None and r >= warmup_rounds:
            margin_tr = booster.predict(dtrain, output_margin=True)
            if adv_state is None:
                adv_state = BertAdversaryState(
                    emb=X_train, hidden=adv_hidden, dropout=adv_dropout,
                    lr=adv_lr, weight_decay=adv_weight_decay, device=torch_device,
                )
            adv_state.fit(
                margin_tr, y_train, g_train,
                sample_weight=sample_weight,
                epochs=adv_epochs_per_round,
                batch_size=adv_batch_size,
            )

        def objective(preds, dmat, _adv=adv_state):
            y = dmat.get_label()
            p_tox = sigmoid(preds)
            grad_tox = p_tox - y
            hess_tox = p_tox * (1.0 - p_tox)

            if _adv is None or lambda_adv == 0.0:
                return grad_tox, np.clip(hess_tox, 1e-6, None)

            grad_adv, hess_adv = _adv.grad_hess(preds, y, g_train)
            grad_pred = _project_out(grad_tox, grad_adv) if use_projection else grad_tox
            grad = grad_pred - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            return grad, np.clip(hess, 1e-6, None)

        booster = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=1,
            obj=objective, xgb_model=booster, verbose_eval=False,
        )

    return booster, history
