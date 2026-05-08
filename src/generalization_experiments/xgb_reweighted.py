from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# CONFIG
# =========================
TEST_CSV = Path("../../data/processed/hateXplain/test.csv")
RESULTS_DIR = Path("../../data/results/generalization_eval/xgb_reweighted")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "tweet"
LABEL_COL = "label"
DIALECT_COL = "dialect_strict"  # expected values: AAE / SAE

# For XGBoost models, provide embeddings
TEST_EMB = Path("../../data/embeddings/hateXplain/test_emb.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LEN = 128

# Update these paths to your actual saved checkpoints
MODEL_SPECS = [
    {
        "name": "2.25",
        "type": "xgb",
        "path": "../../models/xgb_reweighted_alpha_2_25.joblib",
        "input": "emb",
    },
    {
        "name": "2.5",
        "type": "xgb",
        "path": "../../models/xgb_reweighted_alpha_2_5.joblib",
        "input": "emb",
    },
    {
        "name": "5.75",
        "type": "xgb",
        "path": "../../models/xgb_reweighted_alpha_5_75.joblib",
        "input": "emb",
    },
    {
        "name": "6.75",
        "type": "xgb",
        "path": "../../models/xgb_reweighted_alpha_6_75.joblib",
        "input": "emb",
    },
]


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(TEST_CSV)
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[LABEL_COL] = df[LABEL_COL].astype(int)
df[DIALECT_COL] = df[DIALECT_COL].astype(str)

y_true = df[LABEL_COL].values.astype(int)
dialect = df[DIALECT_COL].map({"AAE": 1, "SAE": 0}).values.astype(int)

texts = df[TEXT_COL].tolist()
X_test_emb = np.load(TEST_EMB)


# =========================
# HELPERS
# =========================
def pct(x):
    return x * 100.0


def compute_fpr_fnr_binary(
    y_true: np.ndarray, y_pred: np.ndarray, dialect: np.ndarray
) -> Dict[str, float]:
    """
    Binary fairness metrics:
    label 0 = non-toxic
    label 1 = toxic
    """
    out: Dict[str, float] = {}

    for gname, gval in [("AAE", 1), ("SAE", 0)]:
        idx = dialect == gval
        yt = y_true[idx]
        yp = y_pred[idx]

        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))
        fn = np.sum((yt == 1) & (yp == 0))
        tp = np.sum((yt == 1) & (yp == 1))

        out[f"FPR_{gname}"] = fp / (fp + tn + 1e-8)
        out[f"FNR_{gname}"] = fn / (fn + tp + 1e-8)

    out["FPR_Gap"] = abs(out["FPR_AAE"] - out["FPR_SAE"])
    out["FNR_Gap"] = abs(out["FNR_AAE"] - out["FNR_SAE"])
    return out


def predict_hf(model_path: str, texts: List[str]) -> np.ndarray:
    """
    Predict binary classes from a Hugging Face text checkpoint.
    Works for binary sequence classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    preds = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits

            # Binary can be either one logit or two logits depending on how it was trained
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                batch_pred = (probs >= 0.5).astype(int)
            else:
                batch_pred = torch.argmax(logits, dim=-1).cpu().numpy()

        preds.extend(batch_pred.tolist())

    return np.array(preds, dtype=int)


def predict_xgb(model_path: str, X: np.ndarray) -> np.ndarray:
    model = joblib.load(model_path)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

        # binary case usually returns shape [n_samples, 2]
        if probs.ndim == 2 and probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)

        # fallback
        return np.argmax(probs, axis=1).astype(int)

    return model.predict(X).astype(int)


# =========================
# EVALUATE ALL MODELS
# =========================
rows: List[Dict[str, Any]] = []

for spec in MODEL_SPECS:
    print(f"Evaluating {spec['name']} ...")

    if spec["type"] == "hf":
        y_pred = predict_hf(spec["path"], texts)

    elif spec["type"] == "xgb":
        y_pred = predict_xgb(spec["path"], X_test_emb)

    else:
        raise ValueError(f"Unknown model type: {spec['type']}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fairness = compute_fpr_fnr_binary(y_true, y_pred, dialect)

    row = {
        "Model": spec["name"],
        "Accuracy": acc,
        "F1": f1,
        **fairness,
    }
    rows.append(row)

    print(row)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

print("\nSaved metrics ->", RESULTS_DIR / "metrics.csv")
print(metrics_df)


# =========================
# PLOTTING
# =========================
def plot_accuracy(dfm):
    x = np.arange(len(dfm))
    vals = pct(dfm["Accuracy"])

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals, width=0.5)

    plt.xticks(x, dfm["Model"])
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Across Models")

    for i in range(len(dfm)):
        plt.text(i, vals.iloc[i], f"{vals.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy.png", dpi=300)
    plt.close()


def plot_F1(dfm):
    x = np.arange(len(dfm))
    vals = pct(dfm["F1"])

    plt.figure(figsize=(8, 5))
    plt.bar(x, vals, width=0.5)

    plt.xticks(x, dfm["Model"])
    plt.ylabel("F1 Score (%)")
    plt.title("F1 Score Across Models")

    for i in range(len(dfm)):
        plt.text(i, vals.iloc[i], f"{vals.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1.png", dpi=300)
    plt.close()


def plot_FPR(dfm):
    x = np.arange(len(dfm))
    width = 0.25

    plt.figure(figsize=(8, 5))

    aae = pct(dfm["FPR_AAE"])
    sae = pct(dfm["FPR_SAE"])
    gap = pct(dfm["FPR_Gap"])

    plt.bar(x - width, aae, width=width, label="AAE")
    plt.bar(x, sae, width=width, label="SAE")
    plt.bar(x + width, gap, width=width, label="Gap")

    plt.xticks(x, dfm["Model"])
    plt.ylabel("False Positive Rate (%)")
    plt.title("FPR Across Models")
    plt.legend()

    for i in range(len(dfm)):
        plt.text(
            i - width, aae.iloc[i], f"{aae.iloc[i]:.1f}%", ha="center", va="bottom"
        )
        plt.text(i, sae.iloc[i], f"{sae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(
            i + width, gap.iloc[i], f"{gap.iloc[i]:.1f}%", ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fpr.png", dpi=300)
    plt.close()


def plot_FNR(dfm):
    x = np.arange(len(dfm))
    width = 0.25

    plt.figure(figsize=(8, 5))

    aae = pct(dfm["FNR_AAE"])
    sae = pct(dfm["FNR_SAE"])
    gap = pct(dfm["FNR_Gap"])

    plt.bar(x - width, aae, width=width, label="AAE")
    plt.bar(x, sae, width=width, label="SAE")
    plt.bar(x + width, gap, width=width, label="Gap")

    plt.xticks(x, dfm["Model"])
    plt.ylabel("False Negative Rate (%)")
    plt.title("FNR Across Models")
    plt.legend()

    for i in range(len(dfm)):
        plt.text(
            i - width, aae.iloc[i], f"{aae.iloc[i]:.1f}%", ha="center", va="bottom"
        )
        plt.text(i, sae.iloc[i], f"{sae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(
            i + width, gap.iloc[i], f"{gap.iloc[i]:.1f}%", ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fnr.png", dpi=300)
    plt.close()


# =========================
# RUN ALL PLOTS
# =========================
plot_accuracy(metrics_df)
plot_F1(metrics_df)
plot_FPR(metrics_df)
plot_FNR(metrics_df)
