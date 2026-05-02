import pandas as pd
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../../data/results/all_predictions.csv")

DIALECT_COL = "dialect_strict"

models = {
    "XGBoost": "xgb_pred",
    "ToxicBERT": "toxicbert_pred",
    "BERT": "bert_pred",
    # "VS": "vs_pred"
}

# =========================
# METRIC FUNCTIONS
# =========================

def compute_F1(df, pred_col):
    TP = ((df["label"] == 1) & (df[pred_col] == 1)).sum()
    FP = ((df["label"] == 0) & (df[pred_col] == 1)).sum()
    FN = ((df["label"] == 1) & (df[pred_col] == 0)).sum()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

def compute_F1_per_class(df, pred_col):
    # Toxic class (label = 1)
    TP = ((df["label"] == 1) & (df[pred_col] == 1)).sum()
    FP = ((df["label"] == 0) & (df[pred_col] == 1)).sum()
    FN = ((df["label"] == 1) & (df[pred_col] == 0)).sum()

    precision_tox = TP / (TP + FP + 1e-8)
    recall_tox = TP / (TP + FN + 1e-8)
    f1_tox = 2 * (precision_tox * recall_tox) / (precision_tox + recall_tox + 1e-8)

    # Non-toxic class (label = 0)
    TN = ((df["label"] == 0) & (df[pred_col] == 0)).sum()
    FN_nt = FP   # predicted toxic but actually non-toxic
    FP_nt = FN   # predicted non-toxic but actually toxic

    precision_nt = TN / (TN + FN_nt + 1e-8)
    recall_nt = TN / (TN + FP_nt + 1e-8)
    f1_nt = 2 * (precision_nt * recall_nt) / (precision_nt + recall_nt + 1e-8)

    return f1_tox, f1_nt

def compute_FPR(df, pred_col, group):
    subset = df[df[DIALECT_COL] == group]
    FP = ((subset["label"] == 0) & (subset[pred_col] == 1)).sum()
    TN = ((subset["label"] == 0) & (subset[pred_col] == 0)).sum()
    return FP / (FP + TN + 1e-8)


def compute_FNR(df, pred_col, group):
    subset = df[df[DIALECT_COL] == group]
    TP = ((subset["label"] == 1) & (subset[pred_col] == 1)).sum()
    FN = ((subset["label"] == 1) & (subset[pred_col] == 0)).sum()
    return FN / (FN + TP + 1e-8)


def compute_DI(df, pred_col):
    aae = df[df[DIALECT_COL] == "AAE"]
    sae = df[df[DIALECT_COL] == "SAE"]

    p_aae_non = (aae[pred_col] == 0).mean()
    p_sae_non = (sae[pred_col] == 0).mean()

    p_aae_tox = (aae[pred_col] == 1).mean()
    p_sae_tox = (sae[pred_col] == 1).mean()

    DIfav = p_aae_non / (p_sae_non + 1e-8)
    DIunfav = p_aae_tox / (p_sae_tox + 1e-8)

    return DIfav, DIunfav


# =========================
# COMPUTE METRICS
# =========================

rows = []

for model_name, col in models.items():
    compute_F1_score = compute_F1(df, col)
    f1_tox, f1_nt = compute_F1_per_class(df, col)

    fpr_aae = compute_FPR(df, col, "AAE")
    fpr_sae = compute_FPR(df, col, "SAE")

    fnr_aae = compute_FNR(df, col, "AAE")
    fnr_sae = compute_FNR(df, col, "SAE")

    DIfav, DIunfav = compute_DI(df, col)

    row = {

        "Model": model_name,
        "F1": compute_F1_score,
        "F1_Toxic": f1_tox,
        "F1_NonToxic": f1_nt,

        "FPR_AAE": fpr_aae,
        "FPR_SAE": fpr_sae,
        "FPR_Gap": abs(fpr_aae - fpr_sae),

        "FNR_AAE": fnr_aae,
        "FNR_SAE": fnr_sae,
        "FNR_Gap": abs(fnr_aae - fnr_sae),

        "DIfav": DIfav,
        "DIunfav": DIunfav,
    }

    rows.append(row)

metrics_df = pd.DataFrame(rows)

# =========================
# SAVE RESULTS
# =========================
# os.makedirs("../../data/results", exist_ok=True)

metrics_df.to_csv("../../data/results/metrics.csv", index=False)

# Save readable text file
with open("../../data/results/metrics.txt", "w") as f:
    for _, row in metrics_df.iterrows():
        f.write(f"\n===== {row['Model']} =====\n")

        f.write(f"FPR AAE: {row['FPR_AAE']:.4f}\n")
        f.write(f"FPR SAE: {row['FPR_SAE']:.4f}\n")
        f.write(f"FPR Gap: {row['FPR_Gap']:.4f}\n")

        f.write(f"FNR AAE: {row['FNR_AAE']:.4f}\n")
        f.write(f"FNR SAE: {row['FNR_SAE']:.4f}\n")
        f.write(f"FNR Gap: {row['FNR_Gap']:.4f}\n")

        f.write(f"DIfav: {row['DIfav']:.4f}\n")
        f.write(f"DIunfav: {row['DIunfav']:.4f}\n")

print("Metrics saved to ../../data/results/metrics.csv and ../../data/results/metrics.txt")