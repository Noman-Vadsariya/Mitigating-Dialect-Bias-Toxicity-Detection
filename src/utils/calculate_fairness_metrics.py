import pandas as pd
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/results/all_predictions.csv")

DIALECT_COL = "dialect_strict"

models = {
    "XGBoost": "xgb_pred",
    "ToxicBERT": "toxicbert_pred",
    "BERT": "bert_pred",
    "VS": "vs_pred"
}

# =========================
# METRIC FUNCTIONS
# =========================

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
    fpr_aae = compute_FPR(df, col, "AAE")
    fpr_sae = compute_FPR(df, col, "SAE")

    fnr_aae = compute_FNR(df, col, "AAE")
    fnr_sae = compute_FNR(df, col, "SAE")

    DIfav, DIunfav = compute_DI(df, col)

    row = {
        "Model": model_name,
        "FPR_AAE": fpr_aae,
        "FPR_SAE": fpr_sae,
        "FPR_Gap": abs(fpr_aae - fpr_sae),

        "FNR_AAE": fnr_aae,
        "FNR_SAE": fnr_sae,
        "FNR_Gap": abs(fnr_aae - fnr_sae),

        "DIfav": DIfav,
        "DIunfav": DIunfav
    }

    rows.append(row)

metrics_df = pd.DataFrame(rows)

# =========================
# SAVE RESULTS
# =========================
os.makedirs("../data/results", exist_ok=True)

metrics_df.to_csv("../data/results/metrics.csv", index=False)

# Save readable text file
with open("../data/results/metrics.txt", "w") as f:
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

print("Metrics saved to ../data/results/metrics.csv and ../data/results/metrics.txt")