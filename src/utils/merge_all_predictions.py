import os

import pandas as pd

# Load predictions
# xgb = pd.read_csv("../data/results/xgb_predictions.csv")
# toxic = pd.read_csv("../data/results/toxicbert_predictions.csv")
# bert = pd.read_csv("../data/results/bert_finetuned_predictions.csv")

xgb = pd.read_csv("../../data/results/xgb_predictions.csv")
toxic = pd.read_csv("../../data/results/toxicbert_predictions.csv")
bert = pd.read_csv("../../data/results/bert_finetuned_predictions.csv")

# Merge
df = xgb.copy()

df["toxicbert_pred"] = toxic["toxicbert_pred"]
df["bert_pred"] = bert["bert_pred"]

file_path = "../data/results/vs_xgb_train_time/vs_xgb_predictions.csv"
if os.path.exists(file_path):
    vs = pd.read_csv(file_path)
    # Add VS predictions
    df["vs_pred"] = vs["vs_pred"]


# Save
df.to_csv("../../data/results/all_predictions.csv", index=False)

print("Merged predictions saved!")