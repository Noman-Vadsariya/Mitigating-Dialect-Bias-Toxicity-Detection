import pandas as pd

# Load predictions
xgb = pd.read_csv("../data/results/xgb_predictions.csv")
toxic = pd.read_csv("../data/results/toxicbert_predictions.csv")
bert = pd.read_csv("../data/results/bert_finetuned_predictions.csv")

# Merge
df = xgb.copy()

df["toxicbert_pred"] = toxic["toxicbert_pred"]
df["bert_pred"] = bert["bert_pred"]

# Save
df.to_csv("../data/results/all_predictions.csv", index=False)

print("Merged predictions saved!")