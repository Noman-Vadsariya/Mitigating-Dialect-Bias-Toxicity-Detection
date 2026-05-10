import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report

# LOAD DATA
df = pd.read_csv("../../data/results/twitterAAE_baselines/unbalanced_ternary/all_predictions.csv")
# expected columns:
# label, dialect_strict, xgb_pred, bert_pred, toxicbert_pred

MODELS = {
    "XGBoost": "xgb_pred",
    "BERT": "bert_pred",
    "ToxicBERT": "toxicbert_pred"
}

# HELPERS
def compute_fpr_fnr_ternary(y_true, y_pred, dialect):
    """
    Ternary setup:
    0 = non-toxic
    1 = hate
    2 = offensive

    For fairness metrics:
    - FPR = non-toxic predicted as toxic (pred != 0)
    - FNR = toxic (1 or 2) predicted as non-toxic (pred == 0)
    """
    results = {}

    for group, name in [(1, "AAE"), (0, "SAE")]:
        idx = dialect == group
        yt = y_true[idx]
        yp = y_pred[idx]

        fp = np.sum((yt == 0) & (yp != 0))
        tn = np.sum((yt == 0) & (yp == 0))

        fn = np.sum((yt != 0) & (yp == 0))
        tp = np.sum((yt != 0) & (yp != 0))

        results[f"FPR_{name}"] = fp / (fp + tn + 1e-8)
        results[f"FNR_{name}"] = fn / (fn + tp + 1e-8)

    results["FPR_Gap"] = abs(results["FPR_AAE"] - results["FPR_SAE"])
    results["FNR_Gap"] = abs(results["FNR_AAE"] - results["FNR_SAE"])

    return results


def pct(x):
    return x * 100


# METRIC COMPUTATION
rows = []

y_true = df["label"].values.astype(int)
dialect = df["dialect_strict"].map({"AAE": 1, "SAE": 0}).values.astype(int)

for model_name, col in MODELS.items():
    y_pred = df[col].values.astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    f1_non_toxic = report["0"]["f1-score"]
    f1_hate = report["1"]["f1-score"]
    f1_offensive = report["2"]["f1-score"]

    fairness = compute_fpr_fnr_ternary(y_true, y_pred, dialect)

    row = {
        "Model": model_name,
        "Accuracy": acc,
        "F1": f1_macro,
        "F1_Weighted": f1_weighted,
        "F1_NonToxic": f1_non_toxic,
        "F1_Hate": f1_hate,
        "F1_Offensive": f1_offensive,
        **fairness
    }
    rows.append(row)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("../../data/results/metrics.csv", index=False)

print(metrics_df)

# PLOTTING
def plot_accuracy(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))
    vals = pct(df["Accuracy"])

    plt.bar(x, vals, width=0.5)
    plt.xticks(x, df["Model"])
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Across Models")

    for i in range(len(df)):
        plt.text(i, vals.iloc[i], f"{vals.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.show()


def plot_F1(df):
    x = np.arange(len(df))

    plt.figure(figsize=(8, 5))
    vals = pct(df["F1"])

    plt.bar(x, vals, width=0.5)
    plt.xticks(x, df["Model"])
    plt.ylabel("F1 Score (%)")
    plt.title("Macro F1 Score Across Models")

    for i in range(len(df)):
        plt.text(i, vals.iloc[i], f"{vals.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.show()


def plot_F1_class(df):
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(8, 5))

    non = pct(df["F1_NonToxic"])
    hate = pct(df["F1_Hate"])
    off = pct(df["F1_Offensive"])

    plt.bar(x - width, non, width=width, label="Non-Toxic")
    plt.bar(x, hate, width=width, label="Hate")
    plt.bar(x + width, off, width=width, label="Offensive")

    plt.xticks(x, df["Model"])
    plt.ylabel("F1 Score (%)")
    plt.title("Per-Class F1 Score Across Models")
    plt.legend()

    for i in range(len(df)):
        plt.text(i - width, non.iloc[i], f"{non.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i, hate.iloc[i], f"{hate.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i + width, off.iloc[i], f"{off.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.show()


def plot_FPR(df):
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(8, 5))

    aae = pct(df["FPR_AAE"])
    sae = pct(df["FPR_SAE"])
    gap = pct(df["FPR_Gap"])

    plt.bar(x - width, aae, width=width, label="AAE")
    plt.bar(x, sae, width=width, label="SAE")
    plt.bar(x + width, gap, width=width, label="Gap")

    plt.xticks(x, df["Model"])
    plt.ylabel("False Positive Rate (%)")
    plt.title("FPR Across Models")
    plt.legend()

    for i in range(len(df)):
        plt.text(i - width, aae.iloc[i], f"{aae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i, sae.iloc[i], f"{sae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i + width, gap.iloc[i], f"{gap.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.show()


def plot_FNR(df):
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(8, 5))

    aae = pct(df["FNR_AAE"])
    sae = pct(df["FNR_SAE"])
    gap = pct(df["FNR_Gap"])

    plt.bar(x - width, aae, width=width, label="AAE")
    plt.bar(x, sae, width=width, label="SAE")
    plt.bar(x + width, gap, width=width, label="Gap")

    plt.xticks(x, df["Model"])
    plt.ylabel("False Negative Rate (%)")
    plt.title("FNR Across Models")
    plt.legend()

    for i in range(len(df)):
        plt.text(i - width, aae.iloc[i], f"{aae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i, sae.iloc[i], f"{sae.iloc[i]:.1f}%", ha="center", va="bottom")
        plt.text(i + width, gap.iloc[i], f"{gap.iloc[i]:.1f}%", ha="center", va="bottom")

    plt.show()

# RUN ALL PLOTS
plot_accuracy(metrics_df)
plot_F1(metrics_df)
plot_F1_class(metrics_df)
plot_FPR(metrics_df)
plot_FNR(metrics_df)