import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# =========================
# LOAD DATA
# =========================
X_train = np.load("../data/embeddings/train_emb.npy")
X_val = np.load("../data/embeddings/val_emb.npy")
X_test = np.load("../data/embeddings/test_emb.npy")

train_df = pd.read_csv("../data/processed/train.csv")
val_df = pd.read_csv("../data/processed/val.csv")
test_df = pd.read_csv("../data/processed/test.csv")

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# =========================
# REGULARIZATION GRID
# =========================
configs = [
    {"lambda": 0.1, "alpha": 0},
    {"lambda": 0.5, "alpha": 0},
    {"lambda": 0.6, "alpha": 0},
    {"lambda": 0.7, "alpha": 0},
    {"lambda": 0.8, "alpha": 0},
    {"lambda": 0.9, "alpha": 0},
    {"lambda": 1, "alpha": 0},
    {"lambda": 0.1, "alpha": 0.1},
    {"lambda": 0.5, "alpha": 0.1},
    {"lambda": 0.6, "alpha": 0.1},
    {"lambda": 0.7, "alpha": 0.1},
    {"lambda": 0.8, "alpha": 0.1},
    {"lambda": 0.9, "alpha": 0.1},
    {"lambda": 1, "alpha": 0.1},
    {"lambda": 5, "alpha": 0.1},
    {"lambda": 10, "alpha": 0.1},
    {"lambda": 10, "alpha": 0.2},
    {"lambda": 10, "alpha": 0.3},
    {"lambda": 10, "alpha": 0.4},
    {"lambda": 10, "alpha": 0.5},
    {"lambda": 10, "alpha": 0.6},
    {"lambda": 10, "alpha": 0.7},
    {"lambda": 10, "alpha": 0.8},
    {"lambda": 10, "alpha": 0.9},
    {"lambda": 10, "alpha": 1},
]

results = []

for cfg in configs:
    print(f"\nTraining with lambda={cfg['lambda']}, alpha={cfg['alpha']}")

    model = XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        reg_lambda=cfg["lambda"],
        reg_alpha=cfg["alpha"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)

    val_df["temp_pred"] = preds

    def compute_fpr(group):
        subset = val_df[val_df["dialect_strict"] == group]
        FP = ((subset["label"] == 0) & (subset["temp_pred"] == 1)).sum()
        TN = ((subset["label"] == 0) & (subset["temp_pred"] == 0)).sum()
        return FP / (FP + TN + 1e-8)

    fpr_aae = compute_fpr("AAE")
    fpr_sae = compute_fpr("SAE")

    gap = abs(fpr_aae - fpr_sae)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    results.append({
        "lambda": cfg["lambda"],
        "alpha": cfg["alpha"],
        "accuracy": acc,
        "f1": f1,
        "FPR_AAE": fpr_aae,
        "FPR_SAE": fpr_sae,
        "FPR_gap": gap
    })

results_df = pd.DataFrame(results)
print("\nFinal Results:\n", results_df)