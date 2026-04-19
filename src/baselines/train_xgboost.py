import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Load embeddings
X_train = np.load("../../data/embeddings/twitterAAE/unbalanced/train_emb.npy")
X_val = np.load("../../data/embeddings/twitterAAE/unbalanced/val_emb.npy")
X_test = np.load("../../data/embeddings/twitterAAE/unbalanced/test_emb.npy")

# Load labels
train_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/train.csv")
val_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/val.csv")
test_df = pd.read_csv("../../data/processed/twitterAAE/unbalanced/test.csv")

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# Train model
model = XGBClassifier(
    max_depth=5,
    n_estimators=100,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save model
joblib.dump(model, "../../models/xgb_baseline.joblib")

# =========================
# SAVE PREDICTIONS
# =========================

test_df["xgb_pred"] = y_pred
test_df["xgb_prob"] = model.predict_proba(X_test)[:, 1]

# Save
os.makedirs("../../data/results", exist_ok=True)

test_df.to_csv("../../data/results/xgb_predictions.csv", index=False)

print("Saved XGBoost predictions → ../../data/results/xgb_predictions.csv")