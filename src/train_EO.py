import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42
np.random.seed(SEED)

# =========================
# CONFIG
# =========================
TRAIN_CSV = "../data/processed/train.csv"
VAL_CSV   = "../data/processed/val.csv"
TEST_CSV  = "../data/processed/test.csv"

TRAIN_EMB = "../data/embeddings/train_emb.npy"
VAL_EMB   = "../data/embeddings/val_emb.npy"
TEST_EMB  = "../data/embeddings/test_emb.npy"

DIALECT_COL = "dialect_strict"   # or "dialect_relaxed"

OUT_DIR = "results/eo_xgb_from_scratch"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# UTILITIES
# =========================
def to_1d_numpy(x, dtype=None):
    arr = np.asarray(x).reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def load_split(csv_path, emb_path, dialect_col):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    X = np.load(emb_path)

    if len(df) != len(X):
        raise ValueError(
            f"Row mismatch for {csv_path}: {len(df)} CSV rows vs {len(X)} embedding rows."
        )

    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {csv_path}")
    if dialect_col not in df.columns:
        raise ValueError(f"Missing '{dialect_col}' column in {csv_path}")

    y = df["label"].astype(int).values
    g = df[dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    return df, np.asarray(X, dtype=np.float64), y, g


def compute_group_metrics(df, pred_col, group_col):
    results = {}
    for group_name in ["AAE", "SAE"]:
        sub = df[df[group_col] == group_name]

        tp = ((sub["label"] == 1) & (sub[pred_col] == 1)).sum()
        fp = ((sub["label"] == 0) & (sub[pred_col] == 1)).sum()
        tn = ((sub["label"] == 0) & (sub[pred_col] == 0)).sum()
        fn = ((sub["label"] == 1) & (sub[pred_col] == 0)).sum()

        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        tpr = tp / (tp + fn + 1e-8)

        results[group_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "TPR": float(tpr),
            "N": int(len(sub)),
        }

    results["FPR_gap"] = abs(results["AAE"]["FPR"] - results["SAE"]["FPR"])
    results["FNR_gap"] = abs(results["AAE"]["FNR"] - results["SAE"]["FNR"])
    results["TPR_gap"] = abs(results["AAE"]["TPR"] - results["SAE"]["TPR"])

    p_aae_non = (df[df[group_col] == "AAE"][pred_col] == 0).mean()
    p_sae_non = (df[df[group_col] == "SAE"][pred_col] == 0).mean()
    p_aae_tox = (df[df[group_col] == "AAE"][pred_col] == 1).mean()
    p_sae_tox = (df[df[group_col] == "SAE"][pred_col] == 1).mean()

    results["DIfav"] = float(p_aae_non / (p_sae_non + 1e-8))
    results["DIunfav"] = float(p_aae_tox / (p_sae_tox + 1e-8))
    return results


def save_summary(path, title, metrics_dict):
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                f.write(f"{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")
            f.write("\n")


def apply_group_thresholds(scores, groups, t_aae, t_sae):
    """
    groups: 1 = AAE, 0 = SAE
    """
    scores = np.asarray(scores).reshape(-1)
    groups = np.asarray(groups).reshape(-1)

    preds = np.zeros_like(scores, dtype=int)
    preds[groups == 1] = (scores[groups == 1] >= t_aae).astype(int)
    preds[groups == 0] = (scores[groups == 0] >= t_sae).astype(int)
    return preds


def search_group_thresholds_with_constraints(
    val_scores,
    y_val,
    g_val,
    baseline_val_acc,
    baseline_val_f1,
    grid_size=101,
    epsilon_acc=0.02,
    epsilon_f1=0.02
):
    val_scores = np.asarray(val_scores).reshape(-1)
    y_val = to_1d_numpy(y_val, dtype=int)
    g_val = to_1d_numpy(g_val, dtype=int)

    qs = np.linspace(0.0, 1.0, grid_size)
    cand = np.unique(np.quantile(val_scores, qs))

    best = None
    best_key = None

    for t_aae in cand:
        for t_sae in cand:
            val_pred = apply_group_thresholds(val_scores, g_val, t_aae, t_sae)

            tmp = pd.DataFrame({
                "label": y_val,
                "pred": val_pred,
                "g": np.where(g_val == 1, "AAE", "SAE")
            })
            m = compute_group_metrics(tmp, "pred", "g")

            acc = accuracy_score(y_val, val_pred)
            f1 = f1_score(y_val, val_pred)

            # performance floor
            if acc < baseline_val_acc - epsilon_acc:
                continue
            if f1 < baseline_val_f1 - epsilon_f1:
                continue

            key = (m["FPR_gap"] + m["TPR_gap"], -acc, -f1)

            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "t_aae": float(t_aae),
                    "t_sae": float(t_sae),
                    "val_accuracy": float(acc),
                    "val_f1": float(f1),
                    "val_metrics": m
                }

    return best


# =========================
# LOAD DATA
# =========================
train_df, X_train, y_train, g_train = load_split(TRAIN_CSV, TRAIN_EMB, DIALECT_COL)
val_df,   X_val,   y_val,   g_val   = load_split(VAL_CSV, VAL_EMB, DIALECT_COL)
test_df,  X_test,  y_test,  g_test  = load_split(TEST_CSV, TEST_EMB, DIALECT_COL)

y_train = to_1d_numpy(y_train, dtype=int)
y_val   = to_1d_numpy(y_val, dtype=int)
y_test  = to_1d_numpy(y_test, dtype=int)

g_train = to_1d_numpy(g_train, dtype=int)
g_val   = to_1d_numpy(g_val, dtype=int)
g_test  = to_1d_numpy(g_test, dtype=int)


# =========================
# 1) TRAIN BASE XGBOOST
# =========================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    tree_method="hist",
)

xgb.fit(X_train, y_train)

baseline_val_scores = xgb.predict_proba(X_val)[:, 1]
baseline_test_scores = xgb.predict_proba(X_test)[:, 1]

baseline_val_pred = (baseline_val_scores >= 0.5).astype(int)
baseline_test_pred = (baseline_test_scores >= 0.5).astype(int)

baseline_val_pred = (baseline_val_scores >= 0.5).astype(int)
baseline_val_acc = accuracy_score(y_val, baseline_val_pred)
baseline_val_f1 = f1_score(y_val, baseline_val_pred)
print(f"Baseline XGBoost validation accuracy: {baseline_val_acc:.4f}")
print(f"Baseline XGBoost validation F1: {baseline_val_f1:.4f}")

# =========================
# 2) LEARN GROUP-SPECIFIC THRESHOLDS ON VALIDATION SET
# =========================
best_thr = search_group_thresholds_with_constraints(
    val_scores=baseline_val_scores,
    y_val=y_val,
    g_val=g_val,
    baseline_val_acc=baseline_val_acc,
    baseline_val_f1=baseline_val_f1,
    grid_size=101,
    epsilon_acc=0.02,
    epsilon_f1=0.02
)

t_aae = best_thr["t_aae"]
t_sae = best_thr["t_sae"]

print("\nLearned thresholds:")
print(f"  AAE threshold: {t_aae:.4f}")
print(f"  SAE threshold: {t_sae:.4f}")
print(f"  Validation accuracy: {best_thr['val_accuracy']:.4f}")
print(f"  Validation F1: {best_thr['val_f1']:.4f}")
print(f"  Validation FPR gap: {best_thr['val_metrics']['FPR_gap']:.4f}")
print(f"  Validation TPR gap: {best_thr['val_metrics']['TPR_gap']:.4f}")


# =========================
# 3) APPLY THRESHOLDS ON TEST SET
# =========================
eo_test_pred = apply_group_thresholds(
    scores=baseline_test_scores,
    groups=g_test,
    t_aae=t_aae,
    t_sae=t_sae
)

baseline_test_df = test_df.copy()
baseline_test_df["baseline_score"] = baseline_test_scores
baseline_test_df["baseline_pred"] = baseline_test_pred

eo_test_df = test_df.copy()
eo_test_df["eo_score"] = baseline_test_scores
eo_test_df["eo_pred"] = eo_test_pred


# =========================
# 4) METRICS
# =========================
baseline_metrics = {
    "accuracy": float(accuracy_score(y_test, baseline_test_pred)),
    "f1": float(f1_score(y_test, baseline_test_pred)),
    "precision": float(precision_score(y_test, baseline_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, baseline_test_pred, zero_division=0)),
    "group_metrics": compute_group_metrics(baseline_test_df, "baseline_pred", DIALECT_COL),
}

eo_metrics = {
    "accuracy": float(accuracy_score(y_test, eo_test_pred)),
    "f1": float(f1_score(y_test, eo_test_pred)),
    "precision": float(precision_score(y_test, eo_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, eo_test_pred, zero_division=0)),
    "group_metrics": compute_group_metrics(eo_test_df, "eo_pred", DIALECT_COL),
}


# =========================
# 5) SAVE OUTPUTS
# =========================
baseline_test_df.to_csv(os.path.join(OUT_DIR, "baseline_xgb_predictions.csv"), index=False)
eo_test_df.to_csv(os.path.join(OUT_DIR, "eo_xgb_predictions.csv"), index=False)

all_metrics = {
    "dialect_col": DIALECT_COL,
    "learned_thresholds": {
        "AAE": t_aae,
        "SAE": t_sae
    },
    "validation": best_thr,
    "baseline": baseline_metrics,
    "equalized_odds_postprocess": eo_metrics,
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

save_summary(
    os.path.join(OUT_DIR, "metrics.txt"),
    "XGBoost Equalized Odds from Scratch",
    all_metrics
)

print("\n=== BASELINE XGBOOST ===")
print(json.dumps(baseline_metrics, indent=2))

print("\n=== EO POSTPROCESSING (FROM SCRATCH) ===")
print(json.dumps(eo_metrics, indent=2))

print(f"\nSaved outputs to: {OUT_DIR}")