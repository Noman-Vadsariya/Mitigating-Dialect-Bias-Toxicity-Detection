# Imports
import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost + Vector Scaling model")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--train_emb", type=str, required=True)
    parser.add_argument("--val_emb", type=str, required=True)
    parser.add_argument("--test_emb", type=str, required=True)

    parser.add_argument("--dialect_col", type=str, default="dialect_strict")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--out_dir", type=str, required=True)

    return parser.parse_args()


# Load data:
# Numeric embeddings from .npy
# Labels and group metadata from .csv
def load_data(args):
    X_train = np.load(args.train_emb)
    X_val = np.load(args.val_emb)
    X_test = np.load(args.test_emb)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    y_train = train_df[args.label_col].values
    y_val = val_df[args.label_col].values
    y_test = test_df[args.label_col].values

    return X_train, X_val, X_test, train_df, val_df, test_df, y_train, y_val, y_test


# Train the base XGBoost model
# This is the base toxicity model. Train exactly like the baseline first, then apply vector 
# scaling afterwards.
def train_base_xgb(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return model


# Math helpers
# Sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Logit: z = log(p / (1 - p))
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# Apply vector scaling by group
# s = alpha_g * z + beta_g
# alpha_g rescales the margin
# beta_g shifts the boundary
#
# For each example:
# - If it belongs to AAE, apply the AAE transform
# - If it belongs to SAE, apply the SAE transform
# For simplicity, keep SAE fixed:
# - alpha_sae = 1
# - beta_sae = 0
# Which means SAE scores stay unchanged, only adjusting AAE (since fairness problem is AAE 
# being overflagged)
def apply_vector_scaling(
    probs,
    groups,
    alpha_aae,
    beta_aae,
    alpha_sae=1.0,
    beta_sae=0.0
):
    logits = safe_logit(probs)
    scaled_logits = logits.copy()

    is_aae = (groups == "AAE")
    is_sae = (groups == "SAE")

    scaled_logits[is_aae] = alpha_aae * logits[is_aae] + beta_aae
    scaled_logits[is_sae] = alpha_sae * logits[is_sae] + beta_sae

    scaled_probs = sigmoid(scaled_logits)
    preds = (scaled_probs >= 0.5).astype(int)

    return scaled_probs, preds


# Add fairness metric helpers
# Before tuning VS, measure whether it is actually helping
# FPR = FP / (FP + TN)
# FNR = FN / (FN + TP)
def compute_fpr(df, label_col, pred_col, group_col, group_name):
    subset = df[df[group_col] == group_name]
    fp = ((subset[label_col] == 0) & (subset[pred_col] == 1)).sum()
    tn = ((subset[label_col] == 0) & (subset[pred_col] == 0)).sum()
    return fp / (fp + tn + 1e-8)

def compute_fnr(df, label_col, pred_col, group_col, group_name):
    subset = df[df[group_col] == group_name]
    fn = ((subset[label_col] == 1) & (subset[pred_col] == 0)).sum()
    tp = ((subset[label_col] == 1) & (subset[pred_col] == 1)).sum()
    return fn / (fn + tp + 1e-8)

def compute_disparate_impact(df, pred_col, group_col):
    p_pred1_aae = (df[df[group_col] == "AAE"][pred_col] == 1).mean()
    p_pred1_sae = (df[df[group_col] == "SAE"][pred_col] == 1).mean()

    p_pred0_aae = (df[df[group_col] == "AAE"][pred_col] == 0).mean()
    p_pred0_sae = (df[df[group_col] == "SAE"][pred_col] == 0).mean()

    di_unfav = p_pred1_aae / (p_pred1_sae + 1e-8)
    di_fav = p_pred0_aae / (p_pred0_sae + 1e-8)

    return di_fav, di_unfav


# Metric summary
def evaluate_predictions(df, label_col, pred_col, group_col):
    metrics = {}

    y_true = df[label_col].values
    y_pred = df[pred_col].values

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)

    metrics["FPR_AAE"] = compute_fpr(df, label_col, pred_col, group_col, "AAE")
    metrics["FPR_SAE"] = compute_fpr(df, label_col, pred_col, group_col, "SAE")
    metrics["FPR_gap"] = abs(metrics["FPR_AAE"] - metrics["FPR_SAE"])

    metrics["FNR_AAE"] = compute_fnr(df, label_col, pred_col, group_col, "AAE")
    metrics["FNR_SAE"] = compute_fnr(df, label_col, pred_col, group_col, "SAE")
    metrics["FNR_gap"] = abs(metrics["FNR_AAE"] - metrics["FNR_SAE"])

    di_fav, di_unfav = compute_disparate_impact(df, pred_col, group_col)
    metrics["DIfav"] = di_fav
    metrics["DIunfav"] = di_unfav

    return metrics


# Validation search
def search_vs_parameters(model, X_val, val_df, label_col, group_col):
    # Gets the original toxic probability from XGBoost for every validation example
    base_probs = model.predict_proba(X_val)[:, 1]

    # First search space
    # Intentionally small for quick debugging
    alpha_grid = [0.6, 0.8, 1.0, 1.2]
    beta_grid = [-1.0, -0.5, -0.2, 0.0, 0.2]

    results = []

    for alpha_aae in alpha_grid:
        for beta_aae in beta_grid:
            temp_df = val_df.copy()

            scaled_probs, scaled_preds = apply_vector_scaling(
                probs=base_probs,
                groups=temp_df[group_col].values,
                alpha_aae=alpha_aae,
                beta_aae=beta_aae,
                alpha_sae=1.0,
                beta_sae=0.0
            )

            temp_df["vs_prob"] = scaled_probs
            temp_df["vs_pred"] = scaled_preds

            metrics = evaluate_predictions(
                df=temp_df,
                label_col=label_col,
                pred_col="vs_pred",
                group_col=group_col
            )

            metrics["alpha_aae"] = alpha_aae
            metrics["beta_aae"] = beta_aae

            results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df, base_probs


# Selects best candidate
def pick_best_candidate(results_df, min_f1_ratio=0.98):
    baseline_f1 = results_df.loc[
        (results_df["alpha_aae"] == 1.0) & (results_df["beta_aae"] == 0.0),
        "f1"
    ].iloc[0]

    eligible = results_df[results_df["f1"] >= min_f1_ratio * baseline_f1].copy()

    if len(eligible) == 0:
        eligible = results_df.copy()

    eligible = eligible.sort_values(
        by=["FPR_gap", "FNR_gap", "f1"],
        ascending=[True, True, False]
    )

    best_row = eligible.iloc[0].to_dict()
    return best_row


# Run best setting on test data
def apply_best_vs_to_test(model, X_test, test_df, best_params, group_col):
    base_probs = model.predict_proba(X_test)[:, 1]

    scaled_probs, scaled_preds = apply_vector_scaling(
        probs=base_probs,
        groups=test_df[group_col].values,
        alpha_aae=best_params["alpha_aae"],
        beta_aae=best_params["beta_aae"],
        alpha_sae=1.0,
        beta_sae=0.0
    )

    output_df = test_df.copy()
    output_df["xgb_base_prob"] = base_probs
    output_df["vs_prob"] = scaled_probs
    output_df["vs_pred"] = scaled_preds

    return output_df

def write_metrics_summary_txt(summary_txt_path, test_metrics):
        with open(summary_txt_path, "w") as f:
            f.write("===== VS-XGBoost =====\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"F1: {test_metrics['f1']:.4f}\n")
            f.write(f"FPR AAE: {test_metrics['FPR_AAE']:.4f}\n")
            f.write(f"FPR SAE: {test_metrics['FPR_SAE']:.4f}\n")
            f.write(f"FPR Gap: {test_metrics['FPR_gap']:.4f}\n")
            f.write(f"FNR AAE: {test_metrics['FNR_AAE']:.4f}\n")
            f.write(f"FNR SAE: {test_metrics['FNR_SAE']:.4f}\n")
            f.write(f"FNR Gap: {test_metrics['FNR_gap']:.4f}\n")
            f.write(f"DIfav: {test_metrics['DIfav']:.4f}\n")
            f.write(f"DIunfav: {test_metrics['DIunfav']:.4f}\n")

# Save outputs
def save_outputs(out_dir, model, results_df, best_params, test_output_df, test_metrics):
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "vs_xgb_model.joblib")
    candidates_path = os.path.join(out_dir, "vs_xgb_candidates.csv")
    summary_json_path = os.path.join(out_dir, "vs_xgb_summary.json")
    preds_path = os.path.join(out_dir, "vs_xgb_predictions.csv")

    summary_txt_path = "data/results/vs_xgb_train_time/vs_xgb_summary.txt"

    joblib.dump(model, model_path)
    results_df.to_csv(candidates_path, index=False)
    test_output_df.to_csv(preds_path, index=False)

    with open(summary_json_path, "w") as f:
        json.dump(best_params, f, indent=2)

    write_metrics_summary_txt(summary_txt_path, test_metrics)

    print(f"Saved model -> {model_path}")
    print(f"Saved candidates -> {candidates_path}")
    print(f"Saved summary json -> {summary_json_path}")
    print(f"Saved summary txt -> {summary_txt_path}")
    print(f"Saved predictions -> {preds_path}")


def main():
    args = parse_args()

    X_train, X_val, X_test, train_df, val_df, test_df, y_train, y_val, y_test = load_data(args)

    print("Training base XGBoost model...")
    model = train_base_xgb(X_train, y_train, X_val, y_val)

    print("\nSearching vector scaling parameters on validation set...")
    results_df, _ = search_vs_parameters(
        model=model,
        X_val=X_val,
        val_df=val_df,
        label_col=args.label_col,
        group_col=args.dialect_col
    )

    print("\nCandidate results:")
    print(results_df.sort_values(by=["FPR_gap", "f1"], ascending=[True, False]).head(10))

    best_params = pick_best_candidate(results_df)

    print("\nBest VS parameters:")
    print(best_params)

    print("\nApplying best VS parameters to test set...")
    test_output_df = apply_best_vs_to_test(
        model=model,
        X_test=X_test,
        test_df=test_df,
        best_params=best_params,
        group_col=args.dialect_col
    )

    test_metrics = evaluate_predictions(
        df=test_output_df,
        label_col=args.label_col,
        pred_col="vs_pred",
        group_col=args.dialect_col
    )

    print("\nFinal test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    best_params["test_metrics"] = test_metrics
            
    save_outputs(
        out_dir=args.out_dir,
        model=model,
        results_df=results_df,
        best_params=best_params,
        test_output_df=test_output_df,
        test_metrics=test_metrics
    )


if __name__ == "__main__":
    main()