import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_split(path, dialect_col):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {path}")
    if dialect_col not in df.columns:
        raise ValueError(f"Missing '{dialect_col}' column in {path}")
    return df


def compute_group_metrics(df, pred_col, dialect_col):
    out = {}
    for group in ["AAE", "SAE"]:
        subset = df[df[dialect_col] == group]

        tp = ((subset["label"] == 1) & (subset[pred_col] == 1)).sum()
        fp = ((subset["label"] == 0) & (subset[pred_col] == 1)).sum()
        tn = ((subset["label"] == 0) & (subset[pred_col] == 0)).sum()
        fn = ((subset["label"] == 1) & (subset[pred_col] == 0)).sum()

        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        tpr = tp / (tp + fn + 1e-8)

        out[group] = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "TPR": float(tpr),
            "N": int(len(subset)),
        }

    out["FPR_gap"] = abs(out["AAE"]["FPR"] - out["SAE"]["FPR"])
    out["FNR_gap"] = abs(out["AAE"]["FNR"] - out["SAE"]["FNR"])
    out["TPR_gap"] = abs(out["AAE"]["TPR"] - out["SAE"]["TPR"])

    p_aae_non = (df[df[dialect_col] == "AAE"][pred_col] == 0).mean()
    p_sae_non = (df[df[dialect_col] == "SAE"][pred_col] == 0).mean()
    p_aae_tox = (df[df[dialect_col] == "AAE"][pred_col] == 1).mean()
    p_sae_tox = (df[df[dialect_col] == "SAE"][pred_col] == 1).mean()

    out["DIfav"] = float(p_aae_non / (p_sae_non + 1e-8))
    out["DIunfav"] = float(p_aae_tox / (p_sae_tox + 1e-8))
    return out


def make_vs_objective(group_train, alpha_aae=1.0, beta_aae=0.0, alpha_sae=1.0, beta_sae=0.0):
    """
    Training-time VS-loss objective.

    For each sample:
        s = alpha_g * z + beta_g
        loss = BCEWithLogits(s, y)

    Gradient wrt z:
        dL/dz = alpha_g * (sigmoid(s) - y)

    Hessian wrt z:
        d2L/dz2 = alpha_g^2 * sigmoid(s) * (1 - sigmoid(s))
    """
    group_train = np.asarray(group_train)

    def objective(preds, dtrain):
        y = dtrain.get_label()

        alpha = np.where(group_train == 1, alpha_aae, alpha_sae)
        beta = np.where(group_train == 1, beta_aae, beta_sae)

        s = alpha * preds + beta
        p = sigmoid(s)

        grad = (p - y) * alpha
        hess = p * (1.0 - p) * (alpha ** 2)
        return grad, hess

    return objective


def predict_with_vs(booster, X, group_array,
                    alpha_aae=1.0, beta_aae=0.0, alpha_sae=1.0, beta_sae=0.0):
    dmat = xgb.DMatrix(X)
    raw_margin = booster.predict(dmat, output_margin=True)

    group_array = np.asarray(group_array)
    alpha = np.where(group_array == 1, alpha_aae, alpha_sae)
    beta = np.where(group_array == 1, beta_aae, beta_sae)

    s = alpha * raw_margin + beta
    prob = sigmoid(s)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--val_csv", default="data/processed/val.csv")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--train_emb", default="data/processed/train_emb.npy")
    parser.add_argument("--val_emb", default="data/processed/val_emb.npy")
    parser.add_argument("--test_emb", default="data/processed/test_emb.npy")
    parser.add_argument("--dialect_col", default="dialect_strict")
    parser.add_argument("--num_round", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/vs_xgb_train_time")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = load_split(args.train_csv, args.dialect_col)
    val_df = load_split(args.val_csv, args.dialect_col)
    test_df = load_split(args.test_csv, args.dialect_col)

    X_train = np.load(args.train_emb)
    X_val = np.load(args.val_emb)
    X_test = np.load(args.test_emb)

    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    group_train = train_df[args.dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    group_val = val_df[args.dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    group_test = test_df[args.dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Small, practical search over AAE scaling/shift.
    # SAE stays fixed at alpha=1.0, beta=0.0.
    candidates = [
        {"alpha_aae": 1.0, "beta_aae": 0.0},
        {"alpha_aae": 0.95, "beta_aae": -0.10},
        {"alpha_aae": 0.90, "beta_aae": -0.15},
        {"alpha_aae": 0.90, "beta_aae": -0.25},
        {"alpha_aae": 1.05, "beta_aae": -0.10},
    ]

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": "hist",
        "seed": args.seed,
        "verbosity": 0,
    }

    best_candidate = None
    best_score = None
    best_booster = None

    print("Training VS-XGBoost candidates...\n")

    for cand in candidates:
        alpha_aae = cand["alpha_aae"]
        beta_aae = cand["beta_aae"]

        obj = make_vs_objective(
            group_train=group_train,
            alpha_aae=alpha_aae,
            beta_aae=beta_aae,
            alpha_sae=1.0,
            beta_sae=0.0
        )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            obj=obj
        )

        val_prob, val_pred = predict_with_vs(
            booster,
            X_val,
            group_val,
            alpha_aae=alpha_aae,
            beta_aae=beta_aae,
            alpha_sae=1.0,
            beta_sae=0.0
        )

        val_tmp = val_df.copy()
        val_tmp["vs_pred"] = val_pred

        metrics = compute_group_metrics(val_tmp, "vs_pred", args.dialect_col)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)

        # Prefer lower FPR gap, then higher F1, then higher accuracy
        score = (metrics["FPR_gap"], -val_f1, -val_acc)

        print(
            f"alpha_aae={alpha_aae:.2f}, beta_aae={beta_aae:.2f} | "
            f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
            f"FPR_gap={metrics['FPR_gap']:.4f}, DIunfav={metrics['DIunfav']:.4f}"
        )

        if best_score is None or score < best_score:
            best_score = score
            best_candidate = cand
            best_booster = booster

    print("\nBest candidate:", best_candidate)

    # Final test predictions
    test_prob, test_pred = predict_with_vs(
        best_booster,
        X_test,
        group_test,
        alpha_aae=best_candidate["alpha_aae"],
        beta_aae=best_candidate["beta_aae"],
        alpha_sae=1.0,
        beta_sae=0.0
    )

    out_df = test_df.copy()
    out_df["vs_prob"] = test_prob
    out_df["vs_pred"] = test_pred

    pred_path = os.path.join(args.out_dir, "vs_xgb_predictions.csv")
    out_df.to_csv(pred_path, index=False)

    model_path = os.path.join(args.out_dir, "vs_xgb_model.json")
    best_booster.save_model(model_path)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_metrics = compute_group_metrics(out_df, "vs_pred", args.dialect_col)

    summary = {
        "best_candidate": best_candidate,
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "AAE": test_metrics["AAE"],
        "SAE": test_metrics["SAE"],
        "FPR_gap": test_metrics["FPR_gap"],
        "FNR_gap": test_metrics["FNR_gap"],
        "TPR_gap": test_metrics["TPR_gap"],
        "DIfav": test_metrics["DIfav"],
        "DIunfav": test_metrics["DIunfav"],
    }

    json_path = os.path.join(args.out_dir, "vs_xgb_summary.json")
    txt_path = os.path.join(args.out_dir, "vs_xgb_summary.txt")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("VS-XGBoost Summary\n")
        f.write("==================\n\n")
        f.write(f"Best candidate: {best_candidate}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")

        f.write(f"AAE FPR: {test_metrics['AAE']['FPR']:.4f}\n")
        f.write(f"SAE FPR: {test_metrics['SAE']['FPR']:.4f}\n")
        f.write(f"FPR Gap: {test_metrics['FPR_gap']:.4f}\n\n")

        f.write(f"AAE FNR: {test_metrics['AAE']['FNR']:.4f}\n")
        f.write(f"SAE FNR: {test_metrics['SAE']['FNR']:.4f}\n")
        f.write(f"FNR Gap: {test_metrics['FNR_gap']:.4f}\n\n")

        f.write(f"DIfav: {test_metrics['DIfav']:.4f}\n")
        f.write(f"DIunfav: {test_metrics['DIunfav']:.4f}\n")

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved summaries to: {json_path} and {txt_path}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test FPR Gap: {test_metrics['FPR_gap']:.4f}")
    print(f"Test FNR Gap: {test_metrics['FNR_gap']:.4f}")

if __name__ == "__main__":
    main()