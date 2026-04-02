import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_split(csv_path, emb_path, dialect_col):
    df = pd.read_csv(csv_path)
    X = np.load(emb_path)

    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {csv_path}")
    if dialect_col not in df.columns:
        raise ValueError(f"Missing '{dialect_col}' column in {csv_path}")
    if len(df) != len(X):
        raise ValueError(f"Row mismatch: {csv_path} has {len(df)} rows, embeddings have {len(X)} rows")

    y = df["label"].astype(int).values
    g = df[dialect_col].map({"SAE": 0, "AAE": 1}).astype(int).values
    return df, X, y, g


def compute_group_metrics(df, pred_col, dialect_col):
    out = {}
    for group in ["AAE", "SAE"]:
        sub = df[df[dialect_col] == group]

        TP = ((sub["label"] == 1) & (sub[pred_col] == 1)).sum()
        FP = ((sub["label"] == 0) & (sub[pred_col] == 1)).sum()
        TN = ((sub["label"] == 0) & (sub[pred_col] == 0)).sum()
        FN = ((sub["label"] == 1) & (sub[pred_col] == 0)).sum()

        FPR = FP / (FP + TN + 1e-8)
        FNR = FN / (FN + TP + 1e-8)
        TPR = TP / (TP + FN + 1e-8)

        out[group] = {
            "TP": int(TP),
            "FP": int(FP),
            "TN": int(TN),
            "FN": int(FN),
            "FPR": float(FPR),
            "FNR": float(FNR),
            "TPR": float(TPR),
            "N": int(len(sub)),
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


def fit_adversary(raw_margin, g_train, adv_c=1.0):
    """
    Lightweight adversary:
    predict dialect from current XGBoost raw margin.
    """
    if len(np.unique(g_train)) < 2:
        return 0.0, 0.0

    clf = LogisticRegression(
        C=adv_c,
        solver="lbfgs",
        max_iter=1000
    )
    clf.fit(raw_margin.reshape(-1, 1), g_train)

    w = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return w, b


def train_one_adv_model(
    X_train, y_train, g_train,
    X_val, y_val, g_val,
    params,
    lambda_adv=0.1,
    adv_c=1.0,
    num_round=100,
    dialect_col="dialect_strict"
):
    """
    Training-time adversarial XGBoost.

    Main toxicity objective:
        L_tox

    Adversary:
        predicts dialect from raw margin

    Composite objective:
        L = L_tox - lambda_adv * L_adv

    Gradient wrt margin z:
        grad = (sigmoid(z) - y) - lambda_adv * grad_adv

    Hessian:
        hess = tox_hess + lambda_adv * adv_hess
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "adv_acc_train": [],
        "adv_loss_train": [],
        "val_acc": [],
        "val_f1": [],
        "val_fpr_gap": [],
        "val_fnr_gap": [],
        "val_difunfav": [],
    }

    for r in range(num_round):
        raw_train = np.zeros_like(y_train, dtype=float) if booster is None else booster.predict(dtrain, output_margin=True)

        adv_w, adv_b = fit_adversary(raw_train, g_train, adv_c=adv_c)

        def objective(preds, dmat):
            y = dmat.get_label()

            # Toxicity gradient
            p_tox = sigmoid(preds)
            grad_tox = p_tox - y
            hess_tox = p_tox * (1.0 - p_tox)

            # Adversary gradient wrt margin
            adv_logit = adv_w * preds + adv_b
            p_adv = sigmoid(adv_logit)
            grad_adv = (p_adv - g_train) * adv_w
            hess_adv = p_adv * (1.0 - p_adv) * (adv_w ** 2)

            grad = grad_tox - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            return grad, hess

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            obj=objective,
            xgb_model=booster,
            verbose_eval=False
        )

        # Track train / val behavior
        train_raw = booster.predict(dtrain, output_margin=True)
        val_raw = booster.predict(dval, output_margin=True)

        train_prob = sigmoid(train_raw)
        val_prob = sigmoid(val_raw)

        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)

        # Main losses
        history["train_loss"].append(float(log_loss(y_train, train_prob, labels=[0, 1])))
        history["val_loss"].append(float(log_loss(y_val, val_prob, labels=[0, 1])))

        # Adversary behavior on train
        adv_train_prob = sigmoid(adv_w * raw_train + adv_b)
        adv_train_pred = (adv_train_prob >= 0.5).astype(int)
        history["adv_acc_train"].append(float(accuracy_score(g_train, adv_train_pred)))
        history["adv_loss_train"].append(float(log_loss(g_train, adv_train_prob, labels=[0, 1])))

        # Val metrics
        val_tmp = pd.DataFrame({
            "label": y_val,
            "pred": val_pred,
            dialect_col: ["AAE" if x == 1 else "SAE" for x in g_val]
        })

        m = compute_group_metrics(val_tmp, "pred", dialect_col)
        history["val_acc"].append(float(accuracy_score(y_val, val_pred)))
        history["val_f1"].append(float(f1_score(y_val, val_pred)))
        history["val_fpr_gap"].append(float(m["FPR_gap"]))
        history["val_fnr_gap"].append(float(m["FNR_gap"]))
        history["val_difunfav"].append(float(m["DIunfav"]))

        # print(
        #     f"Round {r+1:03d}/{num_round} | "
        #     f"train_loss={history['train_loss'][-1]:.4f} | "
        #     f"val_loss={history['val_loss'][-1]:.4f} | "
        #     f"adv_acc_train={history['adv_acc_train'][-1]:.4f} | "
        #     f"val_f1={history['val_f1'][-1]:.4f} | "
        #     f"val_fpr_gap={history['val_fpr_gap'][-1]:.4f}"
        # )

    return booster, history


def predict_with_booster(booster, X, g, lambda_adv=0.0, adv_w=0.0, adv_b=0.0):
    """
    Standard inference for the final selected model.
    Since the model is already trained adversarially,
    we use the raw booster margin and apply sigmoid.
    """
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat, output_margin=True)
    prob = sigmoid(raw)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="../data/processed/train.csv")
    parser.add_argument("--val_csv", default="../data/processed/val.csv")
    parser.add_argument("--test_csv", default="../data/processed/test.csv")
    parser.add_argument("--train_emb", default="../data/embeddings/train_emb.npy")
    parser.add_argument("--val_emb", default="../data/embeddings/val_emb.npy")
    parser.add_argument("--test_emb", default="../data/embeddings/test_emb.npy")
    parser.add_argument("--dialect_col", default="dialect_strict")
    parser.add_argument("--out_dir", default="results/adv_xgb")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_grid", default="0.0,0.05,0.1,0.25,0.5")
    parser.add_argument("--adv_c_grid", default="0.5,1.0,2.0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df, X_train, y_train, g_train = load_split(args.train_csv, args.train_emb, args.dialect_col)
    val_df, X_val, y_val, g_val = load_split(args.val_csv, args.val_emb, args.dialect_col)
    test_df, X_test, y_test, g_test = load_split(args.test_csv, args.test_emb, args.dialect_col)

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    adv_c_grid = [float(x) for x in args.adv_c_grid.split(",")]

    base_params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": "hist",
        "seed": args.seed,
        "verbosity": 0,
    }

    candidates = []
    for lam in lambda_grid:
        for adv_c in adv_c_grid:
            candidates.append({"lambda_adv": lam, "adv_c": adv_c})

    best = None
    best_score = None
    best_booster = None
    best_history = None

    summary_rows = []

    print("\nStarting adversarial tuning...\n")

    for cand in candidates:
        lam = cand["lambda_adv"]
        adv_c = cand["adv_c"]

        booster, history = train_one_adv_model(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            X_val=X_val,
            y_val=y_val,
            g_val=g_val,
            params=base_params,
            lambda_adv=lam,
            adv_c=adv_c,
            num_round=args.num_round,
            dialect_col=args.dialect_col
        )

        val_raw = booster.predict(xgb.DMatrix(X_val), output_margin=True)
        val_prob = sigmoid(val_raw)
        val_pred = (val_prob >= 0.5).astype(int)

        val_eval = val_df.copy()
        val_eval["adv_pred"] = val_pred

        m = compute_group_metrics(val_eval, "adv_pred", args.dialect_col)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)

        # Preference: lower FPR gap, then higher F1, then higher accuracy
        score = (m["FPR_gap"], -val_f1, -val_acc)

        summary_rows.append({
            "lambda_adv": lam,
            "adv_c": adv_c,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_FPR_gap": m["FPR_gap"],
            "val_FNR_gap": m["FNR_gap"],
            "val_DIfav": m["DIfav"],
            "val_DIunfav": m["DIunfav"],
        })

        print(
            f"lambda={lam:.3f}, adv_c={adv_c:.3f} | "
            f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, "
            f"FPR_gap={m['FPR_gap']:.4f}, DIunfav={m['DIunfav']:.4f}"
        )

        if best_score is None or score < best_score:
            best_score = score
            best = cand
            best_booster = booster
            best_history = history

    print("\nBest candidate:", best)

    # Final test evaluation with best model
    test_prob, test_pred = predict_with_booster(
        best_booster, X_test, g_test
    )

    out_df = test_df.copy()
    out_df["adv_prob"] = test_prob
    out_df["adv_pred"] = test_pred

    pred_path = os.path.join(args.out_dir, "adv_xgb_predictions.csv")
    out_df.to_csv(pred_path, index=False)

    model_path = os.path.join(args.out_dir, "adv_xgb_model.json")
    best_booster.save_model(model_path)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_eval = compute_group_metrics(out_df, "adv_pred", args.dialect_col)

    # Save tuning table
    tuning_df = pd.DataFrame(summary_rows)
    tuning_df.to_csv(os.path.join(args.out_dir, "adv_xgb_tuning.csv"), index=False)

    # Save history
    with open(os.path.join(args.out_dir, "loss_history.json"), "w") as f:
        json.dump(best_history, f, indent=2)

    # Save human-readable summary
    summary_path = os.path.join(args.out_dir, "adv_xgb_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Adversarial XGBoost Summary\n")
        f.write("==========================\n\n")
        f.write(f"Best candidate: {best}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")

        f.write(f"AAE FPR: {test_eval['AAE']['FPR']:.4f}\n")
        f.write(f"SAE FPR: {test_eval['SAE']['FPR']:.4f}\n")
        f.write(f"FPR Gap: {test_eval['FPR_gap']:.4f}\n\n")

        f.write(f"AAE FNR: {test_eval['AAE']['FNR']:.4f}\n")
        f.write(f"SAE FNR: {test_eval['SAE']['FNR']:.4f}\n")
        f.write(f"FNR Gap: {test_eval['FNR_gap']:.4f}\n\n")

        f.write(f"DIfav: {test_eval['DIfav']:.4f}\n")
        f.write(f"DIunfav: {test_eval['DIunfav']:.4f}\n")

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved tuning table to: {os.path.join(args.out_dir, 'adv_xgb_tuning.csv')}")
    print(f"Saved history to: {os.path.join(args.out_dir, 'loss_history.json')}")
    print(f"Saved summary to: {summary_path}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"AAE FPR: {test_eval['AAE']['FPR']:.4f}")
    print(f"SAE FPR: {test_eval['SAE']['FPR']:.4f}")
    print(f"Test FPR Gap: {test_eval['FPR_gap']:.4f}")
    print(f"Test FNR Gap: {test_eval['FNR_gap']:.4f}")

if __name__ == "__main__":
    main()