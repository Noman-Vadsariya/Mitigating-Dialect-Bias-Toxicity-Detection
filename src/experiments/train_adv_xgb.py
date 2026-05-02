import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
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


def leaves_to_sparse(leaf_ids: np.ndarray, num_trees: int, max_leaves_per_tree: int = 256):
    n_samples = leaf_ids.shape[0]
    rows = []
    cols = []

    for tree_idx in range(num_trees):
        tree_leaves = leaf_ids[:, tree_idx]
        col_offset = tree_idx * max_leaves_per_tree
        rows.extend(range(n_samples))
        cols.extend(col_offset + tree_leaves)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_samples, num_trees * max_leaves_per_tree),
        dtype=np.float32
    )


def fit_leaf_adversary(booster, dtrain, g_train, adv_c=1.0, sample_weight=None):
    if len(np.unique(g_train)) < 2:
        return None, None

    leaf_ids = booster.predict(dtrain, pred_leaf=True)
    num_trees = leaf_ids.shape[1] if len(leaf_ids.shape) > 1 else 1

    if num_trees == 1:
        leaf_ids = leaf_ids.reshape(-1, 1)

    leaf_sparse = leaves_to_sparse(leaf_ids, num_trees)

    clf = LogisticRegression(
        C=adv_c,
        solver="saga",
        max_iter=500,
        warm_start=True,
        n_jobs=-1,
    )
    clf.fit(leaf_sparse, g_train, sample_weight=sample_weight)

    return clf, leaf_sparse


def fit_margin_adversary(raw_margin, g_train, adv_c=1.0, sample_weight=None):
    if len(np.unique(g_train)) < 2:
        return 0.0, 0.0

    clf = LogisticRegression(
        C=adv_c,
        solver="lbfgs",
        max_iter=1000
    )
    clf.fit(raw_margin.reshape(-1, 1), g_train, sample_weight=sample_weight)

    w = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return w, b


def compute_group_weights(g: np.ndarray) -> np.ndarray:
    n = len(g)
    n_aae = (g == 1).sum()
    n_sae = (g == 0).sum()

    w_aae = n / (2.0 * n_aae) if n_aae > 0 else 1.0
    w_sae = n / (2.0 * n_sae) if n_sae > 0 else 1.0

    weights = np.where(g == 1, w_aae, w_sae)
    return weights.astype(np.float32)


def tune_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    dialect_col: str = "dialect_strict",
    grid_size: int = 50,
    min_f1_ratio: float = 0.95,
):
    baseline_pred = (probs >= 0.5).astype(int)
    baseline_f1 = f1_score(y, baseline_pred)
    min_f1 = baseline_f1 * min_f1_ratio

    thresholds = np.linspace(0.1, 0.9, grid_size)

    best_fpr_gap = float('inf')
    best_t_aae = 0.5
    best_t_sae = 0.5
    best_metrics = None

    for t_aae in thresholds:
        for t_sae in thresholds:
            pred = np.zeros_like(y)
            pred[g == 1] = (probs[g == 1] >= t_aae).astype(int)
            pred[g == 0] = (probs[g == 0] >= t_sae).astype(int)

            curr_f1 = f1_score(y, pred)
            if curr_f1 < min_f1:
                continue

            aae_mask = g == 1
            sae_mask = g == 0

            aae_fp = ((y[aae_mask] == 0) & (pred[aae_mask] == 1)).sum()
            aae_tn = ((y[aae_mask] == 0) & (pred[aae_mask] == 0)).sum()
            aae_fpr = aae_fp / (aae_fp + aae_tn + 1e-8)

            sae_fp = ((y[sae_mask] == 0) & (pred[sae_mask] == 1)).sum()
            sae_tn = ((y[sae_mask] == 0) & (pred[sae_mask] == 0)).sum()
            sae_fpr = sae_fp / (sae_fp + sae_tn + 1e-8)

            fpr_gap = abs(aae_fpr - sae_fpr)

            if fpr_gap < best_fpr_gap:
                best_fpr_gap = fpr_gap
                best_t_aae = t_aae
                best_t_sae = t_sae
                best_metrics = {
                    "t_aae": t_aae,
                    "t_sae": t_sae,
                    "f1": curr_f1,
                    "fpr_gap": fpr_gap,
                    "aae_fpr": aae_fpr,
                    "sae_fpr": sae_fpr,
                }
    
    return best_t_aae, best_t_sae, best_metrics


def apply_thresholds(probs: np.ndarray, g: np.ndarray, t_aae: float, t_sae: float) -> np.ndarray:
    pred = np.zeros(len(probs), dtype=int)
    pred[g == 1] = (probs[g == 1] >= t_aae).astype(int)
    pred[g == 0] = (probs[g == 0] >= t_sae).astype(int)
    return pred


def train_one_adv_model(
    X_train, y_train, g_train,
    X_val, y_val, g_val,
    params,
    lambda_adv=0.1,
    adv_c=1.0,
    num_round=100,
    dialect_col="dialect_strict",
    use_leaf_adv=True,
    use_reweighting=True,
):
    sample_weight = compute_group_weights(g_train) if use_reweighting else None

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = None
    leaf_adv_clf = None
    adv_w, adv_b = 0.0, 0.0
    
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
        if booster is not None:
            if use_leaf_adv:
                leaf_adv_clf, leaf_sparse = fit_leaf_adversary(
                    booster, dtrain, g_train, adv_c=adv_c, sample_weight=sample_weight
                )
            else:
                raw_train = booster.predict(dtrain, output_margin=True)
                adv_w, adv_b = fit_margin_adversary(
                    raw_train, g_train, adv_c=adv_c, sample_weight=sample_weight
                )

        def objective(preds, dmat):
            y = dmat.get_label()

            p_tox = sigmoid(preds)
            grad_tox = p_tox - y
            hess_tox = p_tox * (1.0 - p_tox)

            if use_leaf_adv and leaf_adv_clf is not None:
                adv_logits = leaf_adv_clf.decision_function(leaf_sparse)
                p_adv = sigmoid(adv_logits)
                mean_w = np.abs(leaf_adv_clf.coef_).mean() + 1e-8
                grad_adv = (p_adv - g_train) * mean_w
                hess_adv = p_adv * (1.0 - p_adv) * (mean_w ** 2)
            elif adv_w != 0.0:
                adv_logit = adv_w * preds + adv_b
                p_adv = sigmoid(adv_logit)
                grad_adv = (p_adv - g_train) * adv_w
                hess_adv = p_adv * (1.0 - p_adv) * (adv_w ** 2)
            else:
                grad_adv = np.zeros_like(preds)
                hess_adv = np.zeros_like(preds)

            grad = grad_tox - lambda_adv * grad_adv
            hess = hess_tox + lambda_adv * hess_adv
            hess = np.clip(hess, 1e-6, None)
            return grad, hess

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1,
            obj=objective,
            xgb_model=booster,
            verbose_eval=False
        )

        if use_leaf_adv and booster is not None:
            leaf_adv_clf, leaf_sparse = fit_leaf_adversary(
                booster, dtrain, g_train, adv_c=adv_c, sample_weight=sample_weight
            )

        train_raw = booster.predict(dtrain, output_margin=True)
        val_raw = booster.predict(dval, output_margin=True)

        train_prob = sigmoid(train_raw)
        val_prob = sigmoid(val_raw)

        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)

        history["train_loss"].append(float(log_loss(y_train, train_prob, labels=[0, 1])))
        history["val_loss"].append(float(log_loss(y_val, val_prob, labels=[0, 1])))

        if use_leaf_adv and leaf_adv_clf is not None:
            adv_train_prob = sigmoid(leaf_adv_clf.decision_function(leaf_sparse))
            adv_train_pred = (adv_train_prob >= 0.5).astype(int)
        elif adv_w != 0.0:
            adv_train_prob = sigmoid(adv_w * train_raw + adv_b)
            adv_train_pred = (adv_train_prob >= 0.5).astype(int)
        else:
            adv_train_prob = np.full_like(g_train, 0.5, dtype=float)
            adv_train_pred = np.zeros_like(g_train)
        
        history["adv_acc_train"].append(float(accuracy_score(g_train, adv_train_pred)))
        history["adv_loss_train"].append(float(log_loss(g_train, np.clip(adv_train_prob, 1e-7, 1-1e-7), labels=[0, 1])))

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

    return booster, history


def predict_with_booster(booster, X, g, lambda_adv=0.0, adv_w=0.0, adv_b=0.0):
    dmat = xgb.DMatrix(X)
    raw = booster.predict(dmat, output_margin=True)
    prob = sigmoid(raw)
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def main():
    parser = argparse.ArgumentParser(description="Adversarial Debiasing for XGBoost")
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
    parser.add_argument("--tree_method", default="hist",
                        choices=["hist", "approx", "exact", "auto"])
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"])

    parser.add_argument("--lambda_grid", default="0.0,0.05,0.1,0.25,0.5")
    parser.add_argument("--adv_c_grid", default="0.5,1.0,2.0")

    parser.add_argument("--use_leaf_adv", action="store_true", default=True)
    parser.add_argument("--no_leaf_adv", action="store_false", dest="use_leaf_adv")
    parser.add_argument("--use_reweighting", action="store_true", default=True)
    parser.add_argument("--no_reweighting", action="store_false", dest="use_reweighting")
    parser.add_argument("--use_threshold_tuning", action="store_true", default=True)
    parser.add_argument("--no_threshold_tuning", action="store_false", dest="use_threshold_tuning")
    parser.add_argument("--threshold_f1_ratio", type=float, default=0.95)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Adversarial XGBoost with Multi-Technique Debiasing")
    print("=" * 60)
    print(f"  Leaf-based adversary:   {args.use_leaf_adv}")
    print(f"  Sample reweighting:     {args.use_reweighting}")
    print(f"  Threshold tuning:       {args.use_threshold_tuning}")
    print("=" * 60 + "\n")

    train_df, X_train, y_train, g_train = load_split(args.train_csv, args.train_emb, args.dialect_col)
    val_df, X_val, y_val, g_val = load_split(args.val_csv, args.val_emb, args.dialect_col)
    test_df, X_test, y_test, g_test = load_split(args.test_csv, args.test_emb, args.dialect_col)

    print(f"Train: {len(train_df)} samples (AAE: {(g_train==1).sum()}, SAE: {(g_train==0).sum()})")
    print(f"Val:   {len(val_df)} samples (AAE: {(g_val==1).sum()}, SAE: {(g_val==0).sum()})")
    print(f"Test:  {len(test_df)} samples (AAE: {(g_test==1).sum()}, SAE: {(g_test==0).sum()})")

    lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    adv_c_grid = [float(x) for x in args.adv_c_grid.split(",")]

    base_params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method,
        "device": args.device,
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
    best_thresholds = (0.5, 0.5)                  

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
            dialect_col=args.dialect_col,
            use_leaf_adv=args.use_leaf_adv,
            use_reweighting=args.use_reweighting,
        )

        val_raw = booster.predict(xgb.DMatrix(X_val), output_margin=True)
        val_prob = sigmoid(val_raw)
        
        if args.use_threshold_tuning:
            t_aae, t_sae, thresh_metrics = tune_thresholds(
                val_prob, y_val, g_val,
                dialect_col=args.dialect_col,
                min_f1_ratio=args.threshold_f1_ratio
            )
            val_pred = apply_thresholds(val_prob, g_val, t_aae, t_sae)
        else:
            t_aae, t_sae = 0.5, 0.5
            val_pred = (val_prob >= 0.5).astype(int)

        val_eval = val_df.copy()
        val_eval["adv_pred"] = val_pred

        m = compute_group_metrics(val_eval, "adv_pred", args.dialect_col)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)

        score = (m["FPR_gap"], -val_f1, -val_acc)

        summary_rows.append({
            "lambda_adv": lam,
            "adv_c": adv_c,
            "t_aae": t_aae,
            "t_sae": t_sae,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_AAE_FPR": m["AAE"]["FPR"],
            "val_SAE_FPR": m["SAE"]["FPR"],
            "val_AAE_FNR": m["AAE"]["FNR"],
            "val_SAE_FNR": m["SAE"]["FNR"],
            "val_AAE_TPR": m["AAE"]["TPR"],
            "val_SAE_TPR": m["SAE"]["TPR"],
            "val_AAE_acc": (m["AAE"]["TP"] + m["AAE"]["TN"]) / m["AAE"]["N"],
            "val_SAE_acc": (m["SAE"]["TP"] + m["SAE"]["TN"]) / m["SAE"]["N"],
            "val_FPR_gap": m["FPR_gap"],
            "val_FNR_gap": m["FNR_gap"],
            "val_TPR_gap": m["TPR_gap"],
            "val_DIfav": m["DIfav"],
            "val_DIunfav": m["DIunfav"],
        })

        print(
            f"lambda={lam:.3f}, adv_c={adv_c:.3f}, t_aae={t_aae:.2f}, t_sae={t_sae:.2f} | "
            f"val_f1={val_f1:.4f}, FPR_gap={m['FPR_gap']:.4f}, DIunfav={m['DIunfav']:.4f}"
        )

        if best_score is None or score < best_score:
            best_score = score
            best = cand
            best_booster = booster
            best_history = history
            best_thresholds = (t_aae, t_sae)

    print(f"\nBest candidate: {best}")
    print(f"Best thresholds: t_aae={best_thresholds[0]:.3f}, t_sae={best_thresholds[1]:.3f}")

    test_raw = best_booster.predict(xgb.DMatrix(X_test), output_margin=True)
    test_prob = sigmoid(test_raw)

    if args.use_threshold_tuning:
        test_pred = apply_thresholds(test_prob, g_test, best_thresholds[0], best_thresholds[1])
    else:
        test_pred = (test_prob >= 0.5).astype(int)

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

    tuning_df = pd.DataFrame(summary_rows)
    tuning_df.to_csv(os.path.join(args.out_dir, "adv_xgb_tuning.csv"), index=False)

    with open(os.path.join(args.out_dir, "loss_history.json"), "w") as f:
        json.dump(best_history, f, indent=2)

    summary_path = os.path.join(args.out_dir, "adv_xgb_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Adversarial XGBoost Summary\n")
        f.write("==========================\n\n")
        f.write("Techniques Used:\n")
        f.write(f"  - Leaf-based adversary: {args.use_leaf_adv}\n")
        f.write(f"  - Sample reweighting:   {args.use_reweighting}\n")
        f.write(f"  - Threshold tuning:     {args.use_threshold_tuning}\n\n")
        f.write(f"Best candidate: {best}\n")
        f.write(f"Best thresholds: t_aae={best_thresholds[0]:.3f}, t_sae={best_thresholds[1]:.3f}\n\n")
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

    thresh_path = os.path.join(args.out_dir, "thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump({
            "t_aae": best_thresholds[0],
            "t_sae": best_thresholds[1],
            "use_threshold_tuning": args.use_threshold_tuning,
        }, f, indent=2)

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved tuning table to: {os.path.join(args.out_dir, 'adv_xgb_tuning.csv')}")
    print(f"Saved history to: {os.path.join(args.out_dir, 'loss_history.json')}")
    print(f"Saved thresholds to: {thresh_path}")
    print(f"Saved summary to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1:       {test_f1:.4f}")
    print(f"AAE FPR:       {test_eval['AAE']['FPR']:.4f}")
    print(f"SAE FPR:       {test_eval['SAE']['FPR']:.4f}")
    print(f"FPR Gap:       {test_eval['FPR_gap']:.4f}")
    print(f"FNR Gap:       {test_eval['FNR_gap']:.4f}")
    print(f"DIfav:         {test_eval['DIfav']:.4f}")
    print(f"DIunfav:       {test_eval['DIunfav']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()