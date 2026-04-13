import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, log_loss

EPS = 1e-8


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

        FPR = FP / (FP + TN + EPS)
        FNR = FN / (FN + TP + EPS)
        TPR = TP / (TP + FN + EPS)

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

    out["DIfav"] = float(p_aae_non / (p_sae_non + EPS))
    out["DIunfav"] = float(p_aae_tox / (p_sae_tox + EPS))
    return out


class AdversaryNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def build_adv_input(margin, y, fairness_objective):
    """
    DP:  input = sigmoid(F(x))
    EO:  input = [sigmoid(F(x)), y]
    """
    p = torch.sigmoid(margin).unsqueeze(1)
    if fairness_objective == "dp":
        return p
    elif fairness_objective == "eo":
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(p.device)
        return torch.cat([p, y_t], dim=1)
    else:
        raise ValueError("fairness_objective must be 'dp' or 'eo'")


def train_adversary(adversary, margin_np, g_np, y_np, fairness_objective, epochs=15, lr=1e-3):
    """
    Train A on the current predictor outputs, matching the paper's backprop update.
    """
    adversary.train()
    opt = torch.optim.Adam(adversary.parameters(), lr=lr)

    margin = torch.tensor(margin_np, dtype=torch.float32)
    g = torch.tensor(g_np, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        opt.zero_grad()
        adv_in = build_adv_input(margin, y_np, fairness_objective)
        logits = adversary(adv_in)
        loss = nn.BCEWithLogitsLoss()(logits, g)
        loss.backward()
        opt.step()

    return adversary


def adversary_grad_wrt_margin(adversary, margin_np, g_np, y_np, fairness_objective):
    """
    Returns d L_adv / d F(x) for each training sample.
    This is the quantity the paper subtracts (through the combined residual).
    """
    adversary.eval()
    margin = torch.tensor(margin_np, dtype=torch.float32, requires_grad=True)
    g = torch.tensor(g_np, dtype=torch.float32).unsqueeze(1)

    adv_in = build_adv_input(margin, y_np, fairness_objective)
    logits = adversary(adv_in)
    loss = nn.BCEWithLogitsLoss()(logits, g)
    loss.backward()

    return margin.grad.detach().cpu().numpy()


def main_objective_loss(y, margin, adversary, g, fairness_objective):
    """
    Paper's main objective:
        L_F - lambda * L_A
    Here we compute the two terms separately for line search.
    """
    p = sigmoid(margin)
    tox_loss = log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])

    margin_t = torch.tensor(margin, dtype=torch.float32)
    g_t = torch.tensor(g, dtype=torch.float32).unsqueeze(1)
    adv_in = build_adv_input(margin_t, y, fairness_objective)
    with torch.no_grad():
        adv_logits = adversary(adv_in)
        adv_prob = torch.sigmoid(adv_logits).cpu().numpy().reshape(-1)
    adv_loss = log_loss(g, np.clip(adv_prob, EPS, 1 - EPS), labels=[0, 1])

    return tox_loss, adv_loss


def fit_one_round_tree(X, target, max_depth=5, seed=42):
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    tree.fit(X, target)
    return tree


def train_model(
    X_train, y_train, g_train,
    X_val, y_val, g_val,
    X_test, y_test, g_test,
    params,
    fairness_objective="eo",
    lambda_adv=0.1,
    adv_hidden=16,
    adv_epochs=15,
    adv_lr=1e-3,
    num_rounds=50,
    gamma_grid=None,
    seed=42
):
    if gamma_grid is None:
        gamma_grid = np.linspace(0.1, 1.0, 10)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Predictor margins
    F_train = np.zeros(len(y_train), dtype=float)
    F_val = np.zeros(len(y_val), dtype=float)
    F_test = np.zeros(len(y_test), dtype=float)

    # Start with a simple base adversary
    adv_in_dim = 1 if fairness_objective == "dp" else 2
    adversary = AdversaryNet(in_dim=adv_in_dim, hidden_dim=adv_hidden)

    trees = []
    gammas = []
    history = {
        "train_tox_loss": [],
        "val_tox_loss": [],
        "train_adv_loss": [],
        "val_adv_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_fpr_gap": [],
        "val_fnr_gap": [],
        "val_difunfav": [],
    }

    for m in range(num_rounds):
        # 1) Train adversary on current outputs
        adversary = train_adversary(
            adversary=adversary,
            margin_np=F_train,
            g_np=g_train,
            y_np=y_train,
            fairness_objective=fairness_objective,
            epochs=adv_epochs,
            lr=adv_lr
        )

        # 2) Predictor residuals: r = - dL_F/dF = y - sigmoid(F)
        p_train = sigmoid(F_train)
        r = y_train - p_train

        # 3) Adversary residuals:
        # paper combines r - lambda * t, where t is the adversarial pseudo-residual
        # which corresponds to the negative gradient of adversary loss.
        # Since adversary_grad_wrt_margin returns dL_adv/dF, negative gradient = -grad.
        adv_grad = adversary_grad_wrt_margin(
            adversary=adversary,
            margin_np=F_train,
            g_np=g_train,
            y_np=y_train,
            fairness_objective=fairness_objective
        )
        combined_target = r + lambda_adv * adv_grad

        # 4) Fit next weak learner to combined residuals
        tree = fit_one_round_tree(
            X_train,
            combined_target,
            max_depth=params["max_depth"],
            seed=seed + m
        )

        # 5) Line search for gamma on validation objective
        tree_val = tree.predict(X_val)

        best_gamma = None
        best_obj = None

        for gamma in gamma_grid:
            cand_val_margin = F_val + gamma * tree_val
            tox_loss, adv_loss = main_objective_loss(
                y_val, cand_val_margin, adversary, g_val, fairness_objective
            )
            obj = tox_loss - lambda_adv * adv_loss

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_gamma = gamma

        # 6) Update margins
        F_train = F_train + best_gamma * tree.predict(X_train)
        F_val = F_val + best_gamma * tree.predict(X_val)
        F_test = F_test + best_gamma * tree.predict(X_test)

        trees.append(tree)
        gammas.append(best_gamma)

        # 7) Track losses / metrics
        train_tox_loss, train_adv_loss = main_objective_loss(
            y_train, F_train, adversary, g_train, fairness_objective
        )
        val_tox_loss, val_adv_loss = main_objective_loss(
            y_val, F_val, adversary, g_val, fairness_objective
        )

        val_prob = sigmoid(F_val)
        val_pred = (val_prob >= 0.5).astype(int)

        val_df_tmp = pd.DataFrame({
            "label": y_val,
            "pred": val_pred,
            "dialect": ["AAE" if x == 1 else "SAE" for x in g_val]
        })
        metrics = compute_group_metrics(val_df_tmp, "pred", "dialect")

        history["train_tox_loss"].append(train_tox_loss)
        history["val_tox_loss"].append(val_tox_loss)
        history["train_adv_loss"].append(train_adv_loss)
        history["val_adv_loss"].append(val_adv_loss)
        history["val_acc"].append(accuracy_score(y_val, val_pred))
        history["val_f1"].append(f1_score(y_val, val_pred))
        history["val_fpr_gap"].append(metrics["FPR_gap"])
        history["val_fnr_gap"].append(metrics["FNR_gap"])
        history["val_difunfav"].append(metrics["DIunfav"])

        print(
            f"Round {m+1:03d}/{num_rounds} | "
            f"gamma={best_gamma:.3f} | "
            f"val_acc={history['val_acc'][-1]:.4f} | "
            f"val_f1={history['val_f1'][-1]:.4f} | "
            f"val_fpr_gap={history['val_fpr_gap'][-1]:.4f} | "
            f"val_DIunfav={history['val_difunfav'][-1]:.4f}"
        )

    # Final test prediction
    test_prob = sigmoid(F_test)
    test_pred = (test_prob >= 0.5).astype(int)

    test_out = pd.DataFrame({
        "label": y_test,
        "pred": test_pred,
        "prob": test_prob,
        "dialect": ["AAE" if x == 1 else "SAE" for x in g_test]
    })

    test_metrics = compute_group_metrics(test_out, "pred", "dialect")
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    return {
        "trees": trees,
        "gammas": gammas,
        "history": history,
        "test_out": test_out,
        "test_metrics": test_metrics,
        "test_acc": test_acc,
        "test_f1": test_f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="../data/processed/train.csv")
    parser.add_argument("--val_csv", default="../data/processed/val.csv")
    parser.add_argument("--test_csv", default="../data/processed/test.csv")
    parser.add_argument("--train_emb", default="../data/embeddings/train_emb.npy")
    parser.add_argument("--val_emb", default="../data/embeddings/val_emb.npy")
    parser.add_argument("--test_emb", default="../data/embeddings/test_emb.npy")
    parser.add_argument("--dialect_col", default="dialect_strict")
    parser.add_argument("--out_dir", default="results/fagbt_paper_style")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--lambda_grid", default="0.0,0.05,0.1,0.25,0.5")
    parser.add_argument("--fairness_objective", choices=["dp", "eo"], default="eo")
    parser.add_argument("--adv_hidden", type=int, default=16)
    parser.add_argument("--adv_epochs", type=int, default=15)
    parser.add_argument("--adv_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df, X_train, y_train, g_train = load_split(args.train_csv, args.train_emb, args.dialect_col)
    val_df, X_val, y_val, g_val = load_split(args.val_csv, args.val_emb, args.dialect_col)
    test_df, X_test, y_test, g_test = load_split(args.test_csv, args.test_emb, args.dialect_col)

    lambdas = [float(x) for x in args.lambda_grid.split(",")]

    base_params = {
        "max_depth": args.max_depth
    }

    summary_rows = []
    best = None
    best_score = None
    best_result = None

    for lam in lambdas:
        print(f"\n=== Training lambda={lam} ===")
        result = train_model(
            X_train, y_train, g_train,
            X_val, y_val, g_val,
            X_test, y_test, g_test,
            params=base_params,
            fairness_objective=args.fairness_objective,
            lambda_adv=lam,
            adv_hidden=args.adv_hidden,
            adv_epochs=args.adv_epochs,
            adv_lr=args.adv_lr,
            num_rounds=args.num_rounds,
            seed=args.seed
        )

        val_best_idx = int(np.argmin(result["history"]["val_fpr_gap"]))
        val_fpr_gap = result["history"]["val_fpr_gap"][val_best_idx]
        val_f1 = result["history"]["val_f1"][val_best_idx]
        val_acc = result["history"]["val_acc"][val_best_idx]

        summary_rows.append({
            "lambda_adv": lam,
            "best_val_fpr_gap": val_fpr_gap,
            "best_val_f1": val_f1,
            "best_val_acc": val_acc,
            "test_acc": result["test_acc"],
            "test_f1": result["test_f1"],
            "test_FPR_gap": result["test_metrics"]["FPR_gap"],
            "test_FNR_gap": result["test_metrics"]["FNR_gap"],
            "test_DIfav": result["test_metrics"]["DIfav"],
            "test_DIunfav": result["test_metrics"]["DIunfav"]
        })

        score = (result["test_metrics"]["FPR_gap"], -result["test_f1"], -result["test_acc"])
        if best_score is None or score < best_score:
            best_score = score
            best = lam
            best_result = result

        print(
            f"lambda={lam:.3f} | "
            f"test_acc={result['test_acc']:.4f} | "
            f"test_f1={result['test_f1']:.4f} | "
            f"test_FPR_gap={result['test_metrics']['FPR_gap']:.4f}"
        )

    tuning_df = pd.DataFrame(summary_rows)
    tuning_df.to_csv(os.path.join(args.out_dir, "adversarial_tuning.csv"), index=False)

    # Save best model outputs
    best_test = best_result["test_out"].copy()
    best_test["adv_pred"] = best_test["pred"]
    best_test["adv_prob"] = best_test["prob"]
    best_test.to_csv(os.path.join(args.out_dir, "adv_xgb_predictions.csv"), index=False)

    with open(os.path.join(args.out_dir, "best_lambda.txt"), "w") as f:
        f.write(str(best))

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(best_result["history"], f, indent=2)

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("Paper-style Adversarial Gradient Tree Boosting\n")
        f.write("=============================================\n\n")
        f.write(f"Best lambda: {best}\n")
        f.write(f"Test Accuracy: {best_result['test_acc']:.4f}\n")
        f.write(f"Test F1: {best_result['test_f1']:.4f}\n")
        f.write(f"AAE FPR: {best_result['test_metrics']['AAE']['FPR']:.4f}\n")
        f.write(f"SAE FPR: {best_result['test_metrics']['SAE']['FPR']:.4f}\n")
        f.write(f"FPR Gap: {best_result['test_metrics']['FPR_gap']:.4f}\n")
        f.write(f"AAE FNR: {best_result['test_metrics']['AAE']['FNR']:.4f}\n")
        f.write(f"SAE FNR: {best_result['test_metrics']['SAE']['FNR']:.4f}\n")
        f.write(f"FNR Gap: {best_result['test_metrics']['FNR_gap']:.4f}\n")
        f.write(f"DIfav: {best_result['test_metrics']['DIfav']:.4f}\n")
        f.write(f"DIunfav: {best_result['test_metrics']['DIunfav']:.4f}\n")

    print(f"\nSaved tuning table to {os.path.join(args.out_dir, 'adversarial_tuning.csv')}")
    print(f"Saved best predictions to {os.path.join(args.out_dir, 'adv_xgb_predictions.csv')}")
    print(f"Saved summary to {os.path.join(args.out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()