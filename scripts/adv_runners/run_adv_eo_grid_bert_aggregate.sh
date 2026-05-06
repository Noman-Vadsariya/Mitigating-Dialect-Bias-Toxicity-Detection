#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
DATA_TAG="${DATA_TAG:-unbalanced}"
OUT_DIR="data/results/adv_xgb_eo_grid_bert_${DATA_TAG}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python - <<PYEOF
import glob, json, os
import pandas as pd

out_dir = "$OUT_DIR"
frames = []
for p in sorted(glob.glob(os.path.join(out_dir, "lam_*_grid.csv"))):
    frames.append(pd.read_csv(p))
if not frames:
    raise SystemExit("No per-lambda CSVs found")

df = pd.concat(frames, ignore_index=True)
df = df.sort_values(
    ["lambda_adv", "adv_hidden", "adv_lr", "min_f1_ratio"]
).reset_index(drop=True)

def score(prefix):
    return (
        1.5 * df[f"{prefix}_mean_FPR"]
        + 1.0 * df[f"{prefix}_abs_balance"]
        + 0.5 * (df[f"{prefix}_FPR_gap"] + df[f"{prefix}_FNR_gap"])
        - 0.5 * df[f"{prefix}_f1"]
        + df[f"{prefix}_f1"].lt(0.70).astype(float) * 10.0
    )

df["val_score"]  = score("val")
df["test_score"] = score("test")
df.to_csv(os.path.join(out_dir, "grid_summary.csv"), index=False)
print(f"Saved: {out_dir}/grid_summary.csv ({len(df)} rows)")

cols = ["lambda_adv","adv_hidden","adv_lr","min_f1_ratio","t_aae","t_sae",
        "val_f1","val_FPR_gap","val_score",
        "test_f1","test_AAE_FPR","test_SAE_FPR","test_FPR_gap","test_FNR_gap",
        "test_mean_FPR","test_abs_balance","test_score"]

print("\n=== TOP 10 BY VAL SCORE ===")
print(df.sort_values("val_score").head(10)[cols].round(4).to_string(index=False))
print("\n=== TOP 10 BY TEST SCORE (oracle) ===")
print(df.sort_values("test_score").head(10)[cols].round(4).to_string(index=False))

best = df.sort_values("val_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_val_score.json"), "w") as f:
    json.dump(best, f, indent=2, default=str)
print(f"\n>>> BEST by val_score: lam={best['lambda_adv']} hidden={best['adv_hidden']} "
      f"lr={best['adv_lr']} ratio={best['min_f1_ratio']}")
print(f"    t_AAE={best['t_aae']:.4f}  t_SAE={best['t_sae']:.4f}")
print(f"    test F1={best['test_f1']:.4f}  FPR_gap={best['test_FPR_gap']:.4f}  "
      f"mean_FPR={best['test_mean_FPR']:.4f}  abs_bal={best['test_abs_balance']:.4f}")

best_t = df.sort_values("test_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_test_score.json"), "w") as f:
    json.dump(best_t, f, indent=2, default=str)

def ff(d, k):
    try: return float(d.get(k, float("nan")))
    except Exception: return float("nan")

L = []
L.append("Adversarial XGBoost (EO) Summary  --  BERT-MLP adversary")
L.append(f"Dataset: TwitterAAE ${DATA_TAG}")
L.append("=" * 64)
L.append("")
L.append("Best candidate (selected by val_score):")
L.append(f"  lambda_adv = {best['lambda_adv']}")
L.append(f"  adv_hidden = {best['adv_hidden']}    adv_lr = {best['adv_lr']}")
L.append(f"  min_f1_ratio = {best['min_f1_ratio']}")
L.append(f"  thresholds = (t_AAE={ff(best,'t_aae'):.4f}, t_SAE={ff(best,'t_sae'):.4f})")
L.append("")
for split in ("val", "test"):
    L.append(f"--- {split.title()} ---")
    L.append(f"{split.title()} Accuracy: {ff(best, f'{split}_accuracy'):.4f}")
    L.append(f"{split.title()} F1:       {ff(best, f'{split}_f1'):.4f}")
    L.append(f"AAE FPR: {ff(best, f'{split}_AAE_FPR'):.4f}   SAE FPR: {ff(best, f'{split}_SAE_FPR'):.4f}   FPR Gap: {ff(best, f'{split}_FPR_gap'):.4f}")
    L.append(f"AAE FNR: {ff(best, f'{split}_AAE_FNR'):.4f}   SAE FNR: {ff(best, f'{split}_SAE_FNR'):.4f}   FNR Gap: {ff(best, f'{split}_FNR_gap'):.4f}")
    L.append(f"Mean FPR: {ff(best, f'{split}_mean_FPR'):.4f}   Mean FNR: {ff(best, f'{split}_mean_FNR'):.4f}   |EO balance|: {ff(best, f'{split}_abs_balance'):.4f}")
    L.append("")
L.append("--- Oracle best (selected by test_score, for reference) ---")
L.append(f"  lambda_adv={best_t['lambda_adv']} hidden={best_t['adv_hidden']} lr={best_t['adv_lr']} ratio={best_t['min_f1_ratio']}")
L.append(f"  Test F1: {ff(best_t,'test_f1'):.4f}  FPR Gap: {ff(best_t,'test_FPR_gap'):.4f}  Mean FPR: {ff(best_t,'test_mean_FPR'):.4f}")

txt = "\n".join(L) + "\n"
path = os.path.join(out_dir, "adv_xgb_summary.txt")
with open(path, "w") as f: f.write(txt)
print(f"\nWrote {path}")
print(txt)
PYEOF
