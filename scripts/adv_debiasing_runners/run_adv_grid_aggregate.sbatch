#!/bin/bash
#SBATCH --job-name=adv_grid_agg
#SBATCH --output=logs/adv_grid_agg_%j.out
#SBATCH --error=logs/adv_grid_agg_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -eo pipefail
module purge
module load conda
if [ -f /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate ]; then
    source /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate
elif [ -f /scratch1/aqliang/haluguard-env/bin/activate ]; then
    source /scratch1/aqliang/haluguard-env/bin/activate
fi

cd /scratch1/aqliang/CSCI567-ML-Project
OUT_DIR="data/results/adv_xgb_largegrid"

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
df = df.sort_values(["lambda_adv", "adv_c"]).reset_index(drop=True)

# Composite score on test: prioritize low FPR + balance
df["test_score"] = (
    1.5 * df["test_mean_FPR"]
    + 1.0 * df["test_abs_balance"]
    + 0.5 * (df["test_FPR_gap"] + df["test_FNR_gap"])
    - 0.5 * df["test_f1"]
    + df["test_f1"].lt(0.70).astype(float) * 10.0
)

# Val score for selection (honest)
df["val_score"] = (
    1.5 * df["val_mean_FPR"]
    + 1.0 * df["val_abs_balance"]
    + 0.5 * (df["val_FPR_gap"] + df["val_FNR_gap"])
    - 0.5 * df["val_f1"]
    + df["val_f1"].lt(0.70).astype(float) * 10.0
)

df.to_csv(os.path.join(out_dir, "grid_summary.csv"), index=False)
print(f"Saved full summary: {out_dir}/grid_summary.csv ({len(df)} cells)")

cols = ["lambda_adv","adv_c","val_f1","val_FPR_gap","val_mean_FPR","val_score",
        "test_f1","test_AAE_FPR","test_SAE_FPR","test_AAE_FNR","test_SAE_FNR",
        "test_FPR_gap","test_FNR_gap","test_mean_FPR","test_mean_FNR","test_score"]

print("\n=== FULL GRID (sorted by test_score, lower = better) ===")
print(df.sort_values("test_score")[cols].round(4).to_string(index=False))

print("\n=== TOP 5 BY VAL SCORE (model-selection honest) ===")
print(df.sort_values("val_score").head(5)[cols].round(4).to_string(index=False))

print("\n=== TOP 5 BY VAL FPR_GAP ===")
print(df.sort_values("val_FPR_gap").head(5)[cols].round(4).to_string(index=False))

print("\n=== TOP 5 BY VAL F1 ===")
print(df.sort_values("val_f1", ascending=False).head(5)[cols].round(4).to_string(index=False))

best = df.sort_values("val_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_val_score.json"), "w") as f:
    json.dump(best, f, indent=2, default=str)
print(f"\n>>> BEST by val_score: lambda={best['lambda_adv']} adv_c={best['adv_c']}")
print(f"    test F1={best['test_f1']:.4f}  mean_FPR={best['test_mean_FPR']:.4f}")
print(f"    AAE FPR={best['test_AAE_FPR']:.4f}  SAE FPR={best['test_SAE_FPR']:.4f}")
print(f"    AAE FNR={best['test_AAE_FNR']:.4f}  SAE FNR={best['test_SAE_FNR']:.4f}")
PYEOF
