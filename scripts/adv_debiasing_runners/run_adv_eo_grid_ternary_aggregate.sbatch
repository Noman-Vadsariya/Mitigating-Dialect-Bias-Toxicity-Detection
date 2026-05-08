#!/bin/bash
#SBATCH --job-name=adv_eo_agg_tern
#SBATCH --output=logs/adv_eo_agg_tern_%j.out
#SBATCH --error=logs/adv_eo_agg_tern_%j.err
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
OUT_DIR="data/results/adv_xgb_eo_grid_ternary"

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
df = df.sort_values(["lambda_adv", "adv_c", "min_f1_ratio"]).reset_index(drop=True)

def score(prefix):
    # Penalize macro-FPR + abs_balance + 0.5*(mean class FPR_gap + FNR_gap)
    # minus 0.5*macro-F1, with hard penalty if macro-F1 < 0.50.
    return (
        1.5 * df[f"{prefix}_mean_FPR"]
        + 1.0 * df[f"{prefix}_abs_balance"]
        + 0.5 * (df[f"{prefix}_mean_FPR_gap"] + df[f"{prefix}_mean_FNR_gap"])
        - 0.5 * df[f"{prefix}_f1_macro"]
        + df[f"{prefix}_f1_macro"].lt(0.50).astype(float) * 10.0
    )

df["val_score"]  = score("val")
df["test_score"] = score("test")
df.to_csv(os.path.join(out_dir, "grid_summary.csv"), index=False)
print(f"Saved: {out_dir}/grid_summary.csv ({len(df)} rows)")

show_cols = [
    "lambda_adv","adv_c","min_f1_ratio",
    "val_f1_macro","val_mean_FPR_gap","val_mean_FNR_gap","val_score",
    "test_f1_macro","test_mean_FPR_gap","test_mean_FNR_gap",
    "test_AAE_mean_FPR","test_SAE_mean_FPR","test_score",
]

print("\n=== TOP 10 BY VAL SCORE ===")
print(df.sort_values("val_score").head(10)[show_cols].round(4).to_string(index=False))

print("\n=== TOP 10 BY TEST SCORE (oracle) ===")
print(df.sort_values("test_score").head(10)[show_cols].round(4).to_string(index=False))

mask = df["test_f1_macro"] >= 0.55
print("\n=== TOP 10 BY TEST mean_FPR_gap (f1_macro>=0.55) ===")
if mask.any():
    print(df[mask].sort_values("test_mean_FPR_gap").head(10)[show_cols].round(4).to_string(index=False))

best = df.sort_values("val_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_val_score.json"), "w") as f:
    json.dump(best, f, indent=2, default=str)
print(f"\n>>> BEST by val_score: lam={best['lambda_adv']} c={best['adv_c']} ratio={best['min_f1_ratio']}")
print(f"    test f1_macro={best['test_f1_macro']:.4f}  mean_FPR={best['test_mean_FPR']:.4f}  "
      f"mean_FPR_gap={best['test_mean_FPR_gap']:.4f}  mean_FNR_gap={best['test_mean_FNR_gap']:.4f}")

best_t = df.sort_values("test_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_test_score.json"), "w") as f:
    json.dump(best_t, f, indent=2, default=str)
print(f"\n>>> ORACLE BEST by test_score: lam={best_t['lambda_adv']} c={best_t['adv_c']} ratio={best_t['min_f1_ratio']}")
print(f"    test f1_macro={best_t['test_f1_macro']:.4f}  mean_FPR={best_t['test_mean_FPR']:.4f}  "
      f"mean_FPR_gap={best_t['test_mean_FPR_gap']:.4f}  mean_FNR_gap={best_t['test_mean_FNR_gap']:.4f}")
PYEOF
