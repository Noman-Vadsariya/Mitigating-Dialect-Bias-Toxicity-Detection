#!/bin/bash
#SBATCH --job-name=fairness_agg
#SBATCH --output=logs/fairness_agg_%j.out
#SBATCH --error=logs/fairness_agg_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -eo pipefail

echo "=============================================="
echo "Aggregating ablation results"
echo "=============================================="

module purge
module load conda

if [ -f /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate ]; then
    source /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate
elif [ -f /scratch1/aqliang/haluguard-env/bin/activate ]; then
    source /scratch1/aqliang/haluguard-env/bin/activate
fi

cd /scratch1/aqliang/CSCI567-ML-Project
OUT_DIR="data/results/fairness_ablation"

python - <<PYEOF
import json, os, glob
import pandas as pd

out_dir = "$OUT_DIR"
rows = []
for p in sorted(glob.glob(os.path.join(out_dir, "*_metrics.json"))):
    with open(p) as f:
        rows.append(json.load(f))

if not rows:
    print("ERROR: No metrics files found.")
    raise SystemExit(1)

df = pd.DataFrame(rows)

# Composite score (lower = better) prioritizing:
#   1. Low absolute mean FPR (weight 1.5) -- primary
#   2. Balanced mean |FPR - FNR|          (weight 1.0)
#   3. Small group gaps (FPR_gap+FNR_gap) (weight 0.5)
#   4. High F1                             (weight -0.5)
# F1 floor of 0.70 to avoid degenerate all-negative / all-positive predictors.
df["mean_FPR"] = (df["AAE_FPR"] + df["SAE_FPR"]) / 2.0
df["mean_FNR"] = (df["AAE_FNR"] + df["SAE_FNR"]) / 2.0
df["abs_balance"] = (df["mean_FPR"] - df["mean_FNR"]).abs()

F1_FLOOR = 0.70
df["_score"] = (
    1.5 * df["mean_FPR"]
    + 1.0 * df["abs_balance"]
    + 0.5 * (df["FPR_gap"] + df["FNR_gap"])
    - 0.5 * df["f1"]
    + df["f1"].lt(F1_FLOOR).astype(float) * 10.0
)
df = df.sort_values("_score")

csv_path = os.path.join(out_dir, "ablation_summary.csv")
df.to_csv(csv_path, index=False)
print(f"Ablation summary saved: {csv_path}")
print(f"Total runs: {len(df)}")
print(f"\nScoring: 1.5*mean_FPR + 1.0*|mean_FPR - mean_FNR| + 0.5*(FPR_gap+FNR_gap) - 0.5*F1")
print(f"F1 floor: {F1_FLOOR} (penalty 10 below that)")

cols = ["tag", "f1", "mean_FPR", "mean_FNR", "abs_balance",
        "AAE_FPR", "SAE_FPR", "AAE_FNR", "SAE_FNR",
        "FPR_gap", "FNR_gap", "_score",
        "enable_1_ceo", "enable_2_platt", "enable_3_reductions", "enable_4_roc"]
cols = [c for c in cols if c in df.columns]

print("\n=== TOP 5 (lowest score; lower FPR + more balance) ===")
print(df[cols].head(5).to_string(index=False))

best = df.iloc[0].to_dict()
with open(os.path.join(out_dir, "best_setting.json"), "w") as f:
    json.dump(best, f, indent=2, default=str)

print(f"\n>>> BEST SETTING: {best.get('tag')}")
print(f"    F1={best.get('f1', 0):.4f}")
print(f"    mean_FPR={best.get('mean_FPR', 0):.4f}  mean_FNR={best.get('mean_FNR', 0):.4f}  |diff|={best.get('abs_balance', 0):.4f}")
print(f"    AAE: FPR={best.get('AAE_FPR', 0):.4f} FNR={best.get('AAE_FNR', 0):.4f}")
print(f"    SAE: FPR={best.get('SAE_FPR', 0):.4f} FNR={best.get('SAE_FNR', 0):.4f}")
print(f"    FPR_gap={best.get('FPR_gap', 0):.4f}  FNR_gap={best.get('FNR_gap', 0):.4f}")

print("\n=== TOP 5 BY F1 ===")
df_f1 = df.sort_values("f1", ascending=False)
print(df_f1[cols].head(5).to_string(index=False))

print("\n=== ALL RUNS (sorted by composite score) ===")
print(df[cols].to_string(index=False))
PYEOF

echo ""
echo "Aggregation done at $(date)"
