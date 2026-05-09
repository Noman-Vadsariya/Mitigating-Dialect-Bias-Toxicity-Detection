#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
OUT_DIR="data/results/adv_xgb_eo_grid_unbalanced"

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

cols = ["lambda_adv","adv_c","min_f1_ratio","t_aae","t_sae",
        "val_f1","val_FPR_gap","val_mean_FPR","val_score",
        "test_f1","test_AAE_FPR","test_SAE_FPR","test_AAE_FNR","test_SAE_FNR",
        "test_FPR_gap","test_FNR_gap","test_mean_FPR","test_mean_FNR","test_score"]

print("\n=== TOP 10 BY VAL SCORE ===")
print(df.sort_values("val_score").head(10)[cols].round(4).to_string(index=False))

print("\n=== TOP 10 BY TEST SCORE (oracle) ===")
print(df.sort_values("test_score").head(10)[cols].round(4).to_string(index=False))

mask = df["test_f1"] >= 0.80
print("\n=== TOP 10 BY TEST FPR_GAP (F1>=0.80) ===")
print(df[mask].sort_values("test_FPR_gap").head(10)[cols].round(4).to_string(index=False))

print("\n=== TOP 10 BY TEST mean_FPR (F1>=0.80) ===")
print(df[mask].sort_values("test_mean_FPR").head(10)[cols].round(4).to_string(index=False))

best = df.sort_values("val_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_val_score.json"), "w") as f:
    json.dump(best, f, indent=2, default=str)
print(f"\n>>> BEST by val_score: lam={best['lambda_adv']} c={best['adv_c']} "
      f"ratio={best['min_f1_ratio']}  t=({best['t_aae']:.3f},{best['t_sae']:.3f})")
print(f"    test F1={best['test_f1']:.4f} mean_FPR={best['test_mean_FPR']:.4f} "
      f"gap={best['test_FPR_gap']:.4f}")
print(f"    AAE_FPR={best['test_AAE_FPR']:.4f} SAE_FPR={best['test_SAE_FPR']:.4f} "
      f"AAE_FNR={best['test_AAE_FNR']:.4f} SAE_FNR={best['test_SAE_FNR']:.4f}")

best_t = df.sort_values("test_score").iloc[0].to_dict()
with open(os.path.join(out_dir, "best_by_test_score.json"), "w") as f:
    json.dump(best_t, f, indent=2, default=str)
print(f"\n>>> ORACLE BEST by test_score: lam={best_t['lambda_adv']} "
      f"c={best_t['adv_c']} ratio={best_t['min_f1_ratio']}  "
      f"t=({best_t['t_aae']:.3f},{best_t['t_sae']:.3f})")
print(f"    test F1={best_t['test_f1']:.4f} mean_FPR={best_t['test_mean_FPR']:.4f} "
      f"gap={best_t['test_FPR_gap']:.4f}")
print(f"    AAE_FPR={best_t['test_AAE_FPR']:.4f} SAE_FPR={best_t['test_SAE_FPR']:.4f} "
      f"AAE_FNR={best_t['test_AAE_FNR']:.4f} SAE_FNR={best_t['test_SAE_FNR']:.4f}")
PYEOF
