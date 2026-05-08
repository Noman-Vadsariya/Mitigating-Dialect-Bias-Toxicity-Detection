#!/bin/bash
#SBATCH --job-name=adv_xgb_lam_sweep
#SBATCH --output=adv_xgb_lam_sweep_%j.out
#SBATCH --error=adv_xgb_lam_sweep_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -eo pipefail

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "TMPDIR = $TMPDIR"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Redirect caches to scratch to avoid home quota issues
export CONDA_PKGS_DIRS=/scratch1/aqliang/.conda_pkgs
export PIP_CACHE_DIR=/scratch1/aqliang/.pip_cache
export HF_HOME=/scratch1/aqliang/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch1/aqliang/.cache/huggingface
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

module purge
module load conda

if [ -f /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate ]; then
    source /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate
elif [ -f /scratch1/aqliang/haluguard-env/bin/activate ]; then
    source /scratch1/aqliang/haluguard-env/bin/activate
else
    echo "No Python environment found."
    exit 1
fi

cd /scratch1/aqliang/CSCI567-ML-Project

echo "Python: $(which python)"
pip install --no-cache-dir -q -r requirements.txt

# Fixed adversary strength (picked because it has been a strong setting in prior runs)
ADV_C_FIXED="2.0"
LAMBDAS=("0.00" "0.05" "0.10" "0.25" "0.50")

BASE_OUT="data/results/adv_xgb_lambda_sweep"
PER_LAMBDA_DIR="$BASE_OUT/per_lambda"
METRICS_DIR="$BASE_OUT/metrics_by_lambda"
PLOTS_DIR="$BASE_OUT/plots"

mkdir -p "$PER_LAMBDA_DIR" "$METRICS_DIR" "$PLOTS_DIR"

echo "Starting lambda_adv sweep with adv_c=$ADV_C_FIXED at $(date)"

for LAM in "${LAMBDAS[@]}"; do
    LAM_TAG="${LAM/./p}"
    RUN_DIR="$PER_LAMBDA_DIR/lambda_${LAM_TAG}"

    echo "------------------------------------------"
    echo "Running lambda_adv=$LAM | adv_c=$ADV_C_FIXED"
    echo "Output: $RUN_DIR"
    echo "------------------------------------------"

    python src/experiments/train_adv_xgb.py \
      --train_csv data/processed/train.csv \
      --val_csv data/processed/val.csv \
      --test_csv data/processed/test.csv \
      --train_emb data/embeddings/train_emb.npy \
      --val_emb data/embeddings/val_emb.npy \
      --test_emb data/embeddings/test_emb.npy \
      --dialect_col dialect_strict \
      --num_round 100 \
      --lambda_grid "$LAM" \
      --adv_c_grid "$ADV_C_FIXED" \
      --out_dir "$RUN_DIR"
done

echo "Collecting per-run metrics and building plots..."

python - <<'PY'
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

base = Path("data/results/adv_xgb_lambda_sweep")
per_lambda_dir = base / "per_lambda"
metrics_dir = base / "metrics_by_lambda"
plots_dir = base / "plots"
metrics_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

rows = []

for run_dir in sorted(per_lambda_dir.glob("lambda_*")):
    tuning_path = run_dir / "adv_xgb_tuning.csv"
    summary_path = run_dir / "adv_xgb_summary.txt"
    if not tuning_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing expected outputs in {run_dir}")

    tuning = pd.read_csv(tuning_path).iloc[0].to_dict()
    txt = summary_path.read_text()

    def grab(label: str) -> float:
        m = re.search(rf"{re.escape(label)}:\s*([-+]?\d*\.?\d+)", txt)
        if not m:
            raise ValueError(f"Could not parse '{label}' from {summary_path}")
        return float(m.group(1))

    row = {
        "lambda_adv": float(tuning["lambda_adv"]),
        "adv_c": float(tuning["adv_c"]),
        "val_acc": float(tuning["val_acc"]),
        "val_f1": float(tuning["val_f1"]),
        "val_fpr_gap": float(tuning["val_FPR_gap"]),
        "val_fnr_gap": float(tuning["val_FNR_gap"]),
        "val_difav": float(tuning["val_DIfav"]),
        "val_diunfav": float(tuning["val_DIunfav"]),
        "test_accuracy": grab("Test Accuracy"),
        "test_f1": grab("Test F1"),
        "test_aae_fpr": grab("AAE FPR"),
        "test_sae_fpr": grab("SAE FPR"),
        "test_fpr_gap": grab("FPR Gap"),
        "test_aae_fnr": grab("AAE FNR"),
        "test_sae_fnr": grab("SAE FNR"),
        "test_fnr_gap": grab("FNR Gap"),
        "test_difav": grab("DIfav"),
        "test_diunfav": grab("DIunfav"),
    }

    rows.append(row)

    out_file = metrics_dir / f"lambda_{str(row['lambda_adv']).replace('.', 'p')}_metrics.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

df = pd.DataFrame(rows).sort_values("lambda_adv")
df.to_csv(base / "lambda_sweep_metrics.csv", index=False)

metric_cols = [c for c in df.columns if c not in ("lambda_adv", "adv_c")]
for col in metric_cols:
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["lambda_adv"], df[col], marker="o")
    plt.xlabel("lambda_adv")
    plt.ylabel(col)
    plt.title(f"{col} vs lambda_adv (adv_c={df['adv_c'].iloc[0]:.2f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{col}_vs_lambda_adv.png", dpi=200)
    plt.close()

print(f"Saved aggregate metrics: {base / 'lambda_sweep_metrics.csv'}")
print(f"Saved per-lambda metrics in: {metrics_dir}")
print(f"Saved plots in: {plots_dir}")
PY

echo "=========================================="
echo "✓ Lambda sweep and plotting completed at $(date)"
echo "✓ Base output: $BASE_OUT"
echo "=========================================="
