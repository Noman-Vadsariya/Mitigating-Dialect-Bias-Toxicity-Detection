#!/bin/bash
#SBATCH --job-name=adv_xgb_original
#SBATCH --output=adv_xgb_original_%j.out
#SBATCH --error=adv_xgb_original_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

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

echo "Starting ORIGINAL adversarial XGBoost training at $(date)"
python src/experiments/train_adv_xgb.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --test_csv data/processed/test.csv \
  --train_emb data/embeddings/train_emb.npy \
  --val_emb data/embeddings/val_emb.npy \
  --test_emb data/embeddings/test_emb.npy \
  --dialect_col dialect_strict \
  --num_round 100 \
  --lambda_grid "0.0,0.05,0.1,0.25,0.5" \
  --adv_c_grid "0.5,1.0,2.0" \
  --out_dir data/results/adv_xgb_original

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✓ Original adversarial training completed at $(date)"
    echo "✓ Results directory: data/results/adv_xgb_original"
    ls -la data/results/adv_xgb_original
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ Original adversarial training failed at $(date)"
    echo "=========================================="
    exit 1
fi
