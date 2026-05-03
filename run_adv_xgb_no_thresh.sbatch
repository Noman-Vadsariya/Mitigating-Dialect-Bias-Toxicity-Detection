#!/bin/bash
#SBATCH --job-name=adv_xgb_no_thresh
#SBATCH --output=logs/adv_xgb_no_thresh_%j.out
#SBATCH --error=logs/adv_xgb_no_thresh_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail

echo "=============================================="
echo "Adversarial XGBoost with Multi-Technique Debiasing"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Redirect caches to scratch to avoid home quota issues
export CONDA_PKGS_DIRS=/scratch1/aqliang/.conda_pkgs
export PIP_CACHE_DIR=/scratch1/aqliang/.pip_cache
export HF_HOME=/scratch1/aqliang/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch1/aqliang/.cache/huggingface
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

module purge
module load conda

# Activate environment
if [ -f /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate ]; then
    source /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate
elif [ -f /scratch1/aqliang/haluguard-env/bin/activate ]; then
    source /scratch1/aqliang/haluguard-env/bin/activate
else
    echo "ERROR: No Python environment found"
    exit 1
fi

echo "Python: $(which python)"
echo "=============================================="

cd /scratch1/aqliang/CSCI567-ML-Project
mkdir -p logs data/results/adv_xgb_no_thresh

# Run with all three techniques enabled (default)
python src/experiments/train_adv_xgb.py \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --test_csv data/processed/test.csv \
    --train_emb data/embeddings/train_emb.npy \
    --val_emb data/embeddings/val_emb.npy \
    --test_emb data/embeddings/test_emb.npy \
    --out_dir data/results/adv_xgb_no_thresh \
    --num_round 100 \
    --tree_method hist \
    --device cuda \
    --lambda_grid "0.0,0.05,0.1,0.25,0.5" \
    --adv_c_grid "0.5,1.0,2.0" \
    --use_leaf_adv \
    --use_reweighting \
    --no_threshold_tuning \
    

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "Results saved to: data/results/adv_xgb_no_thresh/"
echo "=============================================="
