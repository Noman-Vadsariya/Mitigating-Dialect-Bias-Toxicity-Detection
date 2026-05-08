#!/bin/bash
#SBATCH --job-name=adv_grid
#SBATCH --output=logs/adv_grid_%A_%a.out
#SBATCH --error=logs/adv_grid_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -eo pipefail

echo "=============================================="
echo "Array task: $SLURM_ARRAY_TASK_ID   Node: $(hostname)   Start: $(date)"
echo "=============================================="

export CONDA_PKGS_DIRS=/scratch1/aqliang/.conda_pkgs
export PIP_CACHE_DIR=/scratch1/aqliang/.pip_cache
export HF_HOME=/scratch1/aqliang/.cache/huggingface
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME"

module purge
module load conda

if [ -f /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate ]; then
    source /scratch1/aqliang/CSCI567-ML-Project/venv/bin/activate
elif [ -f /scratch1/aqliang/haluguard-env/bin/activate ]; then
    source /scratch1/aqliang/haluguard-env/bin/activate
else
    echo "ERROR: No Python environment found"; exit 1
fi

cd /scratch1/aqliang/CSCI567-ML-Project
OUT_DIR="data/results/adv_xgb_largegrid"
mkdir -p "$OUT_DIR" logs

# Grid: 10 lambda values x 5 adv_c values = 50 cells
# Each array task handles one lambda value (5 cells).
LAMBDAS=(0.0 0.025 0.05 0.1 0.15 0.25 0.4 0.6 1.0 1.5)
ADV_C="0.1,0.5,1.0,3.0,10.0"

LAM="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"
TAG="lam_${LAM}"

echo "Running lambda=$LAM  adv_c grid=$ADV_C"
echo "=============================================="

python src/experiments/run_adv_grid.py \
    --out_dir "$OUT_DIR" \
    --tag "$TAG" \
    --tree_method hist --device cuda \
    --num_round 100 \
    --lambda_grid "$LAM" \
    --adv_c_grid "$ADV_C"

echo ""
echo "Task $TAG completed at: $(date)"
