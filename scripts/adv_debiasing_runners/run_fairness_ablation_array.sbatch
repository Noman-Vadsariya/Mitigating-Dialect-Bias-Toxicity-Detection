#!/bin/bash
#SBATCH --job-name=fairness_run
#SBATCH --output=logs/fairness_run_%A_%a.out
#SBATCH --error=logs/fairness_run_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-15

set -eo pipefail

echo "=============================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

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
    echo "ERROR: No Python environment found"
    exit 1
fi

python -c "import fairlearn, xgboost, sklearn; print('fairlearn', fairlearn.__version__, '| xgboost', xgboost.__version__)"

cd /scratch1/aqliang/CSCI567-ML-Project
OUT_DIR="data/results/fairness_ablation"
mkdir -p "$OUT_DIR" logs

# All 16 runs: baseline + 15 non-empty subsets of {1, 2, 3, 4}
TAGS=(
    "baseline"
    "T1"           "T2"           "T3"           "T4"
    "T1_T2"        "T1_T3"        "T1_T4"
    "T2_T3"        "T2_T4"        "T3_T4"
    "T1_T2_T3"     "T1_T2_T4"     "T1_T3_T4"     "T2_T3_T4"
    "T1_T2_T3_T4"
)
FLAG_SETS=(
    ""
    "--enable_1"                "--enable_2"                "--enable_3"                "--enable_4"
    "--enable_1 --enable_2"     "--enable_1 --enable_3"     "--enable_1 --enable_4"
    "--enable_2 --enable_3"     "--enable_2 --enable_4"     "--enable_3 --enable_4"
    "--enable_1 --enable_2 --enable_3"
    "--enable_1 --enable_2 --enable_4"
    "--enable_1 --enable_3 --enable_4"
    "--enable_2 --enable_3 --enable_4"
    "--enable_all"
)

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
FLAGS="${FLAG_SETS[$SLURM_ARRAY_TASK_ID]}"

echo "Running: $TAG   ($FLAGS)"
echo "=============================================="

python src/experiments/fairness_postprocess.py \
    --out_dir "$OUT_DIR" \
    --tag "$TAG" \
    --tree_method hist --device cuda \
    --num_round 100 \
    $FLAGS

echo ""
echo "Task $TAG completed at: $(date)"
