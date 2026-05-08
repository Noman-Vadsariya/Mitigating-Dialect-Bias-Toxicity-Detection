#!/bin/bash
#SBATCH --job-name=eval_hatexplain
#SBATCH --output=logs/eval_hatexplain_%j.out
#SBATCH --error=logs/eval_hatexplain_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

echo "=============================================="
echo "Node: $(hostname)   Start: $(date)"
echo "=============================================="

export PYTHONUNBUFFERED=1
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

# Override via:
#   sbatch --export=ALL,DATA_TAG=balanced,BEST_JSON=...,DATA_DIR=...,EMB_DIR=... run_eval_hatexplain.sbatch
DATA_TAG="${DATA_TAG:-unbalanced}"
DATA_DIR="${DATA_DIR:-data/processed/twitterAAE/${DATA_TAG}}"
EMB_DIR="${EMB_DIR:-data/embeddings/twitterAAE/${DATA_TAG}}"

if [ "$DATA_TAG" = "balanced" ]; then
    BEST_JSON="${BEST_JSON:-data/results/adv_xgb_eo_grid/best_by_val_score.json}"
else
    BEST_JSON="${BEST_JSON:-data/results/adv_xgb_eo_grid_unbalanced/best_by_val_score.json}"
fi

OUT_DIR="data/results/hatexplain_generalization/${DATA_TAG}"
mkdir -p "$OUT_DIR" logs

echo "DATA_TAG=$DATA_TAG"
echo "Best config: $BEST_JSON"
echo "Train data:  $DATA_DIR / $EMB_DIR"
echo "Output dir:  $OUT_DIR"
echo "=============================================="

python -u src/experiments/eval_hatexplain.py \
    --best_json "$BEST_JSON" \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv   "$DATA_DIR/val.csv" \
    --src_test_csv "$DATA_DIR/test.csv" \
    --train_emb "$EMB_DIR/train_emb.npy" \
    --val_emb   "$EMB_DIR/val_emb.npy" \
    --src_test_emb "$EMB_DIR/test_emb.npy" \
    --out_dir "$OUT_DIR" \
    --tree_method hist --device cuda \
    --num_round 100

echo ""
echo "Completed at: $(date)"
