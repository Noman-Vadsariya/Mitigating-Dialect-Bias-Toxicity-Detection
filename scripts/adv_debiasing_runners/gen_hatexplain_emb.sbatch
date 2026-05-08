#!/bin/bash
#SBATCH --job-name=gen_emb_hatexplain
#SBATCH --output=logs/gen_emb_hatexplain_%j.out
#SBATCH --error=logs/gen_emb_hatexplain_%j.err
#SBATCH --time=00:30:00
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
mkdir -p data/embeddings/hatexplain logs

python -u - <<'PYEOF'
import os, sys
sys.path.insert(0, "src/baselines")
import pandas as pd, numpy as np
from generate_embeddings import get_embeddings

src = "data/processed/hatexplain/test.csv"
dst = "data/embeddings/hatexplain/test_emb.npy"

df = pd.read_csv(src)
print(f"Loaded {len(df)} rows from {src}")
texts = df["tweet"].astype(str).tolist()
X = get_embeddings(texts)
print(f"Embeddings shape: {X.shape}")
np.save(dst, X)
print(f"Saved: {dst}")
PYEOF

echo "Completed at: $(date)"
