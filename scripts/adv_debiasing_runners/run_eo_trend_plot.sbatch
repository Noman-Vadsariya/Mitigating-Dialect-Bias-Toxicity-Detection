#!/bin/bash
#SBATCH --job-name=eo_trend_plot
#SBATCH --output=logs/eo_trend_plot_%j.out
#SBATCH --error=logs/eo_trend_plot_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

echo "=============================================="
echo "EO Trend Plot   Node: $(hostname)   Start: $(date)"
echo "=============================================="

export PYTHONUNBUFFERED=1
export CONDA_PKGS_DIRS=/scratch1/aqliang/.conda_pkgs
export PIP_CACHE_DIR=/scratch1/aqliang/.pip_cache

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
mkdir -p data/results/plots logs

python -u src/plots/plot_eo_trend.py

echo ""
echo "Done at: $(date)"
