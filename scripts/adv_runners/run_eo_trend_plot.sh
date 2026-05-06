#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
mkdir -p data/results/plots

python -u src/plots/plot_eo_trend.py
