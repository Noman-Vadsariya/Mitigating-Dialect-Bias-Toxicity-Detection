#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project

python -u src/plots/plot_eo_trend_ternary.py \
    --tree_method hist --device cuda \
    --num_round 100
