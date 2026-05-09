#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
DATA_TAG="${DATA_TAG:-unbalanced}"
OUT_DIR="data/results/adv_xgb_eo_grid_bert_${DATA_TAG}"
DATA_DIR="data/processed/twitterAAE/${DATA_TAG}"
EMB_DIR="data/embeddings/twitterAAE/${DATA_TAG}"
mkdir -p "$OUT_DIR"

LAMBDA_GRID="0.0,0.025,0.05,0.1,0.15,0.25,0.4,0.6,1.0,1.5"
HIDDEN_GRID="128,256"
LR_GRID="1e-3,3e-4"
RATIOS="none,0.85,0.90,0.93,0.95,0.97,0.99"
TAG="full"

python -u src/experiments/run_adv_eo_grid_bert.py \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv   "$DATA_DIR/val.csv" \
    --test_csv  "$DATA_DIR/test.csv" \
    --train_emb "$EMB_DIR/train_emb.npy" \
    --val_emb   "$EMB_DIR/val_emb.npy" \
    --test_emb  "$EMB_DIR/test_emb.npy" \
    --out_dir "$OUT_DIR" \
    --tag "$TAG" \
    --tree_method hist --device cuda \
    --num_round 100 --warmup_rounds 5 \
    --lambda_grid "$LAMBDA_GRID" \
    --adv_hidden_grid "$HIDDEN_GRID" \
    --adv_lr_grid "$LR_GRID" \
    --adv_epochs_per_round 2 \
    --adv_dropout 0.2 \
    --adv_weight_decay 1e-4 \
    --adv_batch_size 1024 \
    --min_f1_ratios "$RATIOS"
