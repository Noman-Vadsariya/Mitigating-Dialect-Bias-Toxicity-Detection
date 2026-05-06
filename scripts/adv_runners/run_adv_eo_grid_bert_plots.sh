#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
DATA_TAG="${DATA_TAG:-unbalanced}"
OUT_DIR="data/results/adv_xgb_eo_grid_bert_${DATA_TAG}"
DATA_DIR="data/processed/twitterAAE/${DATA_TAG}"
EMB_DIR="data/embeddings/twitterAAE/${DATA_TAG}"
PLOT_DIR="data/results/plots"
mkdir -p "$PLOT_DIR"

python -u src/plots/plot_eo_trend_bert.py \
    --best_json "$OUT_DIR/best_by_val_score.json" \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv   "$DATA_DIR/val.csv" \
    --test_csv  "$DATA_DIR/test.csv" \
    --train_emb "$EMB_DIR/train_emb.npy" \
    --val_emb   "$EMB_DIR/val_emb.npy" \
    --test_emb  "$EMB_DIR/test_emb.npy" \
    --out_dir   "$PLOT_DIR" \
    --prefix    "bert_${DATA_TAG}" \
    --title_suffix "${DATA_TAG} TwitterAAE" \
    --tree_method hist --device cuda \
    --num_round 100 --warmup_rounds 5 \
    --adv_epochs_per_round 2 --adv_batch_size 1024
