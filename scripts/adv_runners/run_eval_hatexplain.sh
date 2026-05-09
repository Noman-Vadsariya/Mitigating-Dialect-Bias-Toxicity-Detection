#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project

DATA_TAG="${DATA_TAG:-unbalanced}"
DATA_DIR="${DATA_DIR:-data/processed/twitterAAE/${DATA_TAG}}"
EMB_DIR="${EMB_DIR:-data/embeddings/twitterAAE/${DATA_TAG}}"

if [ "$DATA_TAG" = "balanced" ]; then
    BEST_JSON="${BEST_JSON:-data/results/adv_xgb_eo_grid/best_by_val_score.json}"
else
    BEST_JSON="${BEST_JSON:-data/results/adv_xgb_eo_grid_unbalanced/best_by_val_score.json}"
fi

OUT_DIR="data/results/hatexplain_generalization/${DATA_TAG}"
mkdir -p "$OUT_DIR"

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
