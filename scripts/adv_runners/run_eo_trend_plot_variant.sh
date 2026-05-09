#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project

# Configure via:
#   VARIANT=balanced_indomain    ./run_eo_trend_plot_variant.sh
#   VARIANT=unbalanced_indomain  ./run_eo_trend_plot_variant.sh
#   VARIANT=balanced_hatexplain  ./run_eo_trend_plot_variant.sh
#   VARIANT=unbalanced_hatexplain ./run_eo_trend_plot_variant.sh
VARIANT="${VARIANT:-unbalanced_indomain}"

case "$VARIANT" in
  balanced_indomain)
    BEST_JSON="data/results/adv_xgb_eo_grid/best_by_val_score.json"
    DATA_DIR="data/processed/twitterAAE/balanced"
    EMB_DIR="data/embeddings/twitterAAE/balanced"
    TEST_CSV="$DATA_DIR/test.csv"
    TEST_EMB="$EMB_DIR/test_emb.npy"
    OUT_DIR="data/results/plots/eo_trend_balanced"
    PREFIX="eo_balanced"
    TSUF="(balanced TwitterAAE in-domain)"
    ;;
  unbalanced_indomain)
    BEST_JSON="data/results/adv_xgb_eo_grid_unbalanced/best_by_val_score.json"
    DATA_DIR="data/processed/twitterAAE/unbalanced"
    EMB_DIR="data/embeddings/twitterAAE/unbalanced"
    TEST_CSV="$DATA_DIR/test.csv"
    TEST_EMB="$EMB_DIR/test_emb.npy"
    OUT_DIR="data/results/plots/eo_trend_unbalanced"
    PREFIX="eo_unbalanced"
    TSUF="(unbalanced TwitterAAE in-domain)"
    ;;
  balanced_hatexplain)
    BEST_JSON="data/results/adv_xgb_eo_grid/best_by_val_score.json"
    DATA_DIR="data/processed/twitterAAE/balanced"
    EMB_DIR="data/embeddings/twitterAAE/balanced"
    TEST_CSV="data/processed/hatexplain/test.csv"
    TEST_EMB="data/embeddings/hatexplain/test_emb.npy"
    OUT_DIR="data/results/plots/hatexplain_trend_balanced"
    PREFIX="hatexplain_balanced"
    TSUF="(balanced model -> HateXplain)"
    ;;
  unbalanced_hatexplain)
    BEST_JSON="data/results/adv_xgb_eo_grid_unbalanced/best_by_val_score.json"
    DATA_DIR="data/processed/twitterAAE/unbalanced"
    EMB_DIR="data/embeddings/twitterAAE/unbalanced"
    TEST_CSV="data/processed/hatexplain/test.csv"
    TEST_EMB="data/embeddings/hatexplain/test_emb.npy"
    OUT_DIR="data/results/plots/hatexplain_trend_unbalanced"
    PREFIX="hatexplain_unbalanced"
    TSUF="(unbalanced model -> HateXplain)"
    ;;
  *)
    echo "ERROR: unknown VARIANT='$VARIANT'"; exit 1 ;;
esac

mkdir -p "$OUT_DIR"

python -u src/plots/plot_eo_trend.py \
    --best_json "$BEST_JSON" \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv   "$DATA_DIR/val.csv" \
    --train_emb "$EMB_DIR/train_emb.npy" \
    --val_emb   "$EMB_DIR/val_emb.npy" \
    --test_csv  "$TEST_CSV" \
    --test_emb  "$TEST_EMB" \
    --out_dir   "$OUT_DIR" \
    --prefix    "$PREFIX" \
    --title_suffix "$TSUF" \
    --tree_method hist --device cuda \
    --num_round 100
