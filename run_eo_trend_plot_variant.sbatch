#!/bin/bash
#SBATCH --job-name=eo_trend_plot
#SBATCH --output=logs/eo_trend_plot_%j.out
#SBATCH --error=logs/eo_trend_plot_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

echo "=============================================="
echo "Node: $(hostname)   Start: $(date)"
echo "Variant: ${VARIANT:-(unset)}"
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
mkdir -p logs

# Configure via:
#   sbatch --export=ALL,VARIANT=unbalanced_indomain   run_eo_trend_plot_variant.sbatch
#   sbatch --export=ALL,VARIANT=balanced_hatexplain   run_eo_trend_plot_variant.sbatch
#   sbatch --export=ALL,VARIANT=unbalanced_hatexplain run_eo_trend_plot_variant.sbatch

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
echo "best:    $BEST_JSON"
echo "train:   $DATA_DIR / $EMB_DIR"
echo "test:    $TEST_CSV / $TEST_EMB"
echo "out:     $OUT_DIR (prefix=$PREFIX)"
echo "=============================================="

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

echo ""
echo "Completed at: $(date)"
