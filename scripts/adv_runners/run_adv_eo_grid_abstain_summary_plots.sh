#!/bin/bash
set -eo pipefail
export PYTHONUNBUFFERED=1

cd /scratch1/aqliang/CSCI567-ML-Project
DATA_TAG="${DATA_TAG:-unbalanced}"
OUT_DIR="data/results/adv_xgb_eo_grid_abstain_${DATA_TAG}"
DATA_DIR="data/processed/twitterAAE/${DATA_TAG}"
EMB_DIR="data/embeddings/twitterAAE/${DATA_TAG}"
PLOT_DIR="data/results/plots"
mkdir -p "$PLOT_DIR"

python - <<PYEOF
import json, os
out = "$OUT_DIR"
with open(os.path.join(out, "best_by_val_score.json")) as f: b = json.load(f)
with open(os.path.join(out, "best_by_test_score.json")) as f: bt = json.load(f)

def ff(k, default=float('nan')):
    v = b.get(k, default)
    try: return float(v)
    except Exception: return default

lines = []
lines.append("Abstain Adversarial XGBoost (toxic / nontoxic / unsure) Summary")
lines.append("Dataset: TwitterAAE ${DATA_TAG}")
lines.append("=" * 64)
lines.append("")
lines.append("Best candidate (selected by val_score):")
lines.append(f"  lambda_adv   = {b['lambda_adv']}")
lines.append(f"  adv_c        = {b['adv_c']}")
lines.append(f"  min_f1_ratio = {b['min_f1_ratio']}")
lines.append(f"  min_coverage = {b['min_coverage']}")
lines.append(f"  AAE: threshold={ff('t_aae'):.4f}  deadband=+/-{ff('delta_aae'):.4f}")
lines.append(f"  SAE: threshold={ff('t_sae'):.4f}  deadband=+/-{ff('delta_sae'):.4f}")
lines.append("")
for split in ("val", "test"):
    lines.append(f"--- {split.title()} ---")
    lines.append(f"Selective F1:        {ff(f'{split}_sel_f1'):.4f}")
    lines.append(f"Selective Accuracy:  {ff(f'{split}_sel_accuracy'):.4f}")
    lines.append(f"Coverage:            {ff(f'{split}_coverage'):.4f}")
    lines.append(f"Abstain rate:        {ff(f'{split}_abstain_rate'):.4f}")
    lines.append(f"AAE FPR: {ff(f'{split}_AAE_FPR'):.4f}   SAE FPR: {ff(f'{split}_SAE_FPR'):.4f}   FPR Gap: {ff(f'{split}_FPR_gap'):.4f}")
    lines.append(f"AAE FNR: {ff(f'{split}_AAE_FNR'):.4f}   SAE FNR: {ff(f'{split}_SAE_FNR'):.4f}   FNR Gap: {ff(f'{split}_FNR_gap'):.4f}")
    lines.append(f"AAE abstain: {ff(f'{split}_AAE_abstain_rate'):.4f}   SAE abstain: {ff(f'{split}_SAE_abstain_rate'):.4f}   Abstain Gap: {ff(f'{split}_abstain_gap'):.4f}")
    lines.append(f"AAE coverage: {ff(f'{split}_AAE_coverage'):.4f}   SAE coverage: {ff(f'{split}_SAE_coverage'):.4f}")
    lines.append(f"Mean FPR: {ff(f'{split}_mean_FPR'):.4f}   Mean FNR: {ff(f'{split}_mean_FNR'):.4f}   |EO balance|: {ff(f'{split}_abs_balance'):.4f}")
    lines.append("")

lines.append("--- Oracle best (selected by test_score, for reference) ---")
lines.append(f"  lambda_adv={bt['lambda_adv']}  adv_c={bt['adv_c']}  ratio={bt['min_f1_ratio']}  cov>={bt['min_coverage']}")
def fft(k):
    try: return float(bt.get(k, float('nan')))
    except Exception: return float('nan')
lines.append(f"  Test sel_F1={fft('test_sel_f1'):.4f}  coverage={fft('test_coverage'):.4f}  FPR_gap={fft('test_FPR_gap'):.4f}  FNR_gap={fft('test_FNR_gap'):.4f}")

path = os.path.join(out, "adv_xgb_summary.txt")
with open(path, "w") as f: f.write("\n".join(lines) + "\n")
print(f"Wrote {path}")
print("\n".join(lines))
PYEOF

python -u src/plots/plot_eo_trend_abstain.py \
    --best_json "$OUT_DIR/best_by_val_score.json" \
    --train_csv "$DATA_DIR/train.csv" \
    --val_csv   "$DATA_DIR/val.csv" \
    --test_csv  "$DATA_DIR/test.csv" \
    --train_emb "$EMB_DIR/train_emb.npy" \
    --val_emb   "$EMB_DIR/val_emb.npy" \
    --test_emb  "$EMB_DIR/test_emb.npy" \
    --out_dir   "$PLOT_DIR" \
    --prefix    "abstain_${DATA_TAG}" \
    --title_suffix "${DATA_TAG} TwitterAAE" \
    --tree_method hist --device cuda \
    --num_round 100 --warmup_rounds 5
