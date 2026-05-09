# CSCI567-ML-Project

This repository contains code and experiments for dialect bias mitigation in toxicity detection, focusing on the TwitterAAE and HateXplain datasets. The project implements and evaluates several baselines and fairness-aware algorithms for text classification, with an emphasis on reducing false positives for African-American English (AAE) dialects while maintaining overall performance.

## Project Structure

- **data/**: Processed datasets, embeddings, and raw data.
- **refactored_twitter_aae/**: Code for dataset preprocessing and dialect prediction
- **results/**: Model predictions, metrics, and plots.
- **src/**: Source code for baselines, experiments, and plotting utils.

## Baselines Implemented

- **BERT Fine-tuning** (`src/baselines/finetune_bert.py`): Fine-tunes a BERT model for toxicity classification.
- **BERT Inference** (`src/baselines/bert_inference.py`): Runs inference using a fine-tuned BERT model.
- **ToxicBERT Inference** (`src/baselines/toxicbert_inference.py`): Uses the pre-trained ToxicBERT model for toxicity prediction.
- **XGBoost Baseline** (`src/baselines/train_xgboost.py`): Trains a standard XGBoost classifier on text embeddings.

## Fairness Experiments

- **Fair Adversarial Gradient Tree Boosting (FAGTB-NN)** (`src/experiments/fagbt.py`): Implements the method from Grari et al., combining XGBoost with a neural adversary to mitigate dialect bias.
- **FairXGBoost (Paper-style Regularizer)** (`src/experiments/fairxgb_paper.py`): Implements the FairXGBoost objective to penalize bias in toxicity scores.
- **Custom XGBoost Objectives**:
  - **Fairness with Gamma Surrogate** (`src/experiments/fair_xgb.py`): Penalizes higher toxic scores for AAE using a differentiable surrogate.
  - **Covariance-based Fairness** (`src/experiments/fair_xgb_cov.py`): Uses a covariance-based loss to reduce AAE false positives.
  - **Gamma-only Squared Fairness** (`src/experiments/fair_xgb_gamma_only.py`): Directly penalizes the difference in toxic probabilities between AAE and SAE.
- **Adversarial XGBoost**:
  - **Adversarial Training** (`src/experiments/train_adv_xgb.py`): Trains XGBoost with an adversarial fairness objective.
  - **Equalized Odds Grid Search** (`src/experiments/run_adv_eo_grid.py`): Grid search over adversarial and fairness hyperparameters.
- **Postprocessing and Metrics**:
  - **Fairness Postprocessing** (`src/experiments/fairness_postprocess.py`): Computes group metrics and postprocesses predictions.

- **Vector Scaling Loss (VS-XGBoost)** (`src/experiments/train_vs_xgboost.py`):
  - This experiment implements post-hoc vector scaling to mitigate group disparities in toxicity classification. After training a standard XGBoost model, vector scaling parameters (alpha, beta) are tuned for the AAE group to rescale and shift the model's decision boundary, reducing false positive rates for AAE while maintaining overall performance. The method searches for optimal scaling parameters on the validation set and applies them to the test set, reporting group fairness metrics and performance.

## Usage

1. Prepare data and embeddings in the `data/` directory.
2. Run baseline or experiment scripts from the `src/` directory.
3. Results and metrics are saved in the `results/` directory.

## Adversarial Debiasing Runner Scripts

The `run_*.sh` scripts in `scripts/adv_runners/` are convenience runners for the
adversarial-debiasing pipeline. They assume a Python environment with the
project requirements is already active (no environment setup is performed by
the scripts) and that you are running on a machine with a CUDA GPU. Each
script runs exactly one stage of the pipeline.

Make scripts executable once with `chmod +x scripts/adv_runners/*.sh`, then
invoke directly (e.g. `./scripts/adv_runners/run_adv_eo_grid_unbalanced_array.sh`).
Some scripts read environment variables (`DATA_TAG`, `VARIANT`, ...) that may
be set inline:
`DATA_TAG=balanced ./scripts/adv_runners/run_adv_eo_grid_bert_array.sh`.

The pipeline has three stages: **(1) grid search**, **(2) aggregate** the
per-lambda CSVs into a `grid_summary.csv` and pick the best config by
val_score, and **(3) plot/evaluate** using that best config.

### Linear-adversary XGBoost (TwitterAAE, balanced)

- `run_adv_eo_grid_array.sh` — Stage 1. Sweeps `lambda_adv` x `adv_c` x
  `min_f1_ratio` for the EO + projection method on the balanced split,
  writing per-lambda CSVs to `data/results/adv_xgb_eo_grid/`.
- `run_adv_eo_grid_aggregate.sh` — Stage 2. Aggregates the per-lambda CSVs,
  computes val/test scores, and writes `grid_summary.csv`,
  `best_by_val_score.json`, and `best_by_test_score.json`.

### Linear-adversary XGBoost (TwitterAAE, unbalanced)

- `run_adv_eo_grid_unbalanced_array.sh` — Stage 1 on the unbalanced split
  (output dir `data/results/adv_xgb_eo_grid_unbalanced/`).
- `run_adv_eo_grid_unbalanced_aggregate.sh` — Stage 2 for the unbalanced run.

### BERT-MLP adversary

- `run_adv_eo_grid_bert_array.sh` — Stage 1. Replaces the linear adversary
  with a small MLP on top of BERT embeddings; sweeps `lambda_adv`,
  `adv_hidden`, `adv_lr`, `min_f1_ratio`. Set `DATA_TAG=balanced` or
  `DATA_TAG=unbalanced` (default).
- `run_adv_eo_grid_bert_aggregate.sh` — Stage 2; also writes a plain-text
  `adv_xgb_summary.txt`. Honors `DATA_TAG`.
- `run_adv_eo_grid_bert_plots.sh` — Stage 3. Retrains the best BERT-adversary
  config with per-round tracking and produces EO trend plots in
  `data/results/plots/`.

### Three-class abstain variant (toxic / nontoxic / unsure)

- `run_adv_eo_grid_abstain_array.sh` — Stage 1. Adds per-group thresholds
  with deadbands so the model can abstain; also sweeps a coverage floor.
- `run_adv_eo_grid_abstain_aggregate.sh` — Stage 2 (uses the selective-F1 +
  coverage + abstain-gap composite score).
- `run_adv_eo_grid_abstain_summary_plots.sh` — Stage 3. Writes
  `adv_xgb_summary.txt` and the abstain-aware EO trend plots.

### Ternary labels

- `run_adv_eo_grid_ternary_array.sh` — Stage 1, ternary EO + projection on
  `unbalanced_ternary` data.
- `run_adv_eo_grid_ternary_aggregate.sh` — Stage 2 for the ternary run
  (uses macro-F1 + mean FPR/FNR gaps).

### EO trend plots and HateXplain transfer

- `run_eo_trend_plot.sh` — Default EO-vs-rounds trend plot (no args; uses
  the defaults baked into `src/plots/plot_eo_trend.py`).
- `run_eo_trend_plot_ternary.sh` — Same trend plot for the ternary
  experiment.
- `run_eo_trend_plot_variant.sh` — Configurable EO trend plot. Set
  `VARIANT` to one of `balanced_indomain`, `unbalanced_indomain`,
  `balanced_hatexplain`, `unbalanced_hatexplain` to switch the train/test
  data and output directory.
- `run_eval_hatexplain.sh` — Out-of-domain evaluation: takes the best
  TwitterAAE config (selected by `DATA_TAG=balanced` or `unbalanced`) and
  evaluates it on HateXplain, reporting EO metrics under covariate shift.

  ## XGBoost + Vector Scaling Experiments

This section lists the main commands used to run the XGBoost + vector-scaling experiments for this project.

### 1. Binary XGBoost + Vector Scaling on balanced TwitterAAE

Use this command to train and evaluate the binary XGBoost + vector-scaling model on the **balanced** TwitterAAE split.

```bash
python src/experiments/train_vs_xgboost.py \
  --train_csv data/processed/twitterAAE/balanced/train.csv \
  --val_csv data/processed/twitterAAE/balanced/val.csv \
  --test_csv data/processed/twitterAAE/balanced/test.csv \
  --train_emb data/embeddings/twitterAAE/balanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/balanced/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/balanced/test_emb.npy \
  --dialect_col dialect_strict \
  --out_dir data/results/vs_xgb_training
```

### 2. Binary XGBoost + Vector Scaling on unbalanced TwitterAAE

Use this command to train and evaluate the binary XGBoost + vector-scaling model on the **unbalanced** TwitterAAE split.

```bash
python src/experiments/train_vs_xgboost.py \
  --train_csv data/processed/twitterAAE/unbalanced/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced/test_emb.npy \
  --dialect_col dialect_strict \
  --out_dir data/results/twitterAAE_experiments/vs_xgb_unbalanced
```

### 3. Generalization test: TwitterAAE to HateXplain

Use this command to train and calibrate on TwitterAAE, then test the learned vector-scaling setup directly on **HateXplain**.

```bash
python src/experiments/train_vs_xgboost.py \
  --train_csv data/processed/twitterAAE/unbalanced/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced/val.csv \
  --test_csv data/processed/hatexplain/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced/val_emb.npy \
  --test_emb data/embeddings/hatexplain/test_emb.npy \
  --dialect_col dialect_strict \
  --out_dir data/results/generalization_experiments/twitterAAE_to_hatexplain_vs_xgb
```

### 4. Three-way semantic ternary XGBoost

This command runs the semantic 3-class setup: **toxic / offensive / not toxic**.

```bash
python src/experiments/train_xgboost_ternary.py \
  --train_csv data/processed/twitterAAE/unbalanced_ternary/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced_ternary/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced_ternary/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced_ternary/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced_ternary/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced_ternary/test_emb.npy \
  --dialect_col dialect_strict \
  --label_col label \
  --out_dir data/results/twitterAAE_experiments/xgb_ternary
```

### 5. Weighted semantic ternary XGBoost

#### Manual class weights

```bash
python src/experiments/train_xgboost_ternary.py \
  --train_csv data/processed/twitterAAE/unbalanced_ternary/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced_ternary/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced_ternary/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced_ternary/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced_ternary/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced_ternary/test_emb.npy \
  --dialect_col dialect_strict \
  --label_col label \
  --class_weights 2 3 1 \
  --out_dir data/results/twitterAAE_experiments/xgb_ternary_w_2_3_1
```

#### Auto-weighted version

```bash
python src/experiments/train_xgboost_ternary.py \
  --train_csv data/processed/twitterAAE/unbalanced_ternary/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced_ternary/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced_ternary/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced_ternary/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced_ternary/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced_ternary/test_emb.npy \
  --dialect_col dialect_strict \
  --label_col label \
  --auto_class_weights \
  --out_dir data/results/twitterAAE_experiments/xgb_ternary_weighted
```

### 6. Three-way unsure-band VS-XGBoost

This setup creates an uncertainty band around the binary decision boundary and uses the groups **toxic / unsure / not toxic**.

#### low = 0.45, high = 0.55

```bash
python src/experiments/train_vs_xgboost_unsure.py \
  --train_csv data/processed/twitterAAE/unbalanced/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced/test_emb.npy \
  --dialect_col dialect_strict \
  --low 0.45 \
  --high 0.55 \
  --out_dir data/results/twitterAAE_experiments/vs_xgb_unsure_045_055
```

#### low = 0.40, high = 0.60

```bash
python src/experiments/train_vs_xgboost_unsure.py \
  --train_csv data/processed/twitterAAE/unbalanced/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced/test_emb.npy \
  --dialect_col dialect_strict \
  --low 0.40 \
  --high 0.60 \
  --out_dir data/results/twitterAAE_experiments/vs_xgb_unsure_040_060
```

#### low = 0.35, high = 0.65

```bash
python src/experiments/train_vs_xgboost_unsure.py \
  --train_csv data/processed/twitterAAE/unbalanced/train.csv \
  --val_csv data/processed/twitterAAE/unbalanced/val.csv \
  --test_csv data/processed/twitterAAE/unbalanced/test.csv \
  --train_emb data/embeddings/twitterAAE/unbalanced/train_emb.npy \
  --val_emb data/embeddings/twitterAAE/unbalanced/val_emb.npy \
  --test_emb data/embeddings/twitterAAE/unbalanced/test_emb.npy \
  --dialect_col dialect_strict \
  --low 0.35 \
  --high 0.65 \
  --out_dir data/results/twitterAAE_experiments/vs_xgb_unsure_035_065
```

## Notes

- `train_vs_xgboost.py` is the main script for the binary XGBoost + vector-scaling pipeline.
- `train_xgboost_ternary.py` is the semantic ternary XGBoost script.
- `train_vs_xgboost_unsure.py` is the uncertainty-band variant.
- Output folders contain summaries, predictions, and candidate settings for each run.