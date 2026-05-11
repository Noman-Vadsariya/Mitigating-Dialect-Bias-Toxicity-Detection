# CSCI567-ML-Project

This repository contains code and experiments for dialect bias mitigation in toxicity detection, focusing on the TwitterAAE and HateXplain datasets. The project implements and evaluates several baselines and fairness-aware algorithms for text classification, with an emphasis on reducing false positives for African-American English (AAE) dialects while maintaining overall performance.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Baselines](#baselines)
- [Experiments](#experiments)
  - [XGBoost + Vector Scaling Experiments](#xgboost--vector-scaling-experiments)
  - [Adversarial Debiasing Experiments](#adversarial-debiasing-experiments)

## Setup

1. **Clone the repository:**
  ```bash
  git clone https://github.com/Noman-Vadsariya/CSCI567-ML-Project.git
  cd CSCI567-ML-Project
  ```

2. **Create and activate a Python virtual environment (recommended):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

- **data/**: Processed datasets, embeddings, and raw data.
  - **results/**: Model predictions, metrics, and plots.
- **refactored_twitter_aae/**: Code for dataset preprocessing and dialect prediction
- **src/**: Source code for baselines, experiments, and plotting utils.


## Baselines

- **XGBoost Baseline** ([src/baselines/train_xgboost.py](src/baselines/train_xgboost.py)): Trains a standard XGBoost classifier on text embeddings.

  ```bash
  python src/baselines/train_xgboost.py
  ```

- **ToxicBERT Inference** ([src/baselines/toxicbert_inference.py](src/baselines/toxicbert_inference.py)): Uses the pre-trained ToxicBERT model for toxicity prediction.

  ```bash
  python src/baselines/toxicbert_inference.py
  ```

- **BERT Fine-tuning** ([src/baselines/finetune_bert.py](src/baselines/finetune_bert.py)): Fine-tunes a BERT model for toxicity classification.

  ```bash
  python src/baselines/finetune_bert.py
  ```

- **BERT Inference** ([src/baselines/bert_inference.py](src/baselines/bert_inference.py)): Runs inference using a fine-tuned BERT model.

  ```bash
  python src/baselines/bert_inference.py
  ```

## Experiments

- **XGBoost Re-weighting Experiment** ([src/experiments/xgb_reweighted.py](src/experiments/xgb_reweighted.py)): Weight AAE non-toxic examples more heavily to reduce false positives on AAE.

  ```bash
  python src/experiments/xgb_reweighted.py
  ```

- **Fairness-Aware Training with XgBoost** ([src/experiments/fair_xgb.py](src/experiments/fair_xgb.py)): Combines task (BCE) and fairness (gap penalty) losses.
  - BCE: optimizes toxicity detection.
  - Gap penalty: reduces AAE/SAE disparity.
  - Gamma: balances performance and fairness.


  ```bash
  python src/experiments/fair_xgb.py
  ```

- **XGBoost Group Thresholding** ([src/experiments/xgb_group_thresholding.py](src/experiments/xgb_group_thresholding.py)): Here we apply different decision thresholds for AAE and SAE at inference time. The idea was to reduce false positives for AAE by requiring higher confidence before predicting “toxic” for that group.

  ```bash
  python src/experiments/xgb_group_thresholding.py
  ```

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

- [src/experiments/train_vs_xgboost.py](src/experiments/train_vs_xgboost.py) is the main script for the binary XGBoost + vector-scaling pipeline.
- [src/experiments/train_xgboost_ternary.py](src/experiments/train_xgboost_ternary.py) is the semantic ternary XGBoost script.
- [src/experiments/train_vs_xgboost_unsure.py](src/experiments/train_vs_xgboost_unsure.py) is the uncertainty-band variant.




## Adversarial Debiasing Experiments

This section lists the main commands used to run the adversarial debiasing experiments for this project. 

> **Note:** All scripts in this section are run through Slurm. Before running any of the runner scripts in this section, update the `cd` path (e.g., `cd /scratch1/aqliang/CSCI567-ML-Project`) at the top of each script to match your own project directory on your cluster or local machine.


### 1. Linear-adversary XGBoost (TwitterAAE, balanced)

- **run_adv_eo_grid_array.sh**: Runs a grid search over adversarial and fairness hyperparameters for the equalized odds (EO) projection method on the balanced TwitterAAE split. Each configuration writes results to `data/results/adv_xgb_eo_grid/`.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_array.sh
  ```

- **run_adv_eo_grid_aggregate.sh**: Aggregates the per-lambda CSVs from the grid search, computes validation and test scores, and writes summary files (`grid_summary.csv`, `best_by_val_score.json`, etc.).

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_aggregate.sh
  ```

### 2. Linear-adversary XGBoost (TwitterAAE, unbalanced)

- **run_adv_eo_grid_unbalanced_array.sh**: Same as above, but for the unbalanced TwitterAAE split. Results are written to `data/results/adv_xgb_eo_grid_unbalanced/`.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_unbalanced_array.sh
  ```

- **run_adv_eo_grid_unbalanced_aggregate.sh**: Aggregates and summarizes the unbalanced run results.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_unbalanced_aggregate.sh
  ```

### 3. BERT-MLP adversary

- **run_adv_eo_grid_bert_array.sh**: Runs a grid search with a small MLP adversary on top of BERT embeddings. Set `DATA_TAG=balanced` or `DATA_TAG=unbalanced` to select the split. Results are written to the appropriate directory.

  ```bash
  DATA_TAG=balanced ./scripts/adv_runners/run_adv_eo_grid_bert_array.sh
  ```

- **run_adv_eo_grid_bert_aggregate.sh**: Aggregates grid search results for the BERT-MLP adversary. Honors the `DATA_TAG` environment variable.

  ```bash
  DATA_TAG=balanced ./scripts/adv_runners/run_adv_eo_grid_bert_aggregate.sh
  ```

- **run_adv_eo_grid_bert_plots.sh**: Retrains the best BERT-adversary configuration with per-round tracking and produces EO trend plots in `data/results/plots/`.
    
  ```bash
  DATA_TAG=balanced ./scripts/adv_runners/run_adv_eo_grid_bert_plots.sh
  ```

### 4. Three-class abstain variant (toxic / nontoxic / unsure)

- **run_adv_eo_grid_abstain_array.sh**: Runs grid search for the three-class abstain variant, adding per-group thresholds and deadbands, and sweeping a coverage floor.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_abstain_array.sh
  ```

- **run_adv_eo_grid_abstain_aggregate.sh**: Aggregates results for the abstain run, using a composite score (selective-F1, coverage, abstain-gap).

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_abstain_aggregate.sh
  ```

- **run_adv_eo_grid_abstain_summary_plots.sh**: Writes summary files and abstain-aware EO trend plots.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_abstain_summary_plots.sh
  ```

### 5. Ternary labels

- **run_adv_eo_grid_ternary_array.sh**: Runs grid search for ternary EO + projection on the `unbalanced_ternary` data split.

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_ternary_array.sh
  ```

- **run_adv_eo_grid_ternary_aggregate.sh**: Aggregates and summarizes ternary run results (macro-F1, mean FPR/FNR gaps).

  ```bash
  ./scripts/adv_runners/run_adv_eo_grid_ternary_aggregate.sh
  ```

### 6. Generalization Experiments

- **run_eo_trend_plot.sh**: Plots EO-vs-rounds trend for the default configuration (uses defaults in `src/plots/plot_eo_trend.py`).

  ```bash
  ./scripts/adv_runners/run_eo_trend_plot.sh
  ```

- **run_eo_trend_plot_ternary.sh**: Plots EO-vs-rounds trend for the ternary experiment.

  ```bash
  ./scripts/adv_runners/run_eo_trend_plot_ternary.sh
  ```

- **run_eo_trend_plot_variant.sh**: Plots EO trends for a configurable variant. Set the `VARIANT` environment variable to select the data/config (e.g., `balanced_indomain`, `unbalanced_indomain`, etc.).

  ```bash
  VARIANT=balanced_indomain ./scripts/adv_runners/run_eo_trend_plot_variant.sh
  ```

- **run_eval_hatexplain.sh**: Evaluates the best TwitterAAE config on HateXplain for out-of-domain generalization, reporting EO metrics under covariate shift.

  ```bash
  ./scripts/adv_runners/run_eval_hatexplain.sh
  ```