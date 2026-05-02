# CSCI567-ML-Project

This repository contains code and experiments for dialect bias mitigation in toxicity detection, focusing on the TwitterAAE and HateXplain datasets. The project implements and evaluates several baselines and fairness-aware algorithms for text classification, with an emphasis on reducing false positives for African-American English (AAE) dialects.

## Project Structure

- **data/**: Processed datasets, embeddings, and raw data.
- **results/**: Model predictions, metrics, and plots.
- **src/**: Source code for baselines, experiments, and utilities.
- **model/**: Model vocabularies and count tables.

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
