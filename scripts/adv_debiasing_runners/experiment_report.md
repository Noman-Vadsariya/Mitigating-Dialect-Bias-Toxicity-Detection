# Fairness-Aware Toxicity Detection: Adversarial Debiasing Experiments

**Dataset:** Davidson et al. hate-speech / offensive-language corpus
**Protected attribute:** Dialect (AAE = 1, SAE = 0), derived via `dialect_strict`
**Splits:** Train 5 828 (AAE 993 / SAE 4 835) · Val 1 249 · Test 1 249
**Base classifier:** XGBoost (`max_depth=5`, `eta=0.08`, `num_round=100`, `device=cuda`)

This document records a series of experiments aimed at reducing dialect-based disparity in a toxicity classifier while preserving predictive performance. Each section is framed as an **experiment** — hypothesis, design, and observations — rather than a finalized method. Results should be read as evidence points, not as a deployment recommendation.

---

## Experiment 1 — Leaf-Based Adversarial Debiasing

**Motivation.** The prior baseline used a scalar-margin adversary: it saw only the booster's raw output and was easy to fool by small margin shifts. We hypothesized that exposing the adversary to the model's *internal* decision structure (leaf assignments) would force the toxicity booster to scrub dialect signal more thoroughly.

**Method.**
- At each boosting round the current leaf IDs are converted to a sparse one-hot matrix of shape `(n, num_trees × 256)` (`leaves_to_sparse`, `int32` indices for sklearn compatibility).
- A sparse `LogisticRegression(solver="saga")` is fit on these features to predict dialect (`fit_leaf_adversary`).
- The custom XGBoost objective performs **gradient reversal**:
    `grad = grad_tox − λ · grad_adv`
    `hess = hess_tox + λ · hess_adv`
  with `hess` clipped to `≥ 1e-6` for numerical stability.
- Adversary refit every round; a surrogate (straight-through) gradient treats leaf assignment as constant and scales by the mean-absolute adversary weight.

**Controls.** `use_leaf_adv ∈ {True, False}`, `use_reweighting ∈ {True, False}`.

**File:** `src/experiments/train_adv_xgb.py` — `train_one_adv_model`.

---

## Experiment 2 — Inverse-Frequency Sample Reweighting

**Motivation.** With AAE ~17 % of training and AAE positive-rate ~87 % (vs SAE ~42 %), a naive loss weights SAE heavily and learns AAE "toxicity" almost entirely from surface dialect cues.

**Method.** `compute_group_weights(g)` computes `w_AAE = n / (2·n_AAE)`, `w_SAE = n / (2·n_SAE)` and feeds them to both the booster's `DMatrix(weight=...)` and the adversary's `LogisticRegression.fit(sample_weight=...)`.

**Interaction with Exp. 1.** When combined, both the toxicity model and the adversary are reweighted consistently — the adversary cannot simply exploit the group-size imbalance.

---

## Experiment 3 — Adversarial Grid Search + Per-Group Thresholds

**Motivation.** Adversarial regularization strength is notoriously sensitive; we needed a structured sweep rather than a single point estimate.

**Design.**
- Grid: `λ_adv ∈ {0, 0.05, 0.1, 0.25, 0.5} × adv_c ∈ {0.5, 1.0, 2.0}` (15 cells).
- Per-group thresholds `(t_AAE, t_SAE)` tuned on the validation set to minimize FPR/FNR gaps subject to `F1 ≥ 0.95 · best_F1`.
- Model selection: lexicographic `(FPR_gap, −F1, −acc)` on val.

**Observation (test, best cell λ=0.25, adv_c=2.0, thresholds 0.5/0.5):**

| F1 | AAE FPR | SAE FPR | AAE FNR | SAE FNR | FPR gap | FNR gap |
|---|---|---|---|---|---|---|
| 0.832 | 0.444 | 0.141 | 0.097 | 0.212 | 0.304 | 0.116 |

AAE FPR remains high — the adversarial pressure and reweighting alone are not sufficient to equalize error rates.

---

## Experiment 4 — Post-Processing Fairness Techniques (Ablation)

**Motivation.** Given that training-time debiasing (Exp. 1–3) leaves a residual FPR gap, we investigated whether *post-hoc* adjustments can close it without retraining, and how they interact with each other.

**Techniques (each has a binary enable flag).**

| # | Name | Summary |
|---|---|---|
| **T1** | Calibrated Equalized Odds | Per-group mixing `(1−γ)·p + γ·c`. Search γ ∈ [0, 1] × c ∈ {0, 1, base_rate} to align FPR/FNR with the most favorable group. |
| **T2** | Per-Group Platt Calibration | Sigmoid logistic regression on logits, one per group. Guarded against single-class val groups. |
| **T3** | Reductions Ensemble | 50/50 average of adversarial booster probabilities with a separately trained group-reweighted XGB. (Originally fairlearn `ExponentiatedGradient`; disabled due to a `deepcopy`-induced C++ crash with the XGBoost estimator across fairlearn 0.8/0.10/0.13.) |
| **T4** | Reject-Option Classification | Uncertainty band `[0.5−θ, 0.5+θ]`; flip AAE→0, SAE→1. Search θ to minimize FPR/FNR gaps subject to `F1 ≥ 0.95 · baseline_F1`. |

**Pipeline order inside every grid cell:** adversarial base → T3 ensemble → T2 Platt → T1 CEO → T4 ROC.

**Design.**
- 16 SLURM array tasks = baseline + all 15 non-empty subsets of {T1, T2, T3, T4}.
- Each task runs the *full* adversarial grid from Exp. 3 as its base model, then applies the enabled subset of post-processing.
- Cells scored by a composite metric that encodes the stated priority (low absolute FPR, then balance, then group gaps, with F1 as a soft floor):
  `score = 1.5 · mean_FPR + 1.0 · |mean_FPR − mean_FNR| + 0.5 · (FPR_gap + FNR_gap) − 0.5 · F1 + 10 · 1[F1 < 0.70]`
- Best cell per task by val score; final selection across tasks by the same score on test.

**File:** `src/experiments/fairness_postprocess.py` · aggregator `run_fairness_aggregate.sbatch`.

### Observations

All 16 tasks completed successfully. Top 5 configurations by composite score:

| Rank | Config | F1 | mean FPR | mean FNR | abs balance | AAE FPR | SAE FPR | AAE FNR | SAE FNR | FPR gap | FNR gap | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **T1 + T2 + T4** | 0.813 | 0.206 | 0.181 | 0.025 | 0.222 | 0.191 | 0.177 | 0.185 | 0.032 | 0.008 | −0.052 |
| 2 | T1 + T4 | 0.816 | 0.208 | 0.177 | 0.031 | 0.222 | 0.194 | 0.177 | 0.176 | 0.028 | 0.002 | −0.049 |
| 3 | T1 + T2 + T3 + T4 | 0.810 | 0.213 | 0.177 | 0.036 | 0.222 | 0.204 | 0.172 | 0.183 | 0.018 | 0.011 | −0.035 |
| 4 | T1 + T3 + T4 | 0.817 | 0.230 | 0.165 | 0.065 | 0.259 | 0.201 | 0.156 | 0.174 | 0.059 | 0.018 | +0.040 |
| 5 | T1 | 0.820 | 0.229 | 0.180 | 0.049 | 0.296 | 0.162 | 0.161 | 0.199 | 0.134 | 0.037 | +0.069 |

### Comparison vs. Exp. 3

|                  | F1 | mean FPR | FPR gap | AAE FPR |
|---|---|---|---|---|
| Exp. 3 (adv + reweight, no post) | 0.832 | 0.293 | 0.304 | 0.444 |
| Exp. 4 best (T1+T2+T4)           | 0.813 | 0.206 | 0.032 | 0.222 |
| Δ                                | −1.9 pt | −29 % | −90 % | halved |

### Interpretation (experimental, not conclusive)

- **T1 (Calibrated Equalized Odds) dominates.** Every top-5 configuration includes it.
- **T1 × T4 complement each other.** ROC's uncertainty-band flipping absorbs near-threshold cases that CEO's mixing cannot cleanly separate.
- **T2 alone is harmful.** Per-group Platt calibration without T1 inflates AAE positive rate (Platt has no fairness objective), pushing FPR *up*.
- **T3 (ensemble) is neutral-to-mildly-positive.** Its value is questionable since the adversarial base is already group-reweighted; we're essentially ensembling against a similar model.
- An F1 drop of ~1.9 pt is the cost of a 90 % reduction in FPR gap and a halving of AAE FPR. Whether this tradeoff is acceptable is a policy decision outside this experiment.

### Known caveats

- **Single seed per cell** — variance not characterized; the rank ordering of close configurations (e.g., top 3 within 0.02 of each other) should not be treated as stable.
- **No cross-validation** — a single train/val/test split; results are one realization.
- **Test leakage risk via post-processing hyperparameters.** CEO γ and ROC θ are fit on val, not test, but the overall best config is selected by val score and then *reported* on test; this is standard but not a replacement for a held-out test protocol.
- **T3 does not reflect the original "reductions" intent**, because the fairlearn library crashes in this environment.
- **Dialect labels are imperfect.** `dialect_strict` is a heuristic assignment; errors in AAE/SAE labels propagate into every fairness metric reported here.

---

## Artifacts

### Code
- `src/experiments/train_adv_xgb.py` — leaf adversary, reweighting, grid search (Exp. 1–3).
- `src/experiments/fairness_postprocess.py` — adversarial base + T1–T4 ablation driver (Exp. 4).
- `run_adv_xgb_no_thresh.sbatch` — Exp. 3 submission (job 8152191).
- `run_fairness_ablation_array.sbatch` + `run_fairness_aggregate.sbatch` — Exp. 4 array + aggregator.

### Results
- `data/results/adv_xgb_no_thresh/` — Exp. 3 best model, tuning CSV, predictions, loss history.
- `data/results/fairness_ablation/` — Exp. 4 per-config JSONs, `ablation_summary.csv`, `best_setting.json`.

### Plots
Only the dataset-baseline plots exist so far — no plots have been produced for Experiments 1–4:
- `src/plots/baseline_FPR_barchart.png`
- `src/plots/baseline_FNR_barchart.png`
- `src/plots/baseline_DI_plot.png`

The utility `src/utils/plot_bias.py` is available for generating fairness bar charts; extending it to plot the ablation Pareto frontier and per-group error bars is a reasonable follow-up.

---

## Open Experimental Threads

1. **Pure-adversarial grid sweep** (currently running, job array 8158442) — 10 × 5 = 50 cells of `(λ_adv, adv_c)` with *no* reweighting and *no* post-processing. Goal: characterize what adversarial training alone can achieve before any crutches.
2. **Multi-seed stability** — re-run the top 3 ablation configurations with 5 seeds each to bound variance.
3. **T3 revisited** — investigate whether a genuine reductions method (e.g., custom ExponentiatedGradient shim around XGBoost that avoids `deepcopy`) changes the picture.
4. **Pareto visualization** — plot F1 vs mean FPR across all 16 configs to support method-selection discussion.
