# Plan — Playground S6E8 "Predicting Smartphone Addiction" as an ML teaching case

## The competition (verified from kaggle.com, 2026-08-18)

| Fact | Value |
|---|---|
| Slug | `playground-series-s6e8` |
| Task | Binary classification |
| Target | `addicted_label` |
| Metric | **ROC AUC** on predicted probability |
| Submission | `id,addicted_label` — a probability in [0,1], not a hard 0/1 label |
| Columns | 29 |
| Files | train.csv, test.csv, sample_submission.csv (71.23 MB total) |
| Opened | 2026-08-01 |
| Closes | **2026-08-31, 23:59 UTC** |
| Entry deadline | Same as final deadline — joinable until the last day |
| Prizes | Kaggle swag only. No points, no medals. |
| Data origin | Synthetic, generated from the public "Smartphone Addiction Prediction Dataset" |

## Why this competition for teaching

Chosen deliberately over the higher-prize options:

- **Tabular + 29 columns** — the whole pipeline is legible. Nothing is hidden behind a CNN.
- **ROC AUC** — forces the learner to understand probabilities and ranking rather than accuracy, which is the single most common beginner misconception.
- **Runs on CPU in minutes** — the learner iterates, instead of waiting on a GPU queue.
- **Synthetic from real data** — clean enough to learn on, with enough signal to reward real feature work.
- **No medals** — removes the incentive to copy a leaderboard-chasing notebook, which is the wrong mode for learning.

## Teaching thesis

The notebook is built around one idea: **a Kaggle score is the output of a process, and the process is what's worth learning.** So it builds a strictly increasing scoreboard, where every step is justified by a measured change in cross-validated AUC — never by assertion.

Deliberate pedagogical decisions:

1. **Establish a floor first.** A constant predictor scores AUC 0.500 by construction. Every later number is measured against that, so "good" is never abstract.
2. **Teach the metric before optimising it.** A short section derives what ROC AUC actually measures — the probability a random positive outranks a random negative — with a plot. Optimising a metric you can't define is cargo-culting.
3. **Demonstrate leakage, don't just warn about it.** Show the wrong way (fit the scaler on all data, then split) and the right way (fit inside the fold), and show the score gap. A warning is forgettable; a number is not.
4. **Cross-validation over a single split.** Show the fold-to-fold spread so the learner internalises that a single holdout score is noisy.
5. **Out-of-fold predictions** as the honest estimate, and the basis for blending.
6. **Simple beats complex until proven otherwise.** Logistic regression before gradient boosting. If the boosting doesn't beat it, that's a finding, not a failure.

## Notebook structure

Every code cell is followed by a markdown cell explaining what happened and why, per the brief. Sections carry a heading cell first so the document is navigable.

| # | Section | Concept taught |
|---|---|---|
| 0 | Title & orientation | What the competition asks, how to read the notebook |
| 1 | Imports & config | Reproducibility, seeding, path handling for Kaggle vs local |
| 2 | Load data | Shapes, memory, id column handling |
| 3 | First look | dtypes, head, describe — reading a dataset cold |
| 4 | Target distribution | Class balance, why it dictates metric and CV choice |
| 5 | Missing values | Missingness as signal, not just a nuisance |
| 6 | Numeric features vs target | Distributions, separation, what a useful feature looks like |
| 7 | Categorical features vs target | Cardinality, encoding choices |
| 8 | Correlation & redundancy | Multicollinearity, why trees don't care and linear models do |
| 9 | **Understanding ROC AUC** | Threshold-independence, ranking, the 0.5 floor |
| 10 | Baseline 1: constant | The floor. AUC = 0.500 |
| 11 | Validation strategy | StratifiedKFold, fold variance, seed sensitivity |
| 12 | Leakage demonstration | Fit-transform inside vs outside the fold |
| 13 | Preprocessing pipeline | ColumnTransformer, why Pipeline prevents leakage structurally |
| 14 | Baseline 2: logistic regression | A real, defensible score from a simple model |
| 15 | Model 3: LightGBM | Why GBDTs dominate tabular, key hyperparameters explained |
| 16 | Feature importance | Gain vs split vs permutation; how importance misleads |
| 17 | Light hyperparameter tuning | Overfitting the validation set; when to stop |
| 18 | Blending | Rank-averaging, why it helps on AUC specifically |
| 19 | OOF review & scoreboard | The honest final estimate |
| 20 | Predict & write submission | Format, sanity checks, common submission errors |
| 21 | What to try next | Concrete, ordered follow-ups |

## Robustness decision

The competition's Data tab and public notebooks are gated behind joining, so exact column names could not be read directly. The notebook therefore **auto-detects** numeric vs categorical columns from dtypes rather than hardcoding a column list.

This is not a workaround — it is the better teaching pattern and the more reusable code. It means the notebook runs correctly against the real data on first execution regardless of the exact schema, and the learner gets a pipeline they can point at any tabular competition.

## Verification approach

The notebook will be executed end-to-end locally against a synthetic stand-in dataset built to the same contract (29 columns, mixed dtypes, binary `addicted_label`, an `id` column) before delivery, so every cell is known to run and every number in the narrative is real.

---

# Research findings (gathered by parallel agents, 2026-08-18)

## Confirmed schema

Train **691,369 × 14**, test **296,302 × 13**. The Data tab's "29 columns" is the sum across all
three files (14 + 13 + 2), not the width of any one.

9 numeric: `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`,
`work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`,
`weekend_screen_time`. 3 categorical: `gender` (Male/Female/Other), `stress_level`
(Low/Medium/High), `academic_work_impact` (Yes/No). Target 70.94% positive.

**Every feature has missing values**, 4%–19%. Single-feature AUCs: `daily_screen_time_hours`
0.8896, `weekend_screen_time` 0.8810, `social_media_hours` 0.8578, then a cliff to ~0.50–0.65.

## Leaderboard (2,273 teams)

1st 0.97134 · 10th 0.97122 · 100th 0.97113 · 500th 0.96979 · median 0.96553.
Top-100 spread **0.0002**. Estimated Bayes ceiling ≈0.9701 OOF. Public LB ≈ CV + 0.001–0.0015,
tightly correlated, no reported inversions. Adversarial validation finds no distribution drift.

## Findings that changed the notebook

| Finding | Effect on the notebook |
|---|---|
| Numeric missing-indicators carry zero gain and leak split identity | §6 corrected — hypothesis stated, then falsified with evidence |
| Rank-averaged blends lose to the best single model here | §18 rewritten to compare gain against fold noise and prefer the simpler model |
| Original dataset as extra rows *hurts* | §21 "What NOT to bother with" |
| Capacity beats feature engineering 18:1 | §21 reordered — capacity is now step 1, FE demoted |
| Competition is saturated | New §19b calibrating expectations against the real leaderboard |

## Bugs found by executing the notebook

Three defects that reading alone would not have caught:

1. **The single-split demo proved nothing** — it measured the positive rate of a *stratified*
   split, which is constant by construction. Rewritten to fit a real model across 8 splits and
   show the ~0.008 AUC spread.
2. **The scoreboard compared incompatible metrics** — logistic regression and LightGBM rows used
   fold-mean AUC while the blend row used pooled-OOF AUC, making the reported blend "gain" an
   artefact. Introduced `fold_auc_mean` as the single canonical metric.
3. **Blending on mis-scaled OOF vectors** — fold models with wildly different early-stopping
   points produce different output scales. Now rank-normalised *per fold*, and the fold-mean vs
   pooled-OOF gap is surfaced as an explicit diagnostic with a warning.


---

# Expansion: additional model families and stacking (2026-08-19)

Added on request, implementing the three recommendations from the model-selection discussion.

## New sections

| § | Content | Teaching point |
|---|---|---|
| 17 | `num_leaves` capacity sweep | Tune capacity before evaluating features; one-standard-error rule |
| 19 | XGBoost | Level-wise vs leaf-wise growth; `max_depth` ↔ `num_leaves` conversion |
| 20 | CatBoost | Ordered boosting, ordered target statistics, symmetric trees |
| 21 | Neural network (MLP) | A different function class is what earns blend weight |
| 22 | Diversity analysis | Entry condition is two-dimensional: decorrelated **and** strong |
| 23 | Averaging vs stacking | A learned meta-model can assign negative weights; an average cannot |

## Results on the fixture (12,000 rows)

| Method | fold-mean AUC | vs best single |
|---|---|---|
| Best single (LightGBM) | 0.90680 | — |
| Plain average | 0.90194 | −0.00485 |
| Rank average | 0.90472 | −0.00207 |
| **Learned stack** | **0.90759** | **+0.00079** |

This reproduces the pattern competitors reported on the real data: both averaging methods lose to
the best single model, and only the learned combination wins.

## Fixture-specific caveats handled in the text

The 12,000-row fixture is too small for two of the lessons to appear naturally, so the notebook
detects and explains both rather than asserting something the reader cannot see:

- **The capacity sweep is flat** — the whole span (0.00017) sits inside fold noise (0.00570). The
  code now reports this explicitly and applies the one-standard-error rule, choosing the simplest
  setting rather than chasing a noisy argmax.
- **Model correlations are understated** — LightGBM ↔ XGBoost measures 0.80 on the fixture versus
  0.997 reported on full data, because small-sample fold models are unstable. §22 warns that this
  is a sample-size artefact, not evidence of complementarity.
