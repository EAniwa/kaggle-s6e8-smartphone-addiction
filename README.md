# Playground S6E8 — Predicting Smartphone Addiction

A complete, executable machine learning walkthrough built as a **teaching notebook**, using a
live Kaggle competition as the case study.

| | |
|---|---|
| Competition | [playground-series-s6e8](https://www.kaggle.com/competitions/playground-series-s6e8) |
| Task | Binary classification |
| Target | `addicted_label` (70.9% positive) |
| Metric | ROC AUC on predicted probability |
| Data | 691,369 train rows × 12 features · 296,302 test rows |
| Deadline | **31 August 2026, 23:59 UTC** |
| Entry deadline | Same as final deadline — joinable until the last day |

## Files

```
notebook/s6e8_walkthrough.ipynb   the deliverable — 64 cells, upload to Kaggle
PLAN.md                           research findings and design rationale
build/build_notebook.py           regenerates the notebook
build/make_fixture.py             synthetic stand-in data for local testing
build/executed.ipynb              the notebook after a full run, outputs included
```

`build/executed.ipynb` is a committed copy of the notebook executed end-to-end against the
fixture. Open it to read the outputs and plots without running anything yourself.

## How to run it

**On Kaggle (recommended).** Create a new notebook in the competition, upload
`s6e8_walkthrough.ipynb`, and Run All. The data path is auto-detected; `submission.csv`
appears in the Output tab ready to submit.

**Locally.** Download the competition data into `data/`, then:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib && jupyter lab notebook/s6e8_walkthrough.ipynb
```

To regenerate the notebook after editing the builder:

```bash
python build/build_notebook.py notebook/s6e8_walkthrough.ipynb
```

## What it covers

21 sections building a strictly-measured scoreboard, from a constant baseline through logistic
regression to LightGBM and blending. Every code cell is followed by a markdown cell explaining
what happened and why.

The sections that matter most for a learner:

- **§10 Understanding ROC AUC** — derives the metric from its definition, proves it depends only
  on ranking, and shows the score you lose by submitting hard labels instead of probabilities.
- **§13 Leakage, demonstrated** — runs feature selection the wrong way on pure noise and
  manufactures ~0.24 AUC out of nothing, then shows the fix.
- **§12 Why cross-validation** — fits the same model on 8 random splits and shows the score
  moving by ~0.008 on luck alone.
- **§18 Blending** — and the honest verdict when the blend fails to beat a single model.
- **§19b What score to expect** — real leaderboard distribution, so your number means something.

## Verification

The notebook was executed end-to-end against a synthetic fixture matching the real schema
(column names, dtypes, ranges, categorical levels and per-column missingness rates). All 20 code
cells run clean and all 7 submission sanity checks pass.

Note the fixture is 12,000 rows against the real 691,369, so **the AUC values you see on real
data will differ** — expect roughly 0.96 rather than the ~0.90 the fixture produces. The fixture
exists to prove the code runs, not to predict the score.

## Notes on this competition specifically

Findings from the forum and leaderboard that contradict standard Playground advice:

- **Adding the original source dataset as extra training rows hurts.** Normally the single most
  reliable Playground trick; here it costs score, because the generator manufactured its own
  structure and repaired inconsistencies in the source survey. The source dataset has also been
  removed from Kaggle.
- **Missing-indicator flags for numeric columns are worthless.** LightGBM routes `NaN` natively,
  and missingness rates differ between train and test — so the flags leak split identity and can
  raise CV while hurting the leaderboard.
- **Model capacity beats feature engineering.** Competitors measured `num_leaves` 15→31 as worth
  18× more than 15 engineered features.
- **The competition is saturated.** 1st place 0.97134, 100th place 0.97113 — the top 100 fit
  inside 0.0002 AUC. Select final submissions by CV, not public rank.
