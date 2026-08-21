# ───────────────────────────── 21. Next steps ─────────────────────────────
md(r"""
## 21. Where to go next

The pipeline above is complete and honest, but it is deliberately a *foundation* rather than a
maximally-tuned solution. Here is what to try next, **ordered by expected return on effort**.

### 1. Increase model capacity before engineering features

The most counter-intuitive finding from this competition: competitors measured 15 engineered
features as worth **+0.00042** at `num_leaves=15`, while simply changing `num_leaves` from 15 to
31 was worth **+0.00773** — eighteen times more. Re-running the same feature ablation at the
correct capacity turned the features **negative**. The feature engineering had been quietly
papering over an underfitting model.

**The lesson generalises: tune capacity first, then evaluate features.** Otherwise you cannot
tell whether a feature helped or merely compensated for a model that was too small. Also try
raising LightGBM's `max_bin` well above its default, reported here as worth about +0.002.

### 2. Target encoding of the numeric columns

Reported as the single most valuable feature transformation here (~+0.003): treat the continuous
columns as categorical and replace each value with a smoothed out-of-fold mean of the target.
This works because the synthetic data sits on a discrete grid, so the encoding effectively
memorises the generator's lookup table. **Compute it inside each fold** — Section 13 explains
exactly what happens if you do not.

### 3. Generator-artifact features

The synthetic data carries fingerprints of how it was produced. Two that competitors measured as
genuinely positive: the **first decimal digit** of each time column (the addicted rate varies
measurably across digits), and **residual screen time** (`daily_screen_time` minus the sum of
its component activities), which exposes rows where the generator's internal arithmetic does not
balance.

Note that ordinary behavioural feature engineering — ratios, sums, differences — is reported as
roughly **neutral to negative** here. This is not a general truth about tabular ML; it is a fact
about *this* dataset. Measure it on yours.

### 4. Add different model *families* to the blend

Diversity is what pays, and diversity means a different *kind* of model, not another variant.
Measured correlations here: LightGBM to XGBoost **0.997** (blend weight ≈ 0), GBDT to a neural
network with embeddings **0.974** (blend weight **0.22**). **CatBoost**, **XGBoost**, and
especially a **neural net** are worth adding; a second tuned LightGBM is not.

### 5. Stacking rather than weighted averaging

Replace the hand-scanned blend weight with a logistic regression trained on the OOF predictions.
On this data a plain average *loses* to the best single model while a learned stack wins — because
the stack can give a weak model a **negative** coefficient, using it as an error correction. An
average structurally cannot do that.

### 6. Seed averaging

Run your best model with several seeds and rank-average. It does not make the model better; it
reduces prediction variance for a small, reliable gain. Cheapest item on this list.

### What NOT to bother with

Worth stating explicitly, because these are the obvious things to try and they have been measured:

- **Do not add the original real-world dataset as extra training rows.** This is the standard
  Playground trick and here it **actively hurts** (about −0.0001 to −0.005 depending on weighting).
  The generator manufactured its own structure and repaired inconsistencies present in the source
  survey, so real-world rows are off-distribution. (The linked source dataset has also been removed
  from Kaggle.)
- **Do not add missing-indicator flags for numeric columns.** Zero gain — LightGBM already routes
  `NaN` natively — and they leak train/test split identity.
- **Do not over-tune the meta-model.** Sweeping the stacker's regularisation across two orders of
  magnitude moved the score by ±0.000004. Adding another base model beats tuning the blender.
- **Do not calibrate probabilities.** Section 10: it cannot change AUC.

---

## What to take away from this notebook

Above the specific techniques, five habits generalise to every ML problem you will meet:

1. **Understand the metric before optimising it.** ROC AUC measures ranking, which meant
   probabilities not labels, and made calibration irrelevant. Every metric has consequences
   like these.
2. **Establish a floor, then measure everything against it.** A score without a baseline is
   not information.
3. **Prevent leakage structurally, not by being careful.** Put preprocessing in a `Pipeline`
   and the entire class of bug becomes impossible.
4. **Compare every gain to your noise level.** An improvement smaller than `cv_std` is not an
   improvement.
5. **Start simple and make complexity earn its place.** Logistic regression before boosting,
   one model before an ensemble. If the complicated thing cannot beat the simple thing, the
   simple thing wins.

Good luck — and remember that a submission you understand is worth more than a higher score you
copied.
""")

# ───────────────────────────── write ─────────────────────────────
# nbformat 4.5+ requires a unique id on every cell.
for i, c in enumerate(CELLS):
    c["id"] = f"cell-{i:03d}"

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "notebook/s6e8_walkthrough.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
print(f"Wrote {out}")
print(f"  {len(CELLS)} cells total: {n_code} code, {n_md} markdown")
