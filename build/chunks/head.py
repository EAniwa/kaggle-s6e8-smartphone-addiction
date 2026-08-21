"""Generates the S6E8 teaching notebook as a .ipynb file.

Kept as a build script rather than hand-edited JSON so the notebook stays
diffable and regenerable.
"""
import json, sys, pathlib

CELLS = []

def _lines(text):
    # .ipynb "source" is a list of lines, each of which MUST retain its trailing
    # newline (except the last). Splitting without keepends concatenates every
    # line into one and breaks the cell.
    return text.strip("\n").splitlines(keepends=True)

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})

def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(src)})

# ───────────────────────────── 0. Orientation ─────────────────────────────
md(r"""
# Predicting Smartphone Addiction — a step-by-step machine learning walkthrough

**Kaggle Playground Series, Season 6 Episode 8** · [competition page](https://www.kaggle.com/competitions/playground-series-s6e8)

This notebook is built to be *read and executed in order*. It is not a leaderboard-chasing
solution — it is a complete, honest ML pipeline where every decision is explained and every
improvement is **measured** rather than asserted.

### The task

| | |
|---|---|
| **Problem type** | Binary classification |
| **Target** | `addicted_label` — is this person classified as addicted to their smartphone? |
| **Metric** | **ROC AUC** on a predicted *probability* |
| **Submission** | `id,addicted_label` where the second column is a float in [0, 1] |
| **Deadline** | 31 August 2026, 23:59 UTC |
| **Training data** | 691,369 rows x 12 features |
| **Test data** | 296,302 rows |
| **Class balance** | 70.9% positive |

### The single most important thing to notice

The metric is ROC AUC on a **probability**, not accuracy on a label. This changes everything
downstream. We are not being asked "is this person addicted, yes or no?" — we are being asked
to **rank** people by how likely they are to be addicted. A submission of all `0.0001` and a
submission of all `0.9999` score *identically* if the ordering is the same.

Beginners lose a lot of score by submitting hard `0`/`1` labels here. We will come back to
exactly why in Section 9.

### How to read this notebook

Every code cell is followed by a markdown cell explaining what just happened and **why it was
done that way**. Read the code, run it, then read the explanation. If you only read the prose
you will learn the vocabulary but not the craft.

### The plan

We build a scoreboard and add to it. Each model must earn its place by beating the previous
one on cross-validated AUC:

1. A constant prediction — the floor (AUC 0.500 by definition)
2. Logistic regression — a real, simple, defensible model
3. LightGBM — gradient boosting, which dominates tabular competitions
4. A blend of the above

Nothing gets called an "improvement" here without a number attached to it.
""")

md(r"""
### The features

Twelve features describe each person. They divide cleanly into two groups:

| Feature | Type | Range / values |
|---|---|---|
| `age` | numeric | 18–35 |
| `daily_screen_time_hours` | numeric | 0.5–15.0 |
| `social_media_hours` | numeric | 0.0–8.0 |
| `gaming_hours` | numeric | 0.0–4.0 |
| `work_study_hours` | numeric | 0.0–6.0 |
| `sleep_hours` | numeric | 4.5–9.0 |
| `notifications_per_day` | numeric | 20–250 |
| `app_opens_per_day` | numeric | 15–180 |
| `weekend_screen_time` | numeric | 0.5–17.6 |
| `gender` | categorical | Male / Female / Other |
| `stress_level` | categorical | Low / Medium / High |
| `academic_work_impact` | categorical | Yes / No |

**One property of this dataset shapes everything that follows: every single feature has missing
values**, at rates from 4% (`age`) to 19% (`social_media_hours`). There is no clean subset of
complete rows worth using — dropping rows with any missing value would discard most of the
data. Handling missingness well is not an optional refinement here, it is the core of the
problem. Section 6 is where that gets addressed properly.

*(The Data tab reports "29 columns". That is the sum across all three files — 14 in train, 13
in test, 2 in the sample submission — not the width of any one of them. Worth knowing so the
number does not confuse you.)*
""")

# ───────────────────────────── 1. Setup ─────────────────────────────
md("## 1. Setup and configuration")

code(r"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.base import clone

warnings.filterwarnings("ignore")

# Reproducibility. Every random operation in this notebook derives from this seed.
SEED = 42
np.random.seed(SEED)

# Cross-validation configuration, defined once and reused everywhere so that
# every model in the scoreboard is measured on exactly the same folds.
N_SPLITS = 5

TARGET = "addicted_label"
ID_COL = "id"

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)

print(f"pandas      {pd.__version__}")
print(f"numpy       {np.__version__}")
print(f"seed        {SEED}")
print(f"cv folds    {N_SPLITS}")
""")

md(r"""
**What this cell does and why it matters**

*Setting a seed.* `SEED = 42` makes every random operation reproducible — the same fold
splits, the same model initialisation, the same result on every run. This is not a formality.
Without a fixed seed you cannot tell whether a change in your score came from the change you
made or from random noise. **You cannot do science on a moving baseline.**

*Defining `N_SPLITS` once.* Every model we build will be evaluated on the *identical* folds.
Comparing a model scored on 5 folds against one scored on 10 is meaningless — the numbers are
not on the same scale. Defining it once, at the top, makes that mistake structurally impossible.

*Why `warnings.filterwarnings("ignore")`.* Purely for readable output in a teaching document.
In your own work, **leave warnings on**. Sklearn's convergence warnings in particular are
telling you something real about your model.

*`TARGET` and `ID_COL` as constants.* Naming these once means the rest of the notebook never
hardcodes a magic string, and adapting this notebook to a different competition is a two-line
change.
""")

# ───────────────────────────── 2. Load ─────────────────────────────
md("## 2. Loading the data")

code(r"""
# The notebook runs unchanged in two places: inside a Kaggle notebook (where the
# competition data is mounted read-only at /kaggle/input) and on your own machine
# (where you have unzipped the data into ./data). We detect which.
CANDIDATE_DIRS = [
    Path("/kaggle/input/playground-series-s6e8"),
    Path("data"),
    Path("../data"),
    Path("."),
]

DATA_DIR = next((d for d in CANDIDATE_DIRS if (d / "train.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv. If running locally, download the competition "
        "data and place train.csv / test.csv / sample_submission.csv in ./data/"
    )

print(f"Reading from: {DATA_DIR.resolve()}")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"train              {train.shape[0]:>8,} rows x {train.shape[1]:>3} cols")
print(f"test               {test.shape[0]:>8,} rows x {test.shape[1]:>3} cols")
print(f"sample_submission  {sample_submission.shape[0]:>8,} rows x {sample_submission.shape[1]:>3} cols")
print(f"\ntrain memory: {train.memory_usage(deep=True).sum() / 1024**2:,.1f} MB")
""")

md(r"""
**What this cell does and why it matters**

*Path auto-detection.* The `CANDIDATE_DIRS` loop means this exact notebook runs both on
Kaggle and on your laptop with no edits. Hardcoding `/kaggle/input/...` is the single most
common reason a downloaded notebook fails immediately on a local machine.

*Failing loudly.* If no data is found we raise an error with an actionable message rather
than letting the notebook continue and fail confusingly ten cells later. When you write
pipelines, **fail at the earliest point where you know something is wrong.**

*Checking shapes immediately.* Two things to look at the moment data loads:

1. **Does `test` have exactly one fewer column than `train`?** It should — the missing one is
   the target. If the gap is larger, there are features you cannot use at prediction time.
2. **Does `sample_submission` have the same row count as `test`?** If not, you have
   misunderstood the submission format, and you want to know that now rather than after
   training a model for an hour.

*Memory.* Worth a glance on any dataset. If train were several GB you would need to think
about dtype downcasting before doing anything else. At this size we can be relaxed.
""")

# ───────────────────────────── 3. First look ─────────────────────────────
md("## 3. First look at the data")

code(r"""
print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
display(train.head())

print("=" * 70)
print("COLUMN TYPES AND MISSINGNESS")
print("=" * 70)

overview = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "n_unique": train.nunique(),
    "n_missing": train.isna().sum(),
    "pct_missing": (train.isna().mean() * 100).round(2),
    "example": [train[c].dropna().iloc[0] if train[c].notna().any() else None
                for c in train.columns],
})
display(overview)
""")

md(r"""
**What this cell does and why it matters**

This is the *cold read* — the first thing to do with any dataset you have never seen. The
overview table answers four questions at once, and each one drives a later decision:

| Column | Question it answers | What it decides |
|---|---|---|
| `dtype` | Is this a number or a string? | Which preprocessing branch it goes down |
| `n_unique` | How many distinct values? | Encoding strategy for categoricals |
| `n_missing` | Is data absent? | Whether we need imputation |
| `example` | What does a value actually look like? | Catches misparsed columns |

**`n_unique` is the one beginners skip, and it is the most informative.** It silently tells
you things `dtype` cannot:

- `n_unique == 1` → the column is constant and carries **zero** information. Drop it.
- `n_unique == 2` → binary, regardless of whether it's stored as int, string, or bool.
- `n_unique == len(train)` → it's an identifier, not a feature. Feeding an ID to a model is a
  classic way to create a model that memorises rather than generalises.
- A numeric column with very low `n_unique` (say 5) is really **categorical wearing a numeric
  costume**. A rating from 1–5 is not a quantity where "4 is twice as much as 2" — treating it
  as continuous imposes a linear ordering the data may not support.

Read this table carefully before writing any modelling code. Ten minutes here saves hours later.
""")

# ───────────────────────────── 4. Target ─────────────────────────────
md("## 4. The target variable")

code(r"""
counts = train[TARGET].value_counts().sort_index()
rates = train[TARGET].value_counts(normalize=True).sort_index()

print("Target distribution")
print("-" * 40)
for cls in counts.index:
    print(f"  class {cls}:  {counts[cls]:>8,}  ({rates[cls]:6.2%})")

positive_rate = train[TARGET].mean()
imbalance_ratio = counts.max() / counts.min()
print(f"\npositive rate     {positive_rate:.4f}")
print(f"imbalance ratio   {imbalance_ratio:.2f} : 1")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar([str(c) for c in counts.index], counts.values,
       color=["#4C72B0", "#DD8452"], width=0.6)
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({rates.iloc[i]:.1%})", ha="center", va="bottom", fontsize=10)
ax.set_title("Class balance of addicted_label", fontsize=12)
ax.set_ylabel("rows")
ax.set_ylim(0, counts.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
""")

md(r"""
**What this cell does and why it matters**

The class balance is the first number that should influence your modelling strategy, because
it determines what "good" looks like.

**Why this matters more than it appears.** Suppose the positive rate were 5%. A model that
predicts "not addicted" for every single person would be **95% accurate** — and completely
worthless. This is why accuracy is a treacherous metric on imbalanced data, and it is a large
part of why this competition uses ROC AUC instead.

**AUC is insensitive to class balance**, which is exactly why Kaggle chose it. A constant
prediction scores 0.500 under AUC no matter how skewed the classes are. The metric cannot be
gamed by exploiting the base rate — you have to actually separate the classes.

**What the balance still changes:**

- **Stratified** cross-validation becomes important. If we split randomly, some folds could
  get noticeably different positive rates, making fold scores incomparable. `StratifiedKFold`
  preserves the class ratio in every fold. We use it in Section 11.
- If the imbalance were severe (say 100:1), we would consider class weighting or
  `scale_pos_weight` in LightGBM. Note these mainly shift the *calibration* of probabilities,
  not their *ordering* — and since AUC only cares about ordering, they often move AUC very
  little. Try it, measure it, keep it only if the number improves.

**The habit to build:** look at the target *before* you look at anything else. It defines the
problem.
""")

# ───────────────────────────── 5. Feature typing ─────────────────────────────
md("## 5. Splitting features into numeric and categorical")

code(r"""
# Everything that is not the target and not the id is a candidate feature.
feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]

# Auto-detect the two branches by dtype rather than hardcoding a column list.
numeric_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

# A numeric column with very few distinct values is categorical in disguise.
# We flag these for attention but keep them numeric for now.
LOW_CARD_THRESHOLD = 10
suspicious = [c for c in numeric_cols if train[c].nunique() <= LOW_CARD_THRESHOLD]

# A column with a single value carries no information at all.
constant_cols = [c for c in feature_cols if train[c].nunique(dropna=False) <= 1]

print(f"total features       {len(feature_cols)}")
print(f"  numeric            {len(numeric_cols)}")
print(f"  categorical        {len(categorical_cols)}")
print()
print("NUMERIC:", numeric_cols)
print()
print("CATEGORICAL:", categorical_cols)
print()
print(f"Numeric but low-cardinality (<= {LOW_CARD_THRESHOLD} distinct) — "
      f"possibly categorical in disguise:")
print("  ", suspicious if suspicious else "none")
print(f"\nConstant columns (zero information, safe to drop):")
print("  ", constant_cols if constant_cols else "none")
""")

md(r"""
**What this cell does and why it matters**

**Why auto-detect instead of hardcoding a column list?** Three reasons, and the third is the
important one:

1. It is shorter and cannot fall out of sync with the data.
2. It makes this notebook portable — point it at a different tabular competition and it still
   works.
3. **It cannot silently drop a column.** If you hand-write a list of 27 feature names and
   fat-finger one, that feature vanishes from your model and nothing warns you. Deriving the
   list from the data itself removes an entire class of bug.

**The `ID_COL` exclusion is not cosmetic.** An id is assigned arbitrarily, so it carries no
real-world information — but a sufficiently flexible model *will* find spurious patterns in
it, especially if ids were assigned in some order correlated with the target during dataset
construction. That inflates your validation score and collapses on the test set. **Always
drop identifiers from the feature matrix.**

**On "categorical in disguise".** A column like `sleep_quality` stored as 1–5 is numeric to
pandas, but treating it as continuous asserts that the gap from 1→2 is the same size as 4→5,
and that the values are meaningfully orderable at all. Sometimes true, sometimes not. This is
a genuine judgement call, and one worth testing empirically: encode it both ways, measure both,
keep the winner. Tree models are fairly robust to getting this wrong; linear models are not.

**Constant columns** are pure noise in the pipeline — they cost compute, add nothing, and can
break some scalers by producing zero variance. Worth checking for every single time.
""")

# ───────────────────────────── 6. Missing values ─────────────────────────────
md("## 6. Missing values")

code(r"""
missing = train[feature_cols].isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values in any feature column.")
    print("We still build imputation into the pipeline — see the explanation below.")
else:
    pct = (missing / len(train) * 100).round(2)
    display(pd.DataFrame({"n_missing": missing, "pct_missing": pct}))

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(missing))))
    ax.barh(missing.index[::-1], pct.values[::-1], color="#C44E52")
    ax.set_xlabel("% missing")
    ax.set_title("Missingness by feature")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    # Is missingness itself predictive? Compare the target rate when a value is
    # absent against when it is present.
    print("\nIs missingness informative?")
    print("-" * 58)
    print(f"{'feature':<30} {'rate|missing':>12} {'rate|present':>12}")
    print("-" * 58)
    for c in missing.index[:12]:
        m = train[c].isna()
        print(f"{c:<30} {train.loc[m, TARGET].mean():>12.4f} "
              f"{train.loc[~m, TARGET].mean():>12.4f}")
""")

md(r"""
**What this cell does and why it matters**

Two separate questions get asked here, and beginners usually only ask the first.

**Question 1: how much is missing?** This drives the mechanical choice. A feature that is 2%
missing can be imputed with little consequence. A feature that is 80% missing is mostly
fabrication after imputation, and you should consider dropping it or replacing it with a
simple "was this present?" binary flag.

**Question 2: is the missingness itself a signal?** This is the one that wins competitions.
If people who skipped the "hours of sleep" question are addicted at a 40% rate while those who
answered are at 20%, then **the act of not answering is one of your strongest features** — and
naive imputation destroys it completely by overwriting the absence with a plausible-looking
median.

The comparison table above tests exactly this. If the two columns differ substantially for
some feature, add an explicit missing-indicator column before imputing. Sklearn's
`SimpleImputer(add_indicator=True)` does this for you.

**An important empirical correction for *this* competition.** The reasoning above is sound in
general, and it is the right hypothesis to form. But competitors here have measured it, and on
this dataset it does not hold:

- Missingness carries **almost no target signal** — a model using only "how many values are
  missing in this row" scores about 0.502 AUC, essentially the floor.
- For numeric columns, missing-indicator flags contribute **exactly zero gain** in LightGBM,
  because LightGBM already routes `NaN` down its own branch natively. The indicator is redundant.
- Worse, missingness *rates differ between train and test* in all twelve columns. So indicator
  flags partly encode **which split a row came from** — which can raise your CV score while
  hurting your leaderboard score.

Keep the indicators for the **linear** model, where imputation genuinely destroys information
and there is no native `NaN` handling. Drop them for the **tree** models. This is a good
illustration of the general rule: form the hypothesis, then *measure it on your own data*, and
let the number decide.

**Why we build imputation into the pipeline even when nothing is missing.** The training data
having no gaps does not guarantee the *test* data has none — and on Kaggle a crash at
prediction time costs you a submission. Beyond that, keeping the pipeline structurally
complete means it stays correct if you later add engineered features that can produce `NaN`
(any ratio can divide by zero). **Defensive pipeline design costs nothing and prevents a whole
category of late-stage failure.**
""")

# ───────────────────────────── 7. Numeric features ─────────────────────────────
md("## 7. How numeric features relate to the target")

code(r"""
plot_cols = numeric_cols[:12]

if plot_cols:
    n = len(plot_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, plot_cols):
        for cls, colour, label in [(0, "#4C72B0", "not addicted"),
                                   (1, "#DD8452", "addicted")]:
            vals = train.loc[train[TARGET] == cls, col].dropna()
            if len(vals):
                ax.hist(vals, bins=40, alpha=0.55, color=colour,
                        label=label, density=True)
        ax.set_title(col, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Numeric feature distributions, split by target class",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.show()
else:
    print("No numeric features to plot.")
""")

md(r"""
**What this cell does and why it matters**

This plot is doing one job: **showing you which features carry signal, before you train
anything.**

The two coloured histograms are the distribution of a feature for addicted people (orange)
versus non-addicted people (blue). Read them like this:

- **Curves sit on top of each other** → the feature does not distinguish the classes on its
  own. Low individual value. *It may still matter in combination with others* — trees find
  interactions that no single-variable plot can reveal — so this is a weak signal to drop it,
  not a strong one.
- **Curves are visibly shifted apart** → real, usable signal. Expect this feature to rank
  highly in the importance plot in Section 16. If it does not, something is wrong with your
  pipeline and you should investigate.
- **Curves have different shapes rather than different centres** (one narrow, one wide) → the
  *variance* differs. A linear model will struggle with this; a tree will handle it fine.
  This is concrete evidence for preferring a tree-based model.

**Why `density=True` is essential here.** Without it, the taller bar is just whichever class
has more rows, and you would be looking at the class balance rather than the feature. Setting
`density=True` normalises each histogram to unit area so the *shapes* become comparable. Get
this wrong and every plot appears to show that the majority class dominates everywhere —
a genuinely misleading picture.

**A warning about this kind of plot.** It shows *marginal* relationships only — one feature at
a time, ignoring all others. A feature that looks useless here can be highly predictive in
interaction with another. Use these plots to build intuition and generate hypotheses, **never
to make final feature-selection decisions.** For that, measure the cross-validated score with
and without the feature.
""")

# ───────────────────────────── 8. Categorical features ─────────────────────────────
md("## 8. How categorical features relate to the target")

code(r"""
baseline_rate = train[TARGET].mean()

if categorical_cols:
    for col in categorical_cols[:8]:
        stats = (train.groupby(col, observed=True)[TARGET]
                 .agg(n="size", positive_rate="mean")
                 .sort_values("positive_rate", ascending=False))
        stats["lift_vs_baseline"] = stats["positive_rate"] - baseline_rate
        print(f"\n{'=' * 62}\n{col}   ({train[col].nunique()} distinct values)\n{'=' * 62}")
        display(stats.head(12).style.format(
            {"positive_rate": "{:.4f}", "lift_vs_baseline": "{:+.4f}", "n": "{:,}"}))
else:
    print("No categorical features — all columns are numeric.")

print(f"\nOverall positive rate (the baseline every group is compared against): "
      f"{baseline_rate:.4f}")
""")

md(r"""
**What this cell does and why it matters**

For a categorical feature, the equivalent of "are the distributions different?" is **"does the
target rate differ across categories?"**

The `lift_vs_baseline` column is the one to read. It is the group's positive rate minus the
overall rate, so:

- **Lift near 0** → this category behaves like the population average. No information.
- **Large positive or negative lift** → genuinely predictive. A category at +0.25 lift means
  members are 25 percentage points more likely to be addicted than average.

**Always read `lift` and `n` together — this is the critical habit.** A category with 3 rows
and a 100% positive rate has a huge lift and means essentially nothing; you are looking at
noise. With 3 samples, a 100% rate is what you would expect by chance reasonably often even if
the true rate were 50%. A category with 5,000 rows and +0.08 lift is far more trustworthy and
far more useful, despite the smaller number.

This is the intuition behind **smoothing** in target encoding: shrink small groups toward the
global mean in proportion to how little data supports them. A group with 3 rows should be
pulled almost entirely back to the baseline; a group with 5,000 barely moves.

**A caution about target encoding.** Replacing a category with its mean target value is
powerful and it is *the* classic leakage trap — you are putting the answer into the features.
Done naively it produces a spectacular validation score and a terrible leaderboard score. If
you use it, compute the encoding **inside each fold, using only that fold's training portion**.
We stick to one-hot encoding in this notebook, which is leak-free by construction.

**On cardinality.** Count the distinct values. One-hot encoding a 5-category column adds 5
columns; one-hot encoding a 5,000-category column adds 5,000 and will wreck both your memory
and your model. High-cardinality columns need a different strategy — LightGBM's native
categorical handling, or smoothed target encoding done correctly.
""")

# ───────────────────────────── 9. Correlation ─────────────────────────────
md("## 9. Correlation and redundancy")

code(r"""
if len(numeric_cols) >= 2:
    corr = train[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(min(13, 1 + 0.55 * len(numeric_cols)),
                                    min(11, 1 + 0.5 * len(numeric_cols))))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=8)
    ax.set_title("Correlation between numeric features", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.show()

    # Surface strongly correlated pairs explicitly — easier to read than a heatmap.
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (upper.stack().reset_index()
             .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "corr"}))
    strong = pairs[pairs["corr"].abs() > 0.7].sort_values("corr", key=abs, ascending=False)

    print("Feature pairs with |correlation| > 0.7:")
    display(strong if len(strong) else "  none — features are largely independent")

    # Correlation of each feature with the target: a quick univariate signal ranking.
    tgt_corr = (train[numeric_cols].corrwith(train[TARGET])
                .sort_values(key=abs, ascending=False))
    print("\nCorrelation with the target, strongest first:")
    display(tgt_corr.to_frame("corr_with_target").style.format("{:+.4f}"))
""")

md(r"""
**What this cell does and why it matters**

Two different things are being measured here, and conflating them is a common error.

**Feature-to-feature correlation (the heatmap)** detects *redundancy*. Two features
correlated at 0.95 carry nearly the same information. The consequence depends entirely on your
model:

- **Linear models (logistic regression) genuinely suffer.** With correlated inputs the
  coefficients become unstable — the model cannot tell which of the two deserves the credit, so
  it may assign a large positive weight to one and a large negative weight to the other. The
  predictions can still be fine while the coefficients become uninterpretable nonsense. This is
  *multicollinearity*.
- **Tree models barely care.** A tree picks one of the pair to split on and ignores the other.
  No instability. The only real cost is that **feature importance gets diluted** — two
  redundant features split the credit, so both look half as important as the underlying signal
  actually is. Keep this in mind when reading Section 16.

**Feature-to-target correlation** is a quick univariate signal ranking — a numeric version of
the histogram plot from Section 7.

**Its serious limitation:** Pearson correlation only detects *linear* relationships. A feature
with a perfect U-shaped relationship to the target — high risk at both extremes, low in the
middle — has a correlation of approximately **zero** while being extremely predictive. Trees
find that relationship immediately; correlation is blind to it.

**So: never drop a feature because its target correlation is low.** A near-zero correlation
means "no linear relationship", which is a much weaker statement than "no relationship".
""")

# ───────────────────────────── 10. Understanding ROC AUC ─────────────────────────────
md(r"""
## 10. Understanding the metric before optimising it

This section builds no model. It exists because **optimising a metric you cannot define is
cargo-culting**, and ROC AUC is the most commonly misunderstood metric in beginner Kaggle work.
""")

code(r"""
rng = np.random.default_rng(SEED)

# A small synthetic example where we control the answer, so the behaviour of the
# metric is unambiguous.
n = 2000
y_demo = rng.integers(0, 2, n)
# A "model" whose score is the class plus noise: informative but imperfect.
scores_good = y_demo + rng.normal(0, 1.0, n)
scores_random = rng.normal(0, 1.0, n)

print(f"AUC of an informative model : {roc_auc_score(y_demo, scores_good):.4f}")
print(f"AUC of pure noise           : {roc_auc_score(y_demo, scores_random):.4f}")

# ---- AUC computed from its definition, to show what it actually measures ----
# AUC = P(a randomly chosen positive is scored above a randomly chosen negative).
pos = scores_good[y_demo == 1]
neg = scores_good[y_demo == 0]

sample_pos = rng.choice(pos, 200_000)
sample_neg = rng.choice(neg, 200_000)
manual_auc = (sample_pos > sample_neg).mean()

print(f"\nAUC from sklearn                    : {roc_auc_score(y_demo, scores_good):.4f}")
print(f"AUC by sampling random (pos, neg) pairs: {manual_auc:.4f}")
print("\nThese agree. AUC IS that probability — nothing more exotic.")

# ---- Demonstrating that AUC depends only on ORDER, not on the values ----
print("\n" + "=" * 66)
print("AUC IS INVARIANT TO ANY ORDER-PRESERVING TRANSFORMATION")
print("=" * 66)
probs = 1 / (1 + np.exp(-scores_good))            # squash to (0, 1)
print(f"raw scores                        : {roc_auc_score(y_demo, scores_good):.6f}")
print(f"after sigmoid                     : {roc_auc_score(y_demo, probs):.6f}")
print(f"after squeezing into [0.49, 0.51] : {roc_auc_score(y_demo, 0.49 + 0.02 * probs):.6f}")
print(f"after replacing with ranks        : "
      f"{roc_auc_score(y_demo, pd.Series(probs).rank().values):.6f}")

# ---- The cost of submitting hard labels instead of probabilities ----
hard = (probs > 0.5).astype(int)
print("\n" + "=" * 66)
print("WHY YOU MUST SUBMIT PROBABILITIES, NOT 0/1 LABELS")
print("=" * 66)
print(f"AUC with probabilities  : {roc_auc_score(y_demo, probs):.4f}")
print(f"AUC with hard 0/1 labels: {roc_auc_score(y_demo, hard):.4f}   <-- worse")

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for s, lab in [(scores_good, "informative model"), (scores_random, "pure noise")]:
    fpr, tpr, _ = roc_curve(y_demo, s)
    ax.plot(fpr, tpr, lw=2, label=f"{lab} (AUC {roc_auc_score(y_demo, s):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing (AUC 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("The ROC curve")
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
""")

md(r"""
**What this cell does and why it matters**

This is the most important conceptual cell in the notebook. Four things were demonstrated:

**1. What AUC actually is.** The sampling experiment shows AUC equals *the probability that a
randomly chosen positive case is scored higher than a randomly chosen negative case*. That is
the whole definition. It follows immediately that:

- **AUC = 0.5** → your ranking is no better than a coin flip. This is the floor.
- **AUC = 1.0** → every positive is ranked above every negative. Perfect separation.
- **AUC < 0.5** → your ranking is *inverted*. Flip the sign of your predictions and you beat
  0.5. If you ever see this, you have a bug — very often a swapped `predict_proba` column.

**2. AUC only cares about order.** Sigmoid-squashing the scores, compressing them into the
razor-thin range [0.49, 0.51], and replacing them entirely with ranks **all give the identical
AUC**. Any monotonic transformation preserves it exactly.

This has a direct practical consequence: **do not waste time calibrating probabilities for this
competition.** Platt scaling, isotonic regression, and similar techniques make probabilities
better *calibrated* without changing their *order* — so they move AUC by exactly zero. They
matter enormously for log-loss and for real-world decisions. Here they are wasted effort.

**3. Hard labels throw away score.** Collapsing probabilities to 0/1 destroys all ranking
information *within* each predicted class. Everyone you labelled 1 is now tied, so the metric
can no longer tell your most-confident case from your least-confident one. The AUC drop is
visible above. **This is the single most common scoring mistake beginners make in AUC
competitions**, and the fix is one character: use `predict_proba` rather than `predict`.

**4. Reading the ROC curve.** Each point is one possible decision threshold. The curve traces
the trade-off between catching true positives and raising false alarms as you sweep that
threshold from strict to permissive. AUC is the area underneath — a single number summarising
performance across *every* threshold at once. That is precisely why it needs no threshold
chosen, and why it is such a natural fit for a competition metric.
""")

# ───────────────────────────── 11. Scoreboard + baseline ─────────────────────────────
md("## 11. Baseline 1 — the constant prediction (our floor)")

code(r"""
# The scoreboard. Every model must earn its place by beating what came before.
scoreboard = []

def record(name, cv_mean, cv_std=None, note=""):
    scoreboard.append({"model": name, "cv_auc": cv_mean, "cv_std": cv_std, "note": note})
    return pd.DataFrame(scoreboard)

y = train[TARGET].values

# The simplest possible "model": predict the same number for everybody.
constant_pred = np.full(len(y), 0.5)
constant_auc = roc_auc_score(y, constant_pred)

print(f"AUC of a constant prediction: {constant_auc:.4f}")

# It makes no difference WHICH constant we choose — order is unchanged.
for c in [0.001, 0.5, 0.99, train[TARGET].mean()]:
    print(f"  constant = {c:<8.4f} -> AUC {roc_auc_score(y, np.full(len(y), c)):.4f}")

display(record("Constant (0.5)", constant_auc, note="the floor — no information used"))
""")

md(r"""
**What this cell does and why it matters**

**Why bother scoring a model that does nothing?** Because "AUC 0.87" is a meaningless number
in isolation. It only becomes meaningful against a reference point, and this is the reference
point: **0.500, achieved by using no information whatsoever.**

Every subsequent number in this notebook is really a claim about how far above 0.5 we have
managed to climb. Establishing the floor explicitly is what makes the rest of the notebook
interpretable rather than a sequence of unanchored decimals.

Notice that **every constant scores exactly 0.5**, whether it is 0.001 or 0.99. This is the
order-invariance from Section 10 showing up in practice: all predictions tie, so no positive
is ranked above any negative more often than chance. A constant literally cannot beat 0.5.

**The scoreboard pattern is the real lesson here.** Keeping a running table of every model and
its cross-validated score means:

- You always know whether a change actually helped, rather than believing it did.
- You can see *how much* each step contributed, which tells you where to spend your remaining
  time.
- You never lose track of your own best model — surprisingly easy to do after four hours of
  experiments.

Adopt this in your own work. A notebook without a scoreboard is a notebook where you are
guessing.
""")

# ───────────────────────────── 12. CV strategy ─────────────────────────────
md("## 12. Validation strategy — why cross-validation, not a single split")

code(r"""
X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"StratifiedKFold, {N_SPLITS} folds, shuffled, seed={SEED}\n")
print(f"{'fold':<6} {'train n':>10} {'valid n':>10} {'train pos%':>12} {'valid pos%':>12}")
print("-" * 54)
for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    print(f"{i:<6} {len(tr_idx):>10,} {len(va_idx):>10,} "
          f"{y[tr_idx].mean():>12.4%} {y[va_idx].mean():>12.4%}")

print(f"\nThe positive rate is near-identical in every fold — that is what "
      f"'stratified' buys us.")

# How much does a SINGLE split vary depending on which split you happen to get?
print("\n" + "=" * 62)
print("WHY A SINGLE TRAIN/TEST SPLIT IS NOT ENOUGH")
print("=" * 62)
# Fit the SAME model on 8 different random 80/20 splits and watch the score move.
# A deliberately minimal, self-contained model: numeric columns only, imputed
# and scaled. We only need something whose score we can watch move.
quick = Pipeline([
    ("prep", ColumnTransformer([("num", Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler())]), numeric_cols)], remainder="drop")),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])

single_split_scores = []
for seed in range(8):
    X_a, X_b, y_a, y_b = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    m = clone(quick).fit(X_a, y_a)
    single_split_scores.append(roc_auc_score(y_b, m.predict_proba(X_b)[:, 1]))

single_split_scores = np.array(single_split_scores)
print("AUC of an IDENTICAL model across 8 different random splits:")
for i, sc in enumerate(single_split_scores):
    print(f"  split seed {i}:  {sc:.5f}")
print(f"\n  min  {single_split_scores.min():.5f}")
print(f"  max  {single_split_scores.max():.5f}")
print(f"  spread {single_split_scores.max() - single_split_scores.min():.5f}"
      f"   <-- pure luck of the draw, same model every time")
""")

md(r"""
**What this cell does and why it matters**

**Why not just split once into train and validation?** Because a single split gives you a
single noisy number, and you cannot tell noise from signal with a sample size of one.

Suppose you split once, score 0.8650, make a change, and score 0.8660. Did your change help?
**You have no idea.** The difference could easily be smaller than the random variation between
splits. Teams routinely chase improvements this way that are pure noise, and end up with a
model tuned to the quirks of one arbitrary validation set.

Cross-validation fixes this by giving you *five* scores instead of one. Now you have a mean
**and a standard deviation**, and the standard deviation tells you how large a difference has
to be before it means anything. If fold scores vary by ±0.004, then a +0.001 "improvement" is
noise and should be ignored.

**The three configuration choices, each deliberate:**

- **`Stratified`** — preserves the class ratio in every fold. Without it, folds get different
  positive rates purely by chance, adding variance that has nothing to do with your model. The
  table above confirms the rates match.
- **`shuffle=True`** — without it, `KFold` takes contiguous blocks of rows. If the data has any
  ordering (sorted by date, by id, by target during construction), contiguous folds are
  systematically different from each other and your validation becomes badly biased. **Always
  shuffle for tabular data with no time component.**
- **`random_state=SEED`** — fixes the folds so every model in the scoreboard is measured
  against *exactly* the same splits. Without this, comparing two models compares two different
  validation sets as well as two models, and the comparison is worthless.

**The one case where you must not do this:** if the data has a time dimension and you are
predicting the future, random folds let the model train on data from *after* the validation
period. That is leakage, it inflates your score, and it collapses on the real test set. Use
`TimeSeriesSplit` there. This competition has no time structure, so `StratifiedKFold` is right.
""")

# ───────────────────────────── 13. Leakage demo ─────────────────────────────
md("## 13. Data leakage — demonstrated, not just warned about")

code(r"""
from sklearn.feature_selection import SelectKBest, f_classif

# A deliberately extreme illustration: pure noise features, zero real signal.
# Any AUC meaningfully above 0.5 here is by definition an artefact.
n_rows, n_noise = 1200, 4000
X_noise = pd.DataFrame(rng.normal(size=(n_rows, n_noise)))
y_noise = rng.integers(0, 2, n_rows)

# ---------- THE WRONG WAY: select features using ALL the data, then split ----------
selector = SelectKBest(f_classif, k=20).fit(X_noise, y_noise)   # <-- sees every row
X_selected = selector.transform(X_noise)

tr_i, va_i = next(StratifiedKFold(5, shuffle=True, random_state=SEED)
                  .split(X_selected, y_noise))
wrong = LogisticRegression(max_iter=1000).fit(X_selected[tr_i], y_noise[tr_i])
auc_wrong = roc_auc_score(y_noise[va_i], wrong.predict_proba(X_selected[va_i])[:, 1])

# ---------- THE RIGHT WAY: select inside the fold, using training rows only ----------
sel_in = SelectKBest(f_classif, k=20).fit(X_noise.iloc[tr_i], y_noise[tr_i])
right = LogisticRegression(max_iter=1000).fit(
    sel_in.transform(X_noise.iloc[tr_i]), y_noise[tr_i])
auc_right = roc_auc_score(y_noise[va_i],
                          right.predict_proba(sel_in.transform(X_noise.iloc[va_i]))[:, 1])

print("Data is PURE NOISE. The only honest score is ~0.500.\n")
print(f"  WRONG (selected before splitting) : AUC {auc_wrong:.4f}   <-- fantasy")
print(f"  RIGHT (selected inside the fold)  : AUC {auc_right:.4f}   <-- honest")
print(f"\n  Leakage manufactured {auc_wrong - auc_right:+.4f} AUC out of thin air.")
""")

md(r"""
**What this cell does and why it matters**

**This is the most expensive mistake in applied machine learning, so we demonstrate it rather
than merely warning about it.**

The data above is pure random noise. There is no relationship between features and target —
none, by construction. The only truthful score is 0.500. Yet the "wrong" approach reports a
confidently higher number.

**Where did that score come from?** `SelectKBest` was shown *all* the rows, including the
validation rows, and asked to pick the 20 features most associated with the target. Among 4,000
random columns, some will correlate with the target by pure chance — and crucially, they
correlate **on the validation rows too**, because those rows were used to choose them. The
model then gets handed features pre-selected to look good on the very data it is about to be
judged on.

The information flowed backwards: knowledge of the validation set leaked into the training
procedure. Hence *leakage*.

**Why this is so dangerous in practice.** It does not announce itself. There is no error, no
warning — just a validation score that looks great. You believe your model is strong, you
submit, and the leaderboard score is far worse. Every hour spent tuning on a leaked validation
score was wasted, because you were optimising a fiction.

**The rule that prevents it:** *anything that learns from data — scalers, imputers, encoders,
feature selectors, target encoders — must be fitted on the training fold only, then applied to
the validation fold.* Never fit on the full dataset before splitting.

**And the structural fix, which is the real lesson:** use sklearn's `Pipeline`. When your
preprocessing lives inside a `Pipeline` and you pass that pipeline to `cross_val_score`,
**sklearn refits every step inside every fold automatically.** Leakage of this kind becomes
impossible by construction rather than by vigilance. That is why the next section builds one.
""")

# ───────────────────────────── 14. Preprocessing pipeline ─────────────────────────────
md("## 14. Building a leak-proof preprocessing pipeline")

code(r"""
numeric_pipeline = Pipeline([
    # add_indicator keeps the "this value was missing" signal from Section 6.
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    # handle_unknown="ignore" is essential: a category present only in test
    # would otherwise raise at prediction time.
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=10)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
], remainder="drop", verbose_feature_names_out=False)

# Fit once here purely to inspect the output width. The real fitting happens
# inside each CV fold, via the Pipeline.
_probe = preprocessor.fit_transform(X.head(5000), y[:5000])
print(f"input features           {len(feature_cols)}")
print(f"  numeric                {len(numeric_cols)}")
print(f"  categorical            {len(categorical_cols)}")
print(f"output columns after preprocessing: {_probe.shape[1]}")
print(f"\n(One-hot encoding expands categoricals; add_indicator may add missing flags.)")
""")

md(r"""
**What this cell does and why it matters**

Each preprocessing choice above is deliberate, and several are defensive against specific
failures that only appear at submission time.

**Median rather than mean imputation.** The median is robust to outliers. If a "daily screen
time" column contains a data-entry error of 9999 hours, the mean is dragged upward and every
imputed value is wrong; the median is essentially unaffected.

**`add_indicator=True`.** This is Section 6's finding turned into code. It appends a binary
column marking where a value was missing, so the model can still learn from *the fact of*
missingness even after the hole has been filled. If missingness is informative, this recovers
signal that imputation would otherwise destroy.

**`StandardScaler`.** Logistic regression is a distance-based, regularised model: it is
sensitive to feature scale. A feature measured in thousands will dominate the penalty term
against one measured in decimals, purely because of units. Scaling puts everything on
comparable footing. *Tree models do not need this at all* — they split on ordering, and any
monotonic rescaling gives identical trees. We keep it because our first real model is linear.

**`handle_unknown="ignore"` — and this one will save you a submission.** If a category appears
in `test.csv` but never in `train.csv`, the default encoder **raises an exception at prediction
time**, after you have already trained everything. With `ignore`, the unseen category is
encoded as all-zeros and prediction proceeds. On a competition deadline this is the difference
between a submission and no submission.

**`min_frequency=10`.** Categories appearing fewer than 10 times get grouped into a single
"infrequent" bucket. This directly addresses the small-group noise problem from Section 8:
rather than giving a 3-row category its own column to overfit on, we pool the rare ones.

**Why `ColumnTransformer` at all?** It routes numeric and categorical columns down separate
paths and reassembles the result. Without it you would be slicing DataFrames by hand and
stitching arrays together — which works until you make one indexing error and silently train
on misaligned columns.

**The payoff.** Because all of this lives inside a `Pipeline`, when we pass it through
cross-validation every step is refitted on each training fold. The leakage demonstrated in
Section 13 is now **structurally impossible**, not merely something we are being careful about.
""")

# ───────────────────────────── 15. Logistic regression ─────────────────────────────
md("## 15. Baseline 2 — logistic regression")

code(r"""
def fold_auc_mean(oof, y, cv, X):
    # THE canonical scoreboard metric: AUC computed separately within each fold,
    # then averaged. Scoring per-fold matters because different folds can produce
    # predictions on different scales; pooling them into one vector and scoring
    # once would compare incomparable numbers. See the explanation below.
    scores = [roc_auc_score(y[va], oof[va]) for _, va in cv.split(X, y)]
    return float(np.mean(scores)), float(np.std(scores))


def cross_validate_model(model, X, y, X_test, cv, name, verbose=True):
    # Run stratified CV, returning out-of-fold predictions and averaged test predictions.
    #
    # oof[i] is the prediction for row i made by a model that never saw row i
    # during training, so roc_auc_score(y, oof) is an honest generalisation estimate.
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / cv.get_n_splits()

        fold_auc = roc_auc_score(y_va, oof[va_idx])
        fold_scores.append(fold_auc)
        if verbose:
            print(f"  fold {fold}  AUC {fold_auc:.5f}")

    overall = roc_auc_score(y, oof)
    if verbose:
        print(f"  {'-' * 34}")
        print(f"  mean  {np.mean(fold_scores):.5f}  (+/- {np.std(fold_scores):.5f})")
        print(f"  OOF   {overall:.5f}")
    return oof, test_pred, np.mean(fold_scores), np.std(fold_scores), overall


logreg = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
])

print("Logistic regression")
oof_lr, test_lr, lr_mean, lr_std, lr_oof = cross_validate_model(
    logreg, X, y, X_test, cv, "logreg")

display(record("Logistic regression", lr_mean, lr_std, "linear, scaled, one-hot"))
""")

md(r"""
**What this cell does and why it matters**

**Why start with a simple linear model rather than going straight to gradient boosting?**

1. **It establishes a real baseline.** If your fancy model cannot beat logistic regression,
   the fancy model is not working and you need to know that immediately. This happens more
   often than people admit.
2. **It is fast**, so it surfaces pipeline bugs in seconds rather than after a long training
   run.
3. **It tells you about the problem's shape.** If logistic regression gets within a whisker of
   LightGBM, the relationships are mostly linear and additive. If it is far behind, there are
   strong interactions and non-linearities — which tells you where to spend your effort.

**Out-of-fold (OOF) predictions are the key idea in this cell.** For every row in the training
set, we store the prediction made by the one model that did *not* see that row during training.
The result is a full-length prediction vector where no prediction was informed by its own
answer.

This gives us three things at once:
- **An honest score.** `roc_auc_score(y, oof)` estimates generalisation on all training rows.
- **A diagnostic.** We can inspect exactly which rows the model gets wrong.
- **A blending input.** Section 18 combines models using their OOF predictions — and this only
  works because the OOF predictions are unleaked.

**Why average test predictions across folds.** Each fold produces a model trained on 80% of the
data. Rather than discarding four of them, we average all five predictions. This is a small
ensemble in its own right and is reliably a little better and more stable than any single
model. It costs nothing extra — we trained them anyway.

**Read the fold spread, not just the mean.** The `+/-` figure is your noise floor. If folds
vary by ±0.003, then any future "improvement" smaller than about 0.003 is indistinguishable
from luck. **This number tells you when to stop tuning**, and it is the number most people
ignore.

**The fold-mean vs pooled-OOF gap is a diagnostic, not a rounding artefact.**

Two ways to summarise cross-validation:
- **Fold-mean**: score each fold separately, average the five numbers. *This is our scoreboard
  metric.*
- **Pooled OOF**: concatenate all five folds' predictions into one vector, score it once.

When fold models are well-behaved these agree closely. When they diverge, it tells you
something specific and useful: **the fold models are producing predictions on different
scales.** Fold 0 might output probabilities spread across 0.1–0.9 while fold 3 outputs 0.4–0.6.
Each is internally well-ordered — so each scores well on its own — but pooling them interleaves
two different scales, and the combined ranking is worse than any individual fold's.

This is exactly what early stopping can cause: if one fold stops at 300 trees and another at 9,
those two models are not comparable in output scale.

**Fold-mean is the more robust summary**, which is why it is our canonical metric, and why we
rank-normalise *per fold* before blending in Section 18. If you see a large gap, do not panic —
but do check whether your early stopping is behaving erratically across folds.
""")

# ───────────────────────────── 16. LightGBM ─────────────────────────────
md("## 16. Model 3 — LightGBM (gradient boosted trees)")

code(r"""
import lightgbm as lgb

# LightGBM handles categoricals natively, which is better than one-hot for trees.
# It needs them as pandas 'category' dtype, with consistent categories across
# train and test.
X_gbm = X.copy()
X_test_gbm = X_test.copy()

for c in categorical_cols:
    combined = pd.concat([X_gbm[c], X_test_gbm[c]]).astype("category")
    cats = combined.cat.categories
    X_gbm[c] = pd.Categorical(X_gbm[c], categories=cats)
    X_test_gbm[c] = pd.Categorical(X_test_gbm[c], categories=cats)

lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,      # small steps; more trees needed but generalises better
    num_leaves=48,           # main capacity control for LightGBM
    min_child_samples=40,    # a leaf needs >=40 rows: blocks tiny overfit leaves
    feature_fraction=0.8,    # each tree sees 80% of features (decorrelates trees)
    bagging_fraction=0.8,    # each tree sees 80% of rows
    bagging_freq=1,
    reg_lambda=1.0,          # L2 penalty on leaf weights
    n_estimators=3000,       # an upper bound; early stopping picks the real number
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(X_test_gbm))
fold_scores, best_iters = [], []

print("LightGBM")
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_gbm, y)):
    X_tr, X_va = X_gbm.iloc[tr_idx], X_gbm.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    oof_lgb[va_idx] = model.predict_proba(X_va)[:, 1]
    test_lgb += model.predict_proba(X_test_gbm)[:, 1] / N_SPLITS

    fold_auc = roc_auc_score(y_va, oof_lgb[va_idx])
    fold_scores.append(fold_auc)
    best_iters.append(model.best_iteration_)
    print(f"  fold {fold}  AUC {fold_auc:.5f}   best_iter {model.best_iteration_}")

lgb_mean, lgb_std = np.mean(fold_scores), np.std(fold_scores)
pooled_oof = roc_auc_score(y, oof_lgb)
print(f"  {'-' * 42}")
print(f"  fold-mean AUC  {lgb_mean:.5f}  (+/- {lgb_std:.5f})   <-- scoreboard metric")
print(f"  pooled OOF AUC {pooled_oof:.5f}")
print(f"  best_iteration: mean {np.mean(best_iters):.0f}, "
      f"range {min(best_iters)}-{max(best_iters)}")

# DIAGNOSTIC: a large gap between these two means the fold models are producing
# predictions on different scales. See the explanation below.
gap = lgb_mean - pooled_oof
if gap > 0.005:
    print(f"\n  WARNING: fold-mean exceeds pooled OOF by {gap:.5f}.")
    print(f"  The fold models are on inconsistent scales — read the note below.")

display(record("LightGBM", lgb_mean, lgb_std, "native categoricals, early stopping"))
""")

md(r"""
**What this cell does and why it matters**

**Why gradient boosting dominates tabular problems.** A single decision tree is weak. Boosting
builds trees *sequentially*, where each new tree is fitted to the errors the previous ones are
still making. Hundreds of weak learners compose into a strong one. Critically, trees capture
**interactions** (the effect of screen time depends on age) and **non-linearities** (risk rises
sharply after a threshold) automatically, with no manual feature engineering. That is precisely
what logistic regression cannot do, and it is why GBDTs win essentially every tabular
competition.

**The hyperparameters that actually matter, and what each trades off:**

| Parameter | What it controls | Trade-off |
|---|---|---|
| `learning_rate` | Step size per tree | Lower generalises better but needs more trees |
| `num_leaves` | Tree complexity | **The main overfitting dial in LightGBM** |
| `min_child_samples` | Minimum rows per leaf | Higher = more conservative, less noise-fitting |
| `feature_fraction` | Features per tree | Decorrelates trees, acts as regularisation |
| `bagging_fraction` | Rows per tree | Same idea, on the row axis |
| `reg_lambda` | L2 on leaf weights | Direct shrinkage of extreme predictions |

**`num_leaves` deserves special attention.** LightGBM grows trees *leaf-wise* rather than
level-wise: it repeatedly splits whichever leaf promises the largest gain. This is faster and
more accurate, but it produces deep, unbalanced trees that overfit readily. `num_leaves` is
your primary defence. If you come from XGBoost expecting `max_depth` to be the main dial, this
is the key difference to internalise.

**Early stopping is doing real work here.** We set `n_estimators=3000`, but we do not want 3000
trees — we want however many stop helping. Early stopping watches validation AUC and halts
after 200 rounds without improvement, keeping the best iteration. So the tree count is
*learned* rather than guessed, separately for each fold.

**One important caveat, stated plainly:** early stopping uses the validation fold to decide
when to stop, which means that fold is no longer perfectly untouched. This makes the reported
score very slightly optimistic. For this competition the effect is small and it is standard
practice. In a high-stakes setting you would use a separate inner split for early stopping —
nested cross-validation.

**Native categorical handling** is genuinely better than one-hot for trees. One-hot forces the
tree to isolate one category per split; native handling lets it partition categories into
arbitrary groups in a single split. Fewer splits, better use of depth. Note the careful
alignment of categories across train and test above — mismatched category codes are a silent
and nasty source of wrong predictions.
""")
