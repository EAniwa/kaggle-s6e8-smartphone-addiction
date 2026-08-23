# ───────────────────────────── 19. Generator artifacts ─────────────────────────────
md("## 19. Feature engineering — mining the generator's fingerprints")

code(r"""
# Columns that represent hours. Guarded so the notebook still runs if the
# schema ever changes.
TIME_COLS = [c for c in ["daily_screen_time_hours", "social_media_hours",
                         "gaming_hours", "work_study_hours", "sleep_hours",
                         "weekend_screen_time"] if c in X.columns]

def add_artifact_features(df):
    out = df.copy()

    # (a) FIRST DECIMAL DIGIT. Synthetic generators leave rounding fingerprints;
    # the digit itself has no real-world meaning but can track the grid the
    # data was produced on.
    for c in TIME_COLS:
        d = (np.round(out[c] * 10) % 10)
        out[f"{c}__digit1"] = pd.Categorical(d.fillna(-1).astype(int),
                                             categories=list(range(-1, 10)))

    # (b) RESIDUAL SCREEN TIME. Time not accounted for by the named activities.
    # Where the generator's internal arithmetic does not balance, this is where
    # it shows up.
    parts = [c for c in ["social_media_hours", "gaming_hours", "work_study_hours"]
             if c in out.columns]
    if "daily_screen_time_hours" in out.columns and parts:
        out["other_screen"] = out["daily_screen_time_hours"] - out[parts].sum(axis=1)

    # (c) WEEKEND / WEEKDAY RATIO. In the source survey this ratio was bounded;
    # rows outside those bounds are physically implausible.
    if {"weekend_screen_time", "daily_screen_time_hours"} <= set(out.columns):
        out["weekend_ratio"] = (out["weekend_screen_time"]
                                / out["daily_screen_time_hours"].replace(0, np.nan))
    return out

X_fe = add_artifact_features(X_gbm)
X_test_fe = add_artifact_features(X_test_gbm)
new_feats = [c for c in X_fe.columns if c not in X_gbm.columns]
print(f"Added {len(new_feats)} features: {new_feats}\n")

# ---- Ablation: does the enriched feature set actually beat the baseline? ----
def run_lgb(Xtr_all, Xte_all, tag):
    oof = np.zeros(len(y)); te = np.zeros(len(Xte_all))
    for tr, va in cv.split(Xtr_all, y):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr_all.iloc[tr], y[tr],
              eval_set=[(Xtr_all.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(Xtr_all.iloc[va])[:, 1]
        te += m.predict_proba(Xte_all)[:, 1] / N_SPLITS
    mean_, std_ = fold_auc_mean(oof, y, cv, Xtr_all)
    print(f"  {tag:<28} {mean_:.5f}  (+/- {std_:.5f})")
    return oof, te, mean_, std_

print("Ablation")
print("-" * 56)
_, _, base_m, base_s = run_lgb(X_gbm, X_test_gbm, "baseline features")
oof_fe, test_fe, fe_m, fe_s = run_lgb(X_fe, X_test_fe, "+ generator artifacts")

fe_gain = fe_m - base_m
print("-" * 56)
print(f"  gain {fe_gain:+.5f}   fold noise +/- {base_s:.5f}")

if fe_gain > base_s:
    X_gbm, X_test_gbm = X_fe, X_test_fe
    print("\n  ADOPTED: the gain exceeds fold noise.")
else:
    print("\n  REJECTED: the gain is inside fold noise — keeping baseline features.")
    print("  (A negative result is a result. We do not ship what we cannot measure.)")
print(f"  Feature set for the rest of the notebook: {X_gbm.shape[1]} columns")
""")

md(r"""
**What this cell does and why it matters**

This is feature engineering aimed at a very specific target: **the data is synthetic, so it
carries fingerprints of the program that made it.**

**Why the first decimal digit could possibly matter.** In the real world it is meaningless — a
person with 7.6 screen-time hours is not fundamentally different from one with 7.7. But a
generator draws from distributions, rounds, and sometimes composes values arithmetically, and
those operations leave uneven residue on the final decimal place. If the addicted rate varies
measurably across digits, that is not a fact about smartphones. It is a fact about the code that
produced the file — and on a synthetic competition, that still scores.

**Residual screen time** applies the same idea to arithmetic consistency. If the generator drew
total screen time independently from its components, the two will not always reconcile, and the
size of the mismatch tracks how the row was produced.

**Two properties make these features safe to compute outside the CV loop**, and it is worth being
explicit since Section 13 was so emphatic about the opposite case:

1. They are **row-wise** — each output depends only on that row's own values.
2. They **never touch the target**.

That is the test. A transformation that uses only a row's own features leaks nothing. A
transformation that uses the target — like the encoding in Section 20 — must go inside the folds.
Learn to classify transformations this way and leakage stops being mysterious.

**The ablation is the real content of this section.** Notice we do not add the features and
assume they helped. We measure both feature sets on identical folds and compare the gain against
`base_s`, the fold noise — and we **reject** the change if the gain does not clear it.

This matters because published results on this competition show ordinary behavioural feature
engineering — ratios, sums, differences — landing between neutral and *negative*, while capacity
tuning was worth eighteen times more. **Feature engineering is not automatically good.** It adds
columns, which adds ways to overfit, and it must earn its place like anything else.
""")

# ───────────────────────────── 20. Target encoding ─────────────────────────────
md("## 20. Target encoding — the powerful, dangerous one")

code(r"""
def _smooth_means(keys, yy, prior, m):
    # Smoothed group means: a group with few rows is pulled toward the global
    # rate. m is the strength of that pull, measured in "virtual rows".
    g = pd.DataFrame({"k": np.asarray(keys), "y": yy}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


def target_encode(col_tr, y_tr, col_va, prior, m=20, inner_splits=5):
    # Validation rows: encode with statistics from the WHOLE training fold.
    # Training rows: encode with INNER out-of-fold statistics, so no row ever
    # sees its own label. Skipping this inner step is the classic mistake.
    enc_tr = np.full(len(col_tr), prior, dtype=float)
    inner = StratifiedKFold(inner_splits, shuffle=True, random_state=SEED)
    for i_tr, i_va in inner.split(np.zeros(len(col_tr)), y_tr):
        mp = _smooth_means(col_tr.iloc[i_tr], y_tr[i_tr], prior, m)
        enc_tr[i_va] = col_tr.iloc[i_va].map(mp).fillna(prior).to_numpy()
    mp_full = _smooth_means(col_tr, y_tr, prior, m)
    return enc_tr, col_va.map(mp_full).fillna(prior).to_numpy()


# Treat every numeric column as CATEGORICAL by stringifying it. On a synthetic
# grid this lets the model memorise the generator's lookup table.
TE_COLS = numeric_cols
key_tr = X[TE_COLS].round(2).astype(str)
key_te = X_test[TE_COLS].round(2).astype(str)
print(f"Target-encoding {len(TE_COLS)} numeric columns as categories")
print(f"  distinct values per column: "
      f"{ {c: int(key_tr[c].nunique()) for c in TE_COLS[:4]} } ...\n")

oof_te = np.zeros(len(y))
test_te = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(cv.split(X_gbm, y)):
    prior = y[tr].mean()
    Xtr = X_gbm.iloc[tr].copy(); Xva = X_gbm.iloc[va].copy(); Xte = X_test_gbm.copy()

    for c in TE_COLS:
        e_tr, e_va = target_encode(key_tr[c].iloc[tr], y[tr], key_tr[c].iloc[va], prior)
        _,    e_te = target_encode(key_tr[c].iloc[tr], y[tr], key_te[c], prior)
        Xtr[f"te__{c}"], Xva[f"te__{c}"], Xte[f"te__{c}"] = e_tr, e_va, e_te

    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_te[va] = m.predict_proba(Xva)[:, 1]
    test_te += m.predict_proba(Xte)[:, 1] / N_SPLITS
    print(f"  fold {fold}  AUC {roc_auc_score(y[va], oof_te[va]):.5f}")

te_mean, te_std = fold_auc_mean(oof_te, y, cv, X_gbm)
print(f"  {'-' * 42}")
print(f"  fold-mean AUC  {te_mean:.5f}  (+/- {te_std:.5f})")
print(f"  vs plain LightGBM: {te_mean - lgb_mean:+.5f}")

display(record("LightGBM + target encoding", te_mean, te_std,
               "numerics as categories, inner-OOF encoded"))
""")

md(r"""
**What this cell does and why it matters**

**Target encoding replaces a category with the average target value for that category.** It is
one of the most powerful techniques for tabular data and by far the easiest way to destroy your
validation score without noticing.

**Why stringify continuous columns?** It looks perverse — throwing away the ordering of a real
number. But this data is synthetic and sits on a **discrete grid**: the generator produced a
finite set of distinct values. Treating each value as its own category lets the model look up
"what fraction of people with exactly this screen time were addicted", effectively memorising the
generator's own table. On real-world continuous data this would overfit disastrously. Here it is
reported as the single most valuable transformation on the competition, worth about +0.003.

**The smoothing parameter `m` is doing important work.** From Section 8: a category with 3 rows
and a 100% positive rate is noise, not signal. The formula

`(sum + m × prior) / (count + m)`

pulls every group toward the global rate, and `m` sets how hard. A group with 3 rows and `m=20`
sits mostly at the prior; a group with 5,000 rows barely moves. **This is what makes the
technique usable at all** — without it, rare categories inject pure noise straight into your
features.

**The inner out-of-fold loop is the part that everyone gets wrong.** Consider encoding naively:
compute each category's mean target from the training fold, then assign it to those same training
rows. Every row's feature now contains a contribution from **its own label**. The model discovers
this instantly, leans on it entirely, and reports a spectacular validation score that evaporates
on the test set.

The fix is the structure above:

- **Validation and test rows** — encode using the full training fold. Safe, because those rows'
  labels were never in the statistics.
- **Training rows** — encode using an *inner* K-fold split, so each training row's value comes
  only from *other* training rows.

This is Section 13's leakage lesson applied at the feature level, and it is the reason target
encoding has such a bad reputation among people who have been burned by it. **The technique is
not dangerous. Implementing it without the inner loop is.**

**A note on what this model becomes.** Rather than merging these features into the main matrix,
we keep this as its own model. It uses a different *representation* of the same data — which,
per Section 22, is exactly the kind of diversity that earns weight in a stack. It joins the
ensemble as a base model in its own right.
""")
