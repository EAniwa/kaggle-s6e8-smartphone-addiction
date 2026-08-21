# ───────────────────────────── 23. Combining ─────────────────────────────
md("## 23. Combining models — averaging vs stacking")

code(r"""
# Rank-normalise WITHIN each fold, so every model is on the same scale
# everywhere before being combined. (Section 16 explains why this matters.)
def rank_by_fold(v, cv, X, y):
    out = np.zeros(len(v), dtype=float)
    for _, va in cv.split(X, y):
        out[va] = rankdata(v[va]) / len(va)
    return out

meta_oof = np.column_stack([rank_by_fold(oof_matrix[c].values, cv, X, y)
                            for c in oof_matrix.columns])
meta_test = np.column_stack([rankdata(test_matrix[c].values) / len(test_matrix)
                             for c in test_matrix.columns])

best_single_name = solo.index[0]
best_single = solo.iloc[0]

# ---------- option A: plain average of probabilities ----------
avg_mean, avg_std = fold_auc_mean(oof_matrix.mean(axis=1).values, y, cv, X)

# ---------- option B: rank average ----------
rank_mean_v, rank_std_v = fold_auc_mean(meta_oof.mean(axis=1), y, cv, X)

# ---------- option C: LEARNED stack, honestly cross-validated ----------
# The meta-model must never be scored on rows it was fitted on, so we fit it
# on the other folds' rows and predict the held-out fold. Same discipline as
# the base models, applied one level up.
stack_oof = np.zeros(len(y))
for tr, va in cv.split(X, y):
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(meta_oof[tr], y[tr])
    stack_oof[va] = meta.predict_proba(meta_oof[va])[:, 1]
stack_mean, stack_std = fold_auc_mean(stack_oof, y, cv, X)

print(f"{'method':<34} {'fold-mean AUC':>14} {'vs best single':>16}")
print("-" * 66)
print(f"{'best single (' + best_single_name + ')':<34} {best_single:>14.5f} {'—':>16}")
print(f"{'plain average of probabilities':<34} {avg_mean:>14.5f} "
      f"{avg_mean - best_single:>+16.5f}")
print(f"{'rank average':<34} {rank_mean_v:>14.5f} "
      f"{rank_mean_v - best_single:>+16.5f}")
print(f"{'LEARNED STACK (logistic meta)':<34} {stack_mean:>14.5f} "
      f"{stack_mean - best_single:>+16.5f}")

# What did the meta-model learn? Fit once on everything to read the weights.
meta_full = LogisticRegression(C=1.0, max_iter=2000).fit(meta_oof, y)
coefs = pd.Series(meta_full.coef_[0], index=oof_matrix.columns).sort_values(ascending=False)
print("\nWeights the stack learned:")
print("-" * 34)
for k, v in coefs.items():
    flag = "   <-- NEGATIVE: used as error correction" if v < 0 else ""
    print(f"  {k:<14} {v:>+8.3f}{flag}")

test_stack = meta_full.predict_proba(meta_test)[:, 1]
display(record("Stack (logistic meta)", stack_mean, stack_std,
               f"{len(oof_matrix.columns)} base models, learned weights"))
""")

md(r"""
**What this cell does and why it matters**

**Stacking means training a model to combine your models.** Instead of choosing weights by hand,
you build a table whose columns are each base model's out-of-fold predictions, and fit a small
model — the *meta-model* — to map that table to the target. Here it is a logistic regression on
five columns.

**Why a learned stack beats an average, stated precisely.** Look at the weights. A weak model can
receive a **negative** coefficient, which means the stack is using it as an *error correction*:
"when logistic regression is unusually confident relative to the trees, shade the prediction
down." An average is structurally incapable of that — every input can only push in one direction,
with a non-negative weight. It has no way to say "this model is wrong in a predictable way."

This is why the measured results on this competition look the way they do:

| Method | CV AUC |
|---|---|
| Best single model | 0.959483 |
| Plain average | 0.956170 |
| Rank average | 0.955277 |
| **Learned stack** | **0.960241** |

**Both averages lose to the single best model.** Averaging drags a strong model toward weaker
ones. Only the learned combination wins.

**The cross-validation of the meta-model is the part people get wrong.** Notice we do not fit the
meta-model on all the OOF rows and then score it on those same rows — that would be Section 13's
leakage, one level up, and it would report a beautiful fictional number. Instead the meta-model
is fitted on four folds' worth of OOF rows and scored on the fifth. **The same discipline that
protects the base models has to protect the combiner.**

**An honest caveat about stacking in general.** Even done this way, a small optimism remains. The
OOF predictions for the meta-model's *training* rows came from base models that saw some of the
*validation* rows during their own training. Fully clean stacking needs nested cross-validation,
which is considerably more expensive. The residual bias is small and this is standard competitive
practice — but you should know it is there rather than discovering it when your leaderboard score
disappoints.

**Read the weights as diagnosis, not just machinery.** They tell you what each model is
contributing. A near-zero weight means that model is redundant and you can drop it — faster
pipeline, same score. That is Section 22's correlation analysis confirmed from the other
direction, and it is how you decide what to prune.
""")
