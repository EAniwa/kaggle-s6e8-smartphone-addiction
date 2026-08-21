# ───────────────────────────── 17. Capacity ─────────────────────────────
md("## 17. Capacity first — does a bigger model beat better features?")

code(r"""
# Sweeping capacity means refitting many models, so do it on a subsample.
# Relative ordering of hyperparameters is stable under subsampling even though
# absolute scores are not — which is exactly what a sweep needs.
SWEEP_N = min(len(X_gbm), 150_000)
sweep_idx = np.random.RandomState(SEED).choice(len(X_gbm), SWEEP_N, replace=False)
Xs, ys = X_gbm.iloc[sweep_idx].reset_index(drop=True), y[sweep_idx]
cv_s = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print(f"Sweeping num_leaves on {SWEEP_N:,} rows\n")
print(f"{'num_leaves':>11} {'fold-mean AUC':>15} {'+/- std':>10}")
print("-" * 39)

capacity_rows = []
for nl in [7, 15, 31, 63, 127]:
    p = {**lgb_params, "num_leaves": nl, "n_estimators": 1500}
    oof_c = np.zeros(SWEEP_N)
    for tr, va in cv_s.split(Xs, ys):
        m = lgb.LGBMClassifier(**p)
        m.fit(Xs.iloc[tr], ys[tr], eval_set=[(Xs.iloc[va], ys[va])],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_c[va] = m.predict_proba(Xs.iloc[va])[:, 1]
    mean_c, std_c = fold_auc_mean(oof_c, ys, cv_s, Xs)
    capacity_rows.append({"num_leaves": nl, "cv_auc": mean_c, "cv_std": std_c})
    print(f"{nl:>11} {mean_c:>15.5f} {std_c:>10.5f}")

cap = pd.DataFrame(capacity_rows)
best_leaves = int(cap.loc[cap["cv_auc"].idxmax(), "num_leaves"])
span = cap["cv_auc"].max() - cap["cv_auc"].min()

print(f"\nBest num_leaves: {best_leaves}")
print(f"Span across the sweep: {span:.5f}")
print(f"(Compare this against what feature engineering typically buys: ~0.0004)")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.errorbar(cap["num_leaves"], cap["cv_auc"], yerr=cap["cv_std"],
            marker="o", capsize=4, color="#4C72B0")
ax.axvline(best_leaves, color="#DD8452", ls="--", lw=1.2,
           label=f"best = {best_leaves}")
ax.set_xscale("log", base=2)
ax.set_xlabel("num_leaves (log scale)")
ax.set_ylabel("fold-mean AUC")
ax.set_title("Model capacity vs score")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# Adopt the winner for every LightGBM fitted from here on.
lgb_params["num_leaves"] = best_leaves
print(f"\nlgb_params['num_leaves'] set to {best_leaves} for the rest of the notebook.")
""")

md(r"""
**What this cell does and why it matters**

**This is the most valuable and least intuitive lesson in the notebook.**

Most people reach for feature engineering when they want a better score. On this competition,
someone measured both: 15 engineered features were worth **+0.00042**, while changing
`num_leaves` from 15 to 31 was worth **+0.00773** — eighteen times more.

Then the crucial part: when they re-ran the *same* feature ablation at the corrected capacity,
the features turned **negative**. The feature engineering had never been adding information. It
had been **compensating for a model too small to extract what was already there.**

**The general principle: tune capacity before you evaluate features.** Otherwise every feature
experiment is confounded. You cannot distinguish "this feature adds information" from "this
feature helps my underfitting model limp along", and you will happily keep features that a
correctly-sized model finds useless.

**How to read the curve.** It should rise, flatten, and eventually fall:

- **Rising** — the model is underfitting. It lacks the capacity to represent the patterns present.
- **Flat top** — the sweet spot. This is where you want to be.
- **Falling** — overfitting. The model now has enough capacity to memorise noise in the training
  folds, and it generalises worse.

**Why sweep on a subsample?** Twenty-five model fits on the full 691k rows is slow enough to
discourage you from doing it at all — which is how people end up skipping this step. The
*relative ordering* of hyperparameters is stable under subsampling even though absolute scores
drift, and ordering is all a sweep needs. **Do the search cheaply, then confirm the winner on
full data.**

**A caveat on what we just did.** We selected `num_leaves` using cross-validated scores and will
now report cross-validated scores using it. That is mild selection bias — the same effect flagged
for the blend weight in Section 23. With five candidates on one axis it is small. Sweep fifty
hyperparameters this way and your CV becomes meaningfully optimistic; that is when you need a
separate inner split (nested CV).
""")
