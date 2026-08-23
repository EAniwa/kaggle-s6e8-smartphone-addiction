# ───────────────────────────── 26. Seed averaging ─────────────────────────────
md("## 26. Seed averaging — the cheapest reliable gain")

code(r"""
SEEDS = [42, 202, 1337]

seed_oofs, seed_tests, seed_scores = [], [], []

print(f"Running LightGBM with {len(SEEDS)} different random seeds")
print("-" * 46)
for sd in SEEDS:
    p = {**lgb_params, "random_state": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
    oof_s = np.zeros(len(y)); test_s = np.zeros(len(X_test_gbm))
    for tr, va in cv.split(X_gbm, y):
        m = lgb.LGBMClassifier(**p)
        m.fit(X_gbm.iloc[tr], y[tr],
              eval_set=[(X_gbm.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_s[va] = m.predict_proba(X_gbm.iloc[va])[:, 1]
        test_s += m.predict_proba(X_test_gbm)[:, 1] / N_SPLITS
    sc, _ = fold_auc_mean(oof_s, y, cv, X_gbm)
    seed_oofs.append(rank_by_fold(oof_s, cv, X, y))
    seed_tests.append(rankdata(test_s) / len(test_s))
    seed_scores.append(sc)
    print(f"  seed {sd:<6} fold-mean AUC {sc:.5f}")

oof_seedavg = np.mean(seed_oofs, axis=0)
test_seedavg = np.mean(seed_tests, axis=0)
sa_mean, sa_std = fold_auc_mean(oof_seedavg, y, cv, X_gbm)

print("-" * 46)
print(f"  mean of individual seeds   {np.mean(seed_scores):.5f}")
print(f"  spread across seeds        {np.ptp(seed_scores):.5f}")
print(f"  RANK-AVERAGED across seeds {sa_mean:.5f}")
print(f"  gain vs the average seed   {sa_mean - np.mean(seed_scores):+.5f}")
print(f"  gain vs the BEST seed      {sa_mean - max(seed_scores):+.5f}")

display(record(f"LightGBM x{len(SEEDS)} seeds", sa_mean, sa_std,
               "rank-averaged across seeds"))
""")

md(r"""
**What this cell does and why it matters**

**Seed averaging does not make your model better. It makes your model less random.** That
distinction is the whole point, and it is why this technique is both reliable and modest.

**Where the randomness comes from.** A LightGBM fit is not deterministic given the data. The seed
controls which rows each tree sees (`bagging_fraction`), which features it may split on
(`feature_fraction`), and tie-breaking during tree construction. Change the seed and you get a
genuinely different model — same data, same hyperparameters, different fitted result.

The `spread across seeds` figure above measures that directly. **This is a lower bound on how
much of your leaderboard position is luck**, and it is usually larger than people expect. On this
competition, a paired analysis found that two seeds of the same model swap places roughly half
the time, and that ±1 seed of noise spans about sixty leaderboard ranks between positions 10 and
100.

**Why averaging helps.** Each model's prediction is roughly *signal + noise*. The signal is shared
across seeds; the noise is not. Average N models and the noise partially cancels while the signal
survives. Standard error falls with roughly √N, which is why the gain is real but shrinking —
going from 1 to 3 seeds helps meaningfully, 3 to 10 much less.

**Compare against the *average* seed, not the best one.** This is the subtle trap. The best of
three seeds was chosen *after* seeing the scores, so it is optimistically biased — you cannot
know in advance which seed will win. The honest comparison is against what you would have got by
picking a seed blindly, which is the mean. If your gain looks negative against the best seed but
positive against the mean, seed averaging is still doing its job.

**Why rank-average rather than mean the probabilities?** Same reasoning as Section 23 — different
fits produce different output scales, and our metric only cares about order.

**When to use it.** Last, once modelling is finished. It costs N× the compute for a small,
dependable gain and teaches you nothing about your data. It is the final polish before submission,
never a substitute for the work in Sections 17 through 25.
""")
