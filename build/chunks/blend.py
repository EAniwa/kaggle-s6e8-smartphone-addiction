# ───────────────────────────── 18. Blending ─────────────────────────────
md("## 18. Blending — combining models with rank averaging")

code(r"""
from scipy.stats import rankdata

# Rank-normalise WITHIN EACH FOLD. Fold models can output probabilities on
# different scales (see Section 16); ranking inside the fold removes that,
# so the two models are combined on genuinely equal footing everywhere.
def rank_by_fold(oof, cv, X, y):
    out = np.zeros_like(oof, dtype=float)
    for _, va in cv.split(X, y):
        out[va] = rankdata(oof[va]) / len(va)
    return out

lr_r = rank_by_fold(oof_lr, cv, X, y)
lgb_r = rank_by_fold(oof_lgb, cv, X, y)

corr_rank = np.corrcoef(lr_r, lgb_r)[0, 1]
print(f"correlation between the two models' RANKS: {corr_rank:.4f}")
print("Lower correlation => more diversity => more to gain from blending.\n")

# Score every candidate weight with the SAME metric used everywhere else,
# so the blend row is directly comparable to the other scoreboard rows.
print(f"{'weight on LGBM':>15} {'fold-mean AUC':>15}")
print("-" * 32)
best_w, best_auc, best_std = None, -1.0, None
for w in np.arange(0, 1.01, 0.1):
    cand = (1 - w) * lr_r + w * lgb_r
    m, sd = fold_auc_mean(cand, y, cv, X)
    flag = ""
    if m > best_auc:
        best_auc, best_std, best_w, flag = m, sd, w, "  <-- best so far"
    print(f"{w:>15.1f} {m:>15.5f}{flag}")

oof_blend = (1 - best_w) * lr_r + best_w * lgb_r

# Test predictions are fold-averages already, so a single global rank is correct.
test_blend = ((1 - best_w) * rankdata(test_lr) / len(test_lr)
              + best_w * rankdata(test_lgb) / len(test_lgb))

gain = best_auc - lgb_mean
print(f"\nBest weight on LightGBM: {best_w:.1f}  ->  fold-mean AUC {best_auc:.5f}")
print(f"  LightGBM alone : {lgb_mean:.5f}")
print(f"  Gain from blend: {gain:+.5f}   (fold noise +/- {lgb_std:.5f})")
print("  VERDICT:", "blend wins" if gain > lgb_std else
      "gain is within fold noise — prefer the simpler single model")

display(record(f"Blend (rank avg, w={best_w:.1f})", best_auc, best_std,
               "logreg + lightgbm, rank-averaged per fold"))
""")

md(r"""
**What this cell does and why it matters**

**Why blending works at all.** Different model families make *different* mistakes. Logistic
regression misses interactions; LightGBM may overfit small regions. Averaging them lets errors
partially cancel while shared signal reinforces, so the blend can beat both inputs.

**The precondition is diversity, and it is worth measuring rather than assuming.** That is what
the rank correlation at the top reports. Two models correlated at 0.999 are effectively one
model and averaging them gains nothing. **This is why blending two LightGBMs with different
seeds helps far less than blending a GBDT with a linear model** — and why, in this competition
specifically, published results show a neural network earns real blend weight while a second
tuned GBDT earns almost none.

**Why rank-average rather than averaging the probabilities?** Our two models produce
probabilities on different *scales*. LightGBM is often confident, pushing predictions toward 0
and 1; logistic regression is more moderate. A plain average lets the more extreme model
dominate — not because it is better, but because its numbers are bigger.

Ranking strips the scale away and keeps only the ordering. Since Section 10 established that
**AUC depends on nothing but ordering**, discarding scale costs us exactly nothing and puts both
models on equal footing.

**Why we rank *within each fold*.** This is the subtle part, and it follows directly from the
scale problem flagged in Section 16. Each fold's model is trained separately and can output
probabilities on its own scale. Ranking the whole OOF vector at once would mix five different
scales together and blur the comparison. Ranking inside each fold normalises each model's output
against its own fold-mates, which is the only comparison that is actually meaningful.

**The verdict line is the point of this cell.** We compare the blend's gain against `lgb_std`,
the fold-to-fold noise. **If the gain is smaller than the noise, the blend has not been shown to
help**, and the correct decision is to ship the simpler single model. Complexity has to earn its
place with a margin bigger than the measurement error.

**A caution about how the weight was chosen.** We picked `best_w` by scanning weights and taking
whichever scored best — so the weight is *fitted to the validation data*, and the true gain is
slightly smaller than reported. With one parameter on a coarse grid the effect is minor, but the
principle matters: **every decision made by looking at your validation score is a small act of
fitting to it.** Make hundreds of them and your CV quietly detaches from reality.

**A finding from this competition worth knowing.** Competitors here report that plain and
rank-averaged blends actually *lose* to the best single model, while a **learned** meta-model
(logistic regression trained on the OOF predictions — "stacking") wins. The reason is
instructive: a learned combiner can assign a *negative* weight to a weak model, using it as an
error correction. An average can never do that. If you take this notebook further, stacking is
the upgrade — Section 21 covers it.
""")
