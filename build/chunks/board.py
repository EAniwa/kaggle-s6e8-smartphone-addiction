# ───────────────────────────── 19. Scoreboard ─────────────────────────────
md("## 19. The scoreboard")

code(r"""
board = pd.DataFrame(scoreboard)
board["gain_vs_floor"] = board["cv_auc"] - 0.5
board = board[["model", "cv_auc", "cv_std", "gain_vs_floor", "note"]]
display(board.style.format({"cv_auc": "{:.5f}", "cv_std": "{:.5f}",
                            "gain_vs_floor": "{:+.5f}"}, na_rep="—"))

best_row = board.loc[board["cv_auc"].idxmax()]
print(f"Best model: {best_row['model']}  (CV AUC {best_row['cv_auc']:.5f})")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(board["model"][::-1], board["cv_auc"][::-1], color="#4C72B0")
ax.axvline(0.5, color="red", ls="--", lw=1.2, label="floor (0.500)")
ax.set_xlim(0.45, 1.0)
ax.set_xlabel("cross-validated AUC")
ax.set_title("Every model, measured on identical folds")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
""")

md(r"""
**What this cell does and why it matters**

This is the honest summary of everything above, and the habit most worth stealing from this
notebook.

**Why the scoreboard is the point.** Notice what it makes possible: you can see at a glance
which steps actually paid off and by how much. That is what tells you where to spend your
remaining time. If logistic regression got you to 0.85 and LightGBM to 0.87, the jump to trees
was worth it. If the blend added 0.0003 while the fold standard deviation is 0.002, **the blend
did not really do anything** and you should not believe it.

**Always compare a gain against `cv_std`.** An improvement smaller than the fold-to-fold noise
is not an improvement, it is a coincidence you have chosen to believe. This single comparison
prevents an enormous amount of wasted effort.

**On CV versus the public leaderboard.** When you submit, your public leaderboard score will
differ from your CV score. That is expected — different data, different sample. The advice that
matters:

**Trust your cross-validation over the public leaderboard.** Your CV is computed on the full
training set across five folds. The public leaderboard is one score on a fraction of the test
data. It is a *noisier* measurement, and chasing it is how people overfit to the public split
and drop when the private scores are revealed.

The healthy pattern: use CV to decide what to build, and use the leaderboard only as a sanity
check that the two move in roughly the same direction. If CV improves and the leaderboard
worsens repeatedly, something is genuinely wrong — most likely a train/test distribution
difference — and that is worth investigating rather than ignoring.
""")

md(r"""
## 19b. What score should you actually expect?

A number without context is not information, so here is the real competitive landscape as of
mid-August 2026, taken from the public leaderboard of 2,273 teams:

| Position | Public AUC |
|---|---|
| 1st | 0.97134 |
| 10th | 0.97122 |
| 100th | 0.97113 |
| 500th | 0.96979 |
| Median | 0.96553 |

**Read the gaps, because they are the whole story:**

- 1st to 10th: **0.00012**
- 1st to 100th: **0.00021**
- 1st to median: **0.00581**

The top 100 teams are packed inside two ten-thousandths of AUC. **This competition is
saturated.** Independent analysis on the forum estimates the theoretical ceiling for this
label-generating process at roughly 0.9701 on the out-of-fold scale, against public stacks
already sitting at 0.9701 — meaning the remaining *real* signal is smaller than the gap between
1st and 20th place. Most of the ordering at the top is noise on the public split.

**What this means for you as a learner — and it is genuinely good news:**

1. **A straightforward, well-built pipeline like this one lands you in a respectable position.**
   The distance from a competent baseline to the leader is under 0.006 AUC.
2. **Grinding for the last 0.0001 teaches you very little.** It is measurement noise, not
   modelling skill. Two seeds of the same model swap dozens of leaderboard places.
3. **Understanding *why* each step helped is the durable skill.** It transfers to your next
   problem. A 0.0002 leaderboard climb does not.

**Expect your public leaderboard score to land roughly 0.001–0.0015 *above* your CV score.**
Competitors consistently report this offset, and nobody has reported a CV/leaderboard inversion —
adversarial validation on this data finds no meaningful distribution drift between train and
test. So your CV is trustworthy here. **Select your final submission by CV, not by public rank.**
""")
