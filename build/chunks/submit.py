# ───────────────────────────── 20. Submission ─────────────────────────────
md("## 20. Creating the submission file")

code(r"""
candidates = {
    "logistic_regression": (lr_mean, test_lr),
    "lightgbm": (lgb_mean, test_lgb),
    "blend": (best_auc, test_blend),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, best_test_pred = candidates[best_name]

print(f"Selected model: {best_name}  (CV AUC {best_score:.5f})\n")

submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET: best_test_pred,
})

# ---------------------------- sanity checks ----------------------------
checks = {
    "row count matches sample_submission": len(submission) == len(sample_submission),
    "columns match sample_submission":     list(submission.columns) == list(sample_submission.columns),
    "ids match test set exactly":          submission[ID_COL].equals(test[ID_COL]),
    "no missing predictions":              submission[TARGET].notna().all(),
    "predictions within [0, 1]":           submission[TARGET].between(0, 1).all(),
    "predictions are not constant":        submission[TARGET].nunique() > 1,
    "no duplicate ids":                    not submission[ID_COL].duplicated().any(),
}
print("Submission sanity checks")
print("-" * 46)
for label, passed in checks.items():
    print(f"  [{'PASS' if passed else 'FAIL'}]  {label}")

if not all(checks.values()):
    raise ValueError("Submission failed validation — do not upload this file.")

submission.to_csv("submission.csv", index=False)
print(f"\nWrote submission.csv  ({len(submission):,} rows)")
display(submission.head())
print(submission[TARGET].describe())
""")

md(r"""
**What this cell does and why it matters**

**Model selection is automatic, by CV score.** We do not hand-pick a favourite; whichever model
scored best on cross-validation is the one that gets submitted. This removes a subtle bias —
people tend to submit the model they found most interesting to build rather than the one that
performs best.

**Why the sanity checks are not paranoia.** Every check above corresponds to a real, common way
people waste a submission:

| Check | The failure it catches |
|---|---|
| Row count | Dropped rows during processing — a very common merge bug |
| Column names | Kaggle rejects the file outright, sometimes cryptically |
| IDs match test | **Predictions silently misaligned with rows** — the worst one |
| No missing | `NaN` in the file causes rejection |
| Within [0, 1] | Raw model scores submitted instead of probabilities |
| Not constant | A bug produced one value everywhere; would score exactly 0.5 |
| No duplicate ids | A merge accidentally duplicated rows |

**The ID alignment check deserves emphasis.** If your predictions get reordered relative to the
test rows — easy to do with a stray `sort_values`, `groupby`, or `merge` — the file is
*perfectly valid* and *completely wrong*. It uploads fine, and it scores about 0.5. Every one of
these checks is cheap; a wasted submission on a deadline is not.

**Recall from Section 10 that we submit probabilities, not labels.** The `predictions within
[0,1]` and `not constant` checks are guarding exactly that. If you find yourself about to write
`.astype(int)` anywhere near this file, stop.

**To submit:** if you are running this inside a Kaggle notebook, commit the notebook and
`submission.csv` appears under the Output tab, ready to submit to the competition. Locally,
upload the file on the competition's Submit Predictions page. **The deadline is 31 August 2026,
23:59 UTC**, and you may submit multiple times per day — so submit early, verify the pipeline
works end to end, and improve from there.
""")
