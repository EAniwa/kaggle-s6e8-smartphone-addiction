# ───────────────────────────── 17. Feature importance ─────────────────────────────
md("## 17. Feature importance — and how it misleads")

code(r"""
# Guard against a degenerate model if early stopping fired very early on some fold.
final_n = max(50, int(np.mean(best_iters)))
final_lgb = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": final_n})
print(f"Refitting on all training data with n_estimators={final_n}")
final_lgb.fit(X_gbm, y)

imp = pd.DataFrame({
    "feature": X_gbm.columns,
    "gain": final_lgb.booster_.feature_importance(importance_type="gain"),
    "split": final_lgb.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)

display(imp.head(20))

top = imp.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, max(4, 0.34 * len(top))))
ax.barh(top["feature"], top["gain"], color="#55A868")
ax.set_xlabel("importance (gain)")
ax.set_title("LightGBM feature importance — top 20 by gain")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

zero_gain = imp[imp["gain"] == 0]["feature"].tolist()
print(f"Features the model never used: {zero_gain if zero_gain else 'none'}")
""")

md(r"""
**What this cell does and why it matters**

**The two importance types answer different questions, and the difference is instructive:**

- **`split`** counts how many times a feature was used to split. A feature can score highly
  here by being used constantly for small refinements.
- **`gain`** sums how much each split actually improved the loss. **This is the one to trust**
  — it measures contribution rather than frequency.

When a feature has high `split` but low `gain`, the model is fiddling with it repeatedly
without getting much value. That is often a high-cardinality feature offering many possible
split points, and it is frequently a sign of overfitting.

**Three ways this plot will mislead you if you let it:**

1. **Correlated features split the credit.** From Section 9: if two features are 0.95
   correlated, the model uses one arbitrarily and the other appears unimportant. Dropping the
   "unimportant" one may cost you nothing — or the model may simply switch to it. Importance
   does not tell you which.
2. **High cardinality inflates importance.** A feature with many distinct values offers more
   candidate split points and more chances to fit noise. It can rank highly for reasons that
   have nothing to do with genuine predictive value.
3. **This is importance *to this model*, not importance *in the world*.** It says how this
   particular fitted booster used the columns. It is **not** a causal claim. Nothing here says
   screen time *causes* addiction — the arrow could easily point the other way, or a third
   factor could drive both. Keep prediction and causation firmly separate.

**A more trustworthy alternative: permutation importance.** Shuffle one column, re-score, and
measure how far AUC falls. That directly answers "how much does the model's performance depend
on this feature?" It is slower but much harder to fool. `sklearn.inspection.permutation_importance`
implements it, and it is worth running on your final model.

**Zero-gain features** were never used at all. They are safe to drop — you will get identical
predictions and a faster pipeline. Do verify this with a CV run rather than assuming it.
""")
