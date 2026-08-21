# ───────────────────────────── 19. XGBoost ─────────────────────────────
md("## 19. Model 4 — XGBoost")

code(r"""
import xgboost as xgb

# XGBoost handles pandas 'category' columns natively via enable_categorical.
xgb_params = dict(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.03,
    max_depth=6,             # XGBoost's capacity dial (it grows level-wise)
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=3000,
    enable_categorical=True,
    tree_method="hist",
    early_stopping_rounds=200,
    random_state=SEED,
    n_jobs=-1,
)

oof_xgb = np.zeros(len(y))
test_xgb = np.zeros(len(X_test_gbm))
xgb_iters = []

print("XGBoost")
for fold, (tr, va) in enumerate(cv.split(X_gbm, y)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X_gbm.iloc[tr], y[tr],
          eval_set=[(X_gbm.iloc[va], y[va])], verbose=False)
    oof_xgb[va] = m.predict_proba(X_gbm.iloc[va])[:, 1]
    test_xgb += m.predict_proba(X_test_gbm)[:, 1] / N_SPLITS
    xgb_iters.append(m.best_iteration)
    print(f"  fold {fold}  AUC {roc_auc_score(y[va], oof_xgb[va]):.5f}   "
          f"best_iter {m.best_iteration}")

xgb_mean, xgb_std = fold_auc_mean(oof_xgb, y, cv, X_gbm)
print(f"  {'-' * 42}")
print(f"  fold-mean AUC  {xgb_mean:.5f}  (+/- {xgb_std:.5f})")
display(record("XGBoost", xgb_mean, xgb_std, "level-wise trees, native categoricals"))
""")

md(r"""
**What this cell does and why it matters**

XGBoost is the other dominant gradient boosting library. The algorithm is the same idea as
LightGBM — sequential trees fitted to residual errors — but one design choice differs, and it is
the difference worth understanding.

**Level-wise vs leaf-wise growth.** XGBoost grows trees **level-wise**: it splits every node at
depth *d* before moving to depth *d+1*, producing balanced trees. LightGBM grows **leaf-wise**:
it repeatedly splits whichever leaf offers the largest gain, producing deep, lopsided trees.

The practical consequences:

| | XGBoost | LightGBM |
|---|---|---|
| Capacity dial | `max_depth` | `num_leaves` |
| Trees | Balanced | Unbalanced, deeper |
| Speed | Slower | Faster |
| Overfitting risk | Lower by default | Higher — needs `num_leaves` care |

**`max_depth=6` means up to 2⁶ = 64 leaves**, which is roughly comparable to LightGBM at
`num_leaves=64`. That is the mental conversion between the two libraries, and it is the thing
most people get wrong when porting parameters across.

**What to expect.** On this data, competitors measured XGBoost at 0.9595 and LightGBM at 0.9593
— a difference of 0.0002, comfortably inside fold noise. **Treat them as equally strong.** If
your two scores here differ by more than about `2 × cv_std`, that is worth investigating; it
probably means one of them is misconfigured rather than genuinely better.

The interesting question is not which of these wins. It is whether having both helps at all —
Section 22 answers that, and the answer is instructive.
""")

# ───────────────────────────── 20. CatBoost ─────────────────────────────
md("## 20. Model 5 — CatBoost")

code(r"""
from catboost import CatBoostClassifier, Pool

# CatBoost wants categorical columns as strings with no NaN.
X_cat = X.copy()
X_test_cat = X_test.copy()
for c in categorical_cols:
    X_cat[c] = X_cat[c].astype(object).fillna("__missing__").astype(str)
    X_test_cat[c] = X_test_cat[c].astype(object).fillna("__missing__").astype(str)

cat_idx = [X_cat.columns.get_loc(c) for c in categorical_cols]

cb_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    iterations=3000,
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,
    verbose=False,
    allow_writing_files=False,
)

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_test_cat))

print("CatBoost")
for fold, (tr, va) in enumerate(cv.split(X_cat, y)):
    m = CatBoostClassifier(**cb_params)
    m.fit(Pool(X_cat.iloc[tr], y[tr], cat_features=cat_idx),
          eval_set=Pool(X_cat.iloc[va], y[va], cat_features=cat_idx),
          use_best_model=True)
    oof_cb[va] = m.predict_proba(X_cat.iloc[va])[:, 1]
    test_cb += m.predict_proba(X_test_cat)[:, 1] / N_SPLITS
    print(f"  fold {fold}  AUC {roc_auc_score(y[va], oof_cb[va]):.5f}   "
          f"best_iter {m.get_best_iteration()}")

cb_mean, cb_std = fold_auc_mean(oof_cb, y, cv, X_cat)
print(f"  {'-' * 42}")
print(f"  fold-mean AUC  {cb_mean:.5f}  (+/- {cb_std:.5f})")
display(record("CatBoost", cb_mean, cb_std, "ordered boosting, symmetric trees"))
""")

md(r"""
**What this cell does and why it matters**

CatBoost is the third major GBDT, and it differs from the other two in two substantive ways.

**1. Ordered boosting.** Standard boosting computes each tree's target using a model that was
trained on the same rows it is now scoring. This creates a subtle bias CatBoost's authors call
*prediction shift* — a mild form of the leakage problem from Section 13, living inside the
boosting algorithm itself. CatBoost avoids it by computing residuals for each row using a model
trained only on rows that came *before* it in a random permutation. Slower, but less biased,
and it helps most on smaller datasets.

**2. Ordered target statistics for categoricals.** CatBoost does target encoding internally —
and does it with the same permutation trick, so the encoding for a row never uses that row's own
label. This is the leak-safe version of the technique Section 8 warned about, implemented for
you. It is why CatBoost is often the strongest model on categorical-heavy data with no manual
encoding work at all.

**3. Symmetric trees.** Every node at a given depth uses the *same* split condition. This is far
more constrained than either competitor — it acts as strong regularisation and makes prediction
very fast, but it can underfit when the true structure needs asymmetric splits.

**What to expect here.** Competitors measured CatBoost at 0.9561, roughly 0.003 *behind* the
other two GBDTs. With only three low-cardinality categorical columns, CatBoost's main advantage
barely applies — and the symmetric-tree constraint costs it.

**But do not drop it on that basis**, and this is the point of including it: in a learned stack
on this data, CatBoost earned a positive weight of **0.456** despite being the weakest of the
three. Being *different* has value that a raw score comparison does not show. Section 22
measures exactly that.
""")

# ───────────────────────────── 21. Neural network ─────────────────────────────
md("## 21. Model 6 — a neural network")

code(r"""
from sklearn.neural_network import MLPClassifier

# The NN reuses the Section 14 preprocessor: one-hot + scaled. Scaling is not
# optional here the way it is for trees — an unscaled input will not train.
nn_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,              # L2 penalty
        batch_size=1024,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
        random_state=SEED,
    )),
])

print("Neural network (MLP)")
oof_nn, test_nn, nn_mean, nn_std, _ = cross_validate_model(
    nn_pipeline, X, y, X_test, cv, "mlp")

display(record("Neural network (MLP)", nn_mean, nn_std, "128-64 MLP on scaled one-hot"))
""")

md(r"""
**What this cell does and why it matters**

**This is the model that matters most for your ensemble, and it is not because it scores best.**

A neural network approximates a smooth, continuous function of *all* inputs simultaneously. A
gradient boosted tree approximates a piecewise-constant function built from axis-aligned splits.
These are fundamentally different function classes, so they fail in fundamentally different
places — and that is precisely what makes an ensemble work.

The measured evidence on this competition:

| Pair | Rank correlation | Weight earned in blend |
|---|---|---|
| LightGBM ↔ XGBoost | 0.997 | ≈ 0.000 |
| GBDT ↔ neural net | 0.974 | **0.220** |

**A second GBDT earns nothing. A neural network earns real weight.** Diversity is what pays,
and diversity means a different *kind* of model — not another variant of one you already have.

**Why scaling is mandatory here.** Trees are invariant to monotonic rescaling and genuinely do
not care. A neural network absolutely does: it uses gradient descent, and features on wildly
different scales produce wildly different gradient magnitudes, so training either crawls or
diverges. This is why the NN reuses the `preprocessor` from Section 14 while the GBDTs used raw
categorical dtypes.

**Expect a lower score than the GBDTs, and do not read that as failure.** On tabular data of
this size an MLP typically lands somewhat behind well-tuned boosting. It earns its place through
decorrelation, not through raw accuracy. Section 22 shows the difference between those two forms
of value.

**If you want to push this further**, `MLPClassifier` is a deliberately simple choice — it keeps
the notebook dependency-light and readable. The competitive options for tabular neural networks
are **RealMLP** and **TabM**, and in the most recent comparable Playground episode those actually
*beat* CatBoost and XGBoost on CV, leaderboard, and private score. Categorical **embeddings**
(learned dense vectors per category, instead of one-hot) are the other standard upgrade.
""")

# ───────────────────────────── 22. Diversity ─────────────────────────────
md("## 22. Measuring diversity — which models actually differ?")

code(r"""
from scipy.stats import rankdata

oof_matrix = pd.DataFrame({
    "logreg": oof_lr,
    "lightgbm": oof_lgb,
    "xgboost": oof_xgb,
    "catboost": oof_cb,
    "neural_net": oof_nn,
})
test_matrix = pd.DataFrame({
    "logreg": test_lr,
    "lightgbm": test_lgb,
    "xgboost": test_xgb,
    "catboost": test_cb,
    "neural_net": test_nn,
})

solo = pd.Series({c: fold_auc_mean(oof_matrix[c].values, y, cv, X)[0]
                  for c in oof_matrix.columns}).sort_values(ascending=False)
print("Individual fold-mean AUC")
print("-" * 34)
for k, v in solo.items():
    print(f"  {k:<14} {v:.5f}")

# Correlate RANKS, since AUC cares only about ordering.
rank_df = oof_matrix.apply(lambda c: rankdata(c) / len(c))
cmat = rank_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(cmat, cmap="RdYlGn_r", vmin=0.85, vmax=1.0)
ax.set_xticks(range(len(cmat))); ax.set_xticklabels(cmat.columns, rotation=45, ha="right")
ax.set_yticks(range(len(cmat))); ax.set_yticklabels(cmat.columns)
for i in range(len(cmat)):
    for j in range(len(cmat)):
        ax.text(j, i, f"{cmat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
ax.set_title("Rank correlation between model predictions\n(green = diverse, red = redundant)",
             fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

pairs = (cmat.where(np.triu(np.ones(cmat.shape), k=1).astype(bool))
         .stack().reset_index())
pairs.columns = ["model_a", "model_b", "rank_corr"]
print("\nMost redundant pairs (adding the second buys you least):")
display(pairs.sort_values("rank_corr", ascending=False).head(4))
print("Most diverse pairs (best ensemble candidates):")
display(pairs.sort_values("rank_corr").head(4))
""")

md(r"""
**What this cell does and why it matters**

This is the analysis almost everyone skips, and it is what separates a deliberate ensemble from
a pile of models.

**Two numbers decide whether a model belongs in your ensemble, and you need both:**

1. **Individual strength** — is it good enough to contribute? A model far behind the leaders
   drags the blend down regardless of how different it is.
2. **Correlation with what you already have** — is it different enough to add anything?

**Either one alone will mislead you.** The measured example from this competition: ExtraTrees had
a rank correlation of 0.967 (genuinely diverse, better decorrelated than XGBoost) but sat 0.0064
behind the best model. It earned a blend weight of **exactly zero**. Diverse but too weak.
Meanwhile XGBoost is plenty strong and also earns nothing, because at 0.997 correlation it is
LightGBM in a different coat.

**The entry condition is two-dimensional: different direction *and* comparable strength.**

**Read the heatmap for blocks.** You should see the three GBDTs forming a tight red cluster —
they are variations on one algorithm. Logistic regression and the neural network should sit
further out in green. Those green cells are where ensemble gains live.

**Why correlate ranks rather than raw probabilities?** Same reasoning as Section 18. Our metric
depends only on ordering, so two models that produce identical rankings on different scales are
identical *for our purposes*. Correlating raw probabilities would report them as different when
they are not.

**The practical rule this gives you:** before adding a sixth model, ask what it is decorrelated
*from*. If the answer is "nothing I have", it will not help — no matter how good its solo score
looks. Adding a genuinely new *family* beats adding another variant, every time.
""")
