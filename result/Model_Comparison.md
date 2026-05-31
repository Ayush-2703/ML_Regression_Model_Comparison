# Model Comparison Results

This document summarises the performance of all three models trained on the California Housing Dataset and explains why Decision Tree Regressor was selected as the final model.

---

## Performance Metrics

All models were evaluated on the same 20% test split (4,129 samples), using two metrics:

- **RMSE (Root Mean Squared Error)** — measures prediction error in the same unit as the target variable. Lower is better.
- **R² Score** — measures the proportion of variance in house prices explained by the model. Higher is better (max is 1.0).

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge Regression (α=1.0) | 0.7455 | 0.5758 |
| **Decision Tree Regressor** (depth=5) | **0.7242** | **0.5997** |

---

## What the Numbers Mean

**Linear Regression** is the baseline. An RMSE of 0.7456 on this dataset means predictions are off by roughly $74,560 on average (since house prices are scaled in units of $100,000). An R² of 0.58 means the model explains about 58% of the variation in house prices — not bad for a straight line through 8 features.

**Ridge Regression** is almost identical. The difference (0.7456 vs 0.7455) is negligible and wouldn't influence any real decision. This tells us the features aren't highly collinear and Linear Regression isn't suffering from the kind of coefficient inflation that Ridge is designed to fix. Ridge adds value in other datasets — just not this one.

**Decision Tree Regressor** improves meaningfully on both metrics. An R² of 0.5997 vs 0.5758 means it explains about 2.4 percentage points more of the variance. The RMSE drops from 0.7456 to 0.7242 — not dramatic, but real. Limiting depth to 5 keeps it from overfitting.

---

## Why Decision Tree Won

The main reason is non-linearity. Linear models assume each feature has a straight-line relationship with the target. Housing prices don't work that way — a median income of $8 (in tens of thousands) in a coastal area predicts very differently from the same income inland. The Decision Tree splits on feature thresholds and can capture these kinds of interactions without explicitly engineering them.

The depth limit of 5 matters. An unlimited tree would score near-perfectly on training data and poorly on the test set — that's overfitting. At depth=5 the tree makes at most 32 terminal predictions (2^5), which is enough granularity to capture the main patterns without memorising noise.

---

## What the Actual vs Predicted Plot Shows

The scatter plot from Step 8 shows test set predictions against actual values. A perfect model would have all points sitting exactly on the diagonal line. In practice:

- The cluster around prices of 1–2 (i.e. $100K–$200K) is dense and reasonably tight — the model handles the typical range well
- At higher prices (3.0–5.0+) the predictions spread out and the model tends to underestimate
- This is expected: expensive homes have more unique characteristics that aren't captured in the 8 input features

---

## Final Model Choice

**Decision Tree Regressor with max_depth=5.**

It outperforms both linear approaches on RMSE and R², it handles the non-linear relationships in this dataset better, and the depth constraint keeps it from overfitting. For a production deployment, this would be the starting point — with further tuning (trying depths 3–10, adding cross-validation) before committing.

---

## Notes on Evaluation

One thing worth being honest about: a single 80/20 split gives us one estimate of performance. If we re-ran with a different `random_state`, the numbers would shift slightly. A more robust evaluation would use 5-fold or 10-fold cross-validation to average performance across multiple splits. That's a natural next improvement for this project.
