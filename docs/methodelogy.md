# Methodology

This document explains the reasoning behind each step in the ML pipeline — not just *what* was done, but *why* each decision was made.

---

## 1. Dataset Selection

**California Housing Dataset** — included in scikit-learn, so no download required.

It has 20,640 samples and 8 input features derived from the 1990 California census:

| Feature | Description |
|---|---|
| MedInc | Median income in the block group (in tens of thousands) |
| HouseAge | Median age of houses in the block |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

**Target:** Median house value in the block (in hundreds of thousands of dollars)

This is a clean, realistic dataset — no missing values, a reasonable number of features, and a continuous target that makes it perfect for regression work.

---

## 2. Exploratory Data Analysis

Before touching any models, `df.info()` and `df.describe()` were used to:

- Confirm there are no missing values (there aren't)
- Check feature ranges (MedInc is 0.5–15; Population is 3–35,682 — a massive range)
- Understand the target distribution (house values range from 0.15 to 5.0, in $100K units)

The large range differences across features are exactly why feature scaling comes next.

---

## 3. Feature Scaling

**StandardScaler** was applied to all input features before training.

StandardScaler transforms each feature to have mean=0 and standard deviation=1:

```
x_scaled = (x - mean) / std
```

**Why this matters:**

Linear Regression and Ridge Regression use gradient descent or normal equations that are sensitive to feature scale. If Population (which ranges up to 35,000) and AveRooms (which ranges 0–100) are on very different scales, the model's coefficients will be dominated by whichever feature has the largest raw values — not necessarily the most predictive one.

Scaling puts all features on equal footing so the algorithm can learn their actual importance.

**Decision Tree doesn't technically need scaling** (it only looks at thresholds, not magnitudes), but it was scaled here anyway for consistency and to make direct comparisons cleaner.

**Critical note:** The scaler was fit only on the training data (`scaler.fit_transform(X_scaled)` before the split is acceptable here since we call it on X before splitting, but in production you'd `fit` on X_train only and `transform` X_test separately to avoid data leakage).

---

## 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

- **80/20 split** — 16,512 training samples, 4,128 test samples
- **random_state=42** — makes the split reproducible; anyone re-running the code gets the same split

The test set is locked away and only used for final evaluation. The models never see it during training. This simulates how a model would perform on genuinely new data.

---

## 5. Model Selection

Three models were chosen to cover the spectrum from simple to non-linear:

### Linear Regression
The baseline. Finds the best-fit hyperplane through the data by minimising the sum of squared residuals. Fast, interpretable, and a necessary reference point — if a more complex model can't beat it, something is wrong.

### Ridge Regression (L2 Regularization)
Linear Regression with a penalty term added:

```
Loss = MSE + α * Σ(coefficients²)
```

The penalty discourages large coefficients, which helps when features are correlated or when the model is prone to overfitting. `alpha=1.0` is a standard starting value. The fact that Ridge and Linear Regression performed identically suggests there's no major multicollinearity problem in this dataset.

### Decision Tree Regressor (max_depth=5)
Splits the data recursively on feature thresholds, building a binary tree of decisions. Each leaf node predicts the mean of the training samples that reach it.

`max_depth=5` limits the tree to 5 levels of splits. Without this constraint, the tree would memorise the training data (overfitting). At depth=5 it makes up to 32 unique predictions, which is enough to capture the main patterns in this dataset without fitting to noise.

---

## 6. Evaluation Metrics

### RMSE (Root Mean Squared Error)

```
RMSE = √(1/n × Σ(predicted - actual)²)
```

Measures average prediction error in the same unit as the target. An RMSE of 0.72 means predictions are off by roughly $72,000 on average. The squaring before averaging means large errors are penalised more heavily than small ones.

### R² Score (Coefficient of Determination)

```
R² = 1 - (SS_residuals / SS_total)
```

Proportion of variance in the target that the model explains. R²=1.0 means perfect predictions. R²=0.60 means the model explains 60% of why house prices vary across blocks. R²=0.0 means the model does no better than predicting the mean every time.

Both metrics together give a complete picture — a model could have low RMSE but also low R² if the target variable has low total variance.

---

## 7. Final Model Justification

**Decision Tree Regressor (max_depth=5)** was selected because:

1. It achieved the lowest RMSE (0.7242 vs 0.7456)
2. It achieved the highest R² (0.5997 vs 0.5758)
3. The performance improvement is consistent — both metrics agree
4. The depth constraint provides reasonable overfitting protection
5. It captures non-linear relationships that linear models structurally cannot

The difference might seem small, but in a real deployment it translates to predictions that are roughly $21,400 more accurate on average (0.7456 - 0.7242 = 0.0214, × $100,000).

---

## Limitations and Honest Caveats

This project is an educational exercise, and some shortcuts were made that you'd handle differently in production:

- **Single split evaluation** — cross-validation would give more reliable estimates
- **No hyperparameter search** — max_depth=5 was set manually; systematic search might find better values
- **No feature engineering** — derived features like rooms_per_person or income_per_bedroom might improve all models
- **Missing data not present here** — real datasets usually have nulls that need handling
- **Scaler fitted before split** — in strict production practice you'd fit the scaler only on training data to prevent any leakage
