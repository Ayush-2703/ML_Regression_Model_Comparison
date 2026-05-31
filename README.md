# 🏠 House Price Prediction — Regression Model Comparison

> Comparing Linear Regression, Ridge Regression, and Decision Tree on the California Housing Dataset to find out which one actually earns its keep.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Project Is About

When I started this, the goal was simple: don't just train one model and call it a day. Real ML work means preprocessing your data properly, trying a few different algorithms, measuring them honestly, and then making a justified call on which one to use.

This project walks through exactly that — using California housing data to predict median house prices, while comparing three regression approaches side by side.

It was built as **Task 2 of the AI & ML Internship at Maincrafts Technology**, and focuses on the parts that actually matter in real projects: feature scaling, model diversity, and metrics-driven decision making.

---

## The Results (spoiler)

| Model | RMSE ↓ | R² Score ↑ |
|---|---|---|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge Regression | 0.7455 | 0.5758 |
| **Decision Tree Regressor** | **0.7242** | **0.5997** |

**Winner: Decision Tree Regressor (max_depth=5)**

The Decision Tree edged out both linear models — not by a massive margin, but consistently. The reason makes sense: housing prices have non-linear relationships (location interacts with income in ways a straight line can't capture), and the tree naturally picks those up. Ridge barely differed from plain Linear Regression because the features were already well-scaled, so regularization had little extra work to do.

See the full analysis in [`results/model_comparison.md`](results/model_comparison.md).

---

## Project Structure

```
house-price-prediction-ml/
│
├── notebooks/
│   ├── AI_ML_Task2_Model_Comparison.ipynb   ← main notebook, run this
│   └── README.md                            ← what's in the notebook
│
├── docs/
│   └── methodology.md                       ← written explanation of every decision
│
├── results/
│   └── model_comparison.md                  ← metrics, observations, model choice
│
├── .gitignore
├── LICENSE
├── README.md                                ← you are here
├── requirements.txt
└── CONTRIBUTING.md
```

---

## The ML Pipeline

Here's what happens inside the notebook, step by step:

1. **Load the dataset** — California Housing from scikit-learn (20,640 samples, 8 features)
2. **Explore it** — `df.head()`, `df.info()`, `df.describe()` to understand what we're working with
3. **Separate features and target** — X gets the input columns, y gets `HousePrice`
4. **Scale the features** — StandardScaler brings everything to the same range; skipping this step makes gradient-based models behave badly
5. **Train/test split** — 80% training, 20% testing, random_state=42 for reproducibility
6. **Train three models** — Linear Regression (baseline), Ridge Regression (regularized), Decision Tree (non-linear)
7. **Evaluate with RMSE and R²** — lower RMSE and higher R² both mean better
8. **Visualise predictions** — scatter plot of actual vs predicted values for the best model

---

## Why These Three Models?

**Linear Regression** is the honest baseline. If you can't beat a straight line, something is wrong with your more complex approach.

**Ridge Regression** is Linear Regression with L2 regularization added. It punishes large coefficients, which helps when features are correlated or when you're worried about overfitting. In this case, it performed almost identically to Linear — a useful data point in itself.

**Decision Tree Regressor** splits data based on feature thresholds, building a tree of if-else rules. It handles non-linearity without any feature engineering and doesn't care about feature scale (though we scaled anyway for consistency). Limiting depth to 5 keeps it from memorising the training data.

---

## Tech Stack

- **Python 3.8+**
- **pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **scikit-learn** — models, scaler, metrics, dataset
- **matplotlib** — visualisations
- **Jupyter Notebook** — interactive development environment

---

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the notebook

```bash
jupyter notebook notebooks/AI_ML_Task2_Model_Comparison.ipynb
```

Run all cells from top to bottom. The dataset loads automatically from scikit-learn — no downloads needed.

---

## Key Takeaways

A few things this project makes clear that are easy to miss when you're just starting out:

- **Feature scaling isn't optional for linear models.** Without it, a feature with values in the thousands (like population) will dominate one with values in the single digits (like average rooms), and your model will suffer for it.

- **Training accuracy isn't the metric that matters.** A Decision Tree with no depth limit will score perfectly on training data and terribly on anything new. Always evaluate on held-out test data.

- **Comparing models beats picking one by gut feeling.** Ridge and Linear Regression being nearly identical tells you something real about this dataset — the features aren't highly collinear and there's no major overfitting problem to regularize away.

- **R² of 0.60 is not a failure.** Housing prices are genuinely hard to predict from 8 features. Location matters, school districts matter, recent renovations matter — none of that is in this dataset. A model that explains 60% of the variance with clean, simple inputs is doing reasonable work.

---

## What Could Be Improved

This project is intentionally scoped to the internship task, but here's what would push it further:

- Try **Random Forest** or **Gradient Boosting** (XGBoost, LightGBM) — ensemble methods almost always beat a single tree
- Add **cross-validation** instead of a single train/test split for more reliable estimates
- Do proper **hyperparameter tuning** with GridSearchCV or Optuna
- Engineer new features — rooms per household, income-to-price ratio, etc.
- Build a simple **Streamlit app** to make predictions on user inputs

---

## Author

**AYUSH KUMAR SINGH**

AI & ML Intern — Maincrafts Technology

