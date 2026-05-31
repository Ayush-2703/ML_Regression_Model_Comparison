# Notebooks

## AI_ML_Task2_Model_Comparison.ipynb

This is the main notebook for the project. Run it from top to bottom — all cells are designed to execute in order.

### What each section does

| Section | What it does |
|---|---|
| **Setup** | Installs and imports required libraries |
| **Step 1: Load Data** | Fetches the California Housing Dataset via scikit-learn, no download needed |
| **Step 2: Explore** | Runs `df.head()`, `df.info()`, `df.describe()` — gives you a feel for the data before modelling |
| **Step 3: Split X and y** | Separates input features from the target variable |
| **Step 4: Feature Scaling** | Applies StandardScaler — all features get scaled to mean=0, std=1 |
| **Step 5: Train/Test Split** | 80% training, 20% test, random_state=42 |
| **Step 6: Train Models** | Trains Linear Regression, Ridge Regression, and Decision Tree in a loop |
| **Step 7: Evaluate** | Computes RMSE and R² for each model, builds a results DataFrame |
| **Step 8: Visualise** | Scatter plot of actual vs predicted values for the best model |

### Expected outputs

When you run the notebook you should see:

- A `df.head()` table showing the first 5 rows of the dataset
- Summary statistics from `df.describe()`
- A results table comparing all three models (Decision Tree should win)
- A scatter plot with actual prices on the x-axis and predicted prices on the y-axis

### Running time

The whole notebook runs in under 30 seconds on a standard laptop. The California Housing Dataset isn't large enough to cause any slowdowns.

### Troubleshooting

**ModuleNotFoundError** — run `pip install -r requirements.txt` from the project root before launching Jupyter.

**Kernel not found** — make sure you activated the right virtual environment before running `jupyter notebook`.

**Different numbers than the README** — this shouldn't happen if you use the same `random_state=42`, but small scikit-learn version differences can occasionally shift metrics by a tiny amount.
