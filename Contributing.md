# Contributing

Thanks for taking a look at this project. Contributions are welcome, whether that's fixing a typo, improving the analysis, or adding a new model.

---

## Getting the project running locally

```bash
git clone https://github.com/your-username/house-price-prediction-ml.git
cd house-price-prediction-ml
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/AI_ML_Task2_Model_Comparison.ipynb
```

---

## What's welcome

**Bug fixes** — if something errors on your machine or a result seems wrong, open an issue with what you're seeing.

**Notebook improvements** — better markdown cells, cleaner code, additional visualisations.

**New models** — want to add Random Forest or Gradient Boosting? Go for it. Just add the results to the comparison table in `results/model_comparison.md` too.

**Documentation** — if something in the docs was unclear, improving it is a perfectly valid contribution.

---

## How to contribute

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Commit with a clear message: `git commit -m "Add RandomForest to model comparison"`
5. Push and open a pull request

---

## Things to keep consistent

- Keep `random_state=42` for anything involving randomness, so results are reproducible
- If you add a model, evaluate it with the same RMSE and R² metrics on the same test split
- Run the full notebook from scratch before submitting to make sure it executes cleanly

---

## Code style

No strict linter configured. Just try to match the style of what's already there — clear variable names, comments where the logic isn't obvious, consistent spacing.

---

## Questions?

Open an issue. That's what they're there for.
