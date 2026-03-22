# training/

ML pipeline for the Titanic survival predictor. Trains three candidate models, selects the best by cross-validated F1, logs all runs to MLflow, and saves the winner to the MLflow model registry and as a DVC-tracked artifact.

---

## Structure

```
training/
├── train.py      # Full training pipeline — feature engineering, model selection, MLflow logging
└── validate.py   # Post-training validation — asserts CV F1 > 0.7 before CI proceeds
```

---

## Pipeline

1. **Load data** — Titanic dataset downloaded from `raw.githubusercontent.com` during CI, or provided locally
2. **Feature engineering** — selects `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`; one-hot encodes `Embarked`; median-imputes `Age`
3. **Train candidates** — LogisticRegression, RandomForestClassifier, XGBClassifier; each run logged to MLflow
4. **Select best model** — ranked by `cv_f1_mean` (5-fold cross-validation)
5. **Register model** — winner logged to MLflow model registry as `titanic-survival-model` with `production` alias
6. **Save artifact** — winner saved to `models/best_model.joblib` and tracked by DVC

---

## Models Trained

| Model | Key Parameters |
|-------|---------------|
| LogisticRegression | `max_iter=1000` |
| RandomForestClassifier | `n_estimators=100`, `random_state=42` |
| XGBClassifier | `n_estimators=100`, `random_state=42` |

XGBoost currently wins with CV F1 ≈ 0.754.

---

## MLflow Tracking

- **Tracking server:** DagsHub-hosted MLflow instance
- **Experiment name:** `titanic-survival`
- **Logged per run:** parameters, CV F1 mean/std, test F1, test accuracy, model artifact
- **Authentication:** `DAGSHUB_USER_TOKEN` environment variable; `dagshub.init()` called at the top of `train()`

View experiment runs at: https://dagshub.com/PrithviAbhishetty/ml-titanic-predictor

---

## DVC Model Tracking

`best_model.joblib` is not committed to git. After training, the file is tracked by DVC and pushed to DagsHub:

```bash
dvc push  # uploads best_model.joblib to DagsHub remote
```

The `models/best_model.joblib.dvc` pointer file is committed to git, recording which model artifact corresponds to the current code state.

---

## Running Locally

```bash
DAGSHUB_USER_TOKEN=<your_token> uv run python training/train.py
uv run python training/validate.py  # optional — CI runs this automatically
dvc push
```

The Titanic dataset must be present locally. Download it from:
```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```