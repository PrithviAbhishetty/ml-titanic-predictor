import pandas as pd
import os
import joblib
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.data.pandas_dataset import from_pandas as mlflow_from_pandas
from mlflow.tracking import MlflowClient
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from xgboost import XGBClassifier
from git import Repo

def preprocess(df):
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)
    return df

def get_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name('titanic-survival')
    if experiment is None:
        raise ValueError("Experiment 'titanic-survival' not found")
    
    # Search all runs in the experiment, ordered by cv_f1_mean descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_f1_mean DESC"],
        max_results=1
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id

    print(f"Best run: {best_run.data.tags['mlflow.runName']}")
    print(f"CV F1 Mean: {best_run.data.metrics['cv_f1_mean']:.3f}")

    # Load and save the best model
    model = mlflow_sklearn.load_model(f"runs:/{best_run_id}/model")
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/best_model.joblib')

    return best_run_id

def register_best_model(best_run_id: str):
    client = MlflowClient()

    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "titanic-survival-model")
    new_version = registered_model.version

    new_run = client.get_run(best_run_id)
    new_f1 = new_run.data.metrics.get('cv_f1_mean', 0)

    try:
        production_version = client.get_model_version_by_alias(
            name="titanic-survival-model",
            alias="production"
        )
        prod_run_id = production_version.run_id
        if prod_run_id is None:
            raise ValueError("Production model has no associated run ID")
        prod_run = client.get_run(prod_run_id)
        prod_f1 = prod_run.data.metrics.get('cv_f1_mean', 0)
        delta = new_f1 - prod_f1
        print(f"Previous production F1: {prod_f1:.3f}")
        print(f"New model F1: {new_f1:.3f}")
        print(f"Improvement: {delta:+.3f}")
    except Exception:
        print(f"First registration - no previous production model")

    client.set_registered_model_alias(
        name="titanic-survival-model",
        alias="production",
        version=new_version
    )
    print(f"Model version {new_version} promoted to production")

    return new_version

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, '../data/titanic.csv'))
    df = preprocess(df)

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'logistic_regression': (
            LogisticRegression(max_iter=1000),
            {'max_iter': 1000}
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=100, random_state=42),
            {'n_estimators': 100, 'random_state': 42}
        ),
        'xgboost': (
            XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'logloss'}
        )
    }

    # Create MLflow dataset object
    titanic_dataset = mlflow_from_pandas(
        df, 
        source='../data/titanic.csv',
        name='titanic',
        targets='Survived'
    )

    # Get Git repo URL for tagging
    repo = Repo(search_parent_directories=True)
    repo_url = repo.remotes.origin.url

    # Create MLflow experiment
    mlflow.set_experiment('titanic-survival')

    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            train_proba = model.predict_proba(X_train)[:, 1]
            predictions = model.predict(X_test)
            predictions_proba = model.predict_proba(X_test)[:, 1]

            # Holdout metrics
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_f1 = f1_score(y_train, train_predictions)
            train_roc_auc = roc_auc_score(y_train, train_proba)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, predictions_proba)
            logloss = log_loss(y_test, predictions_proba)
            cm = confusion_matrix(y_test, predictions)

            # Cross-validation metrics
            cv_f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            cv_accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

            # Log model name and hyperparameters
            mlflow.log_param('model', model_name)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log dataset info
            mlflow.log_input(titanic_dataset, context='training')
            mlflow.log_param('dataset', 'titanic')
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            mlflow.log_param('test_split', 0.2)
            mlflow.log_param('random_state', 42)
            mlflow.log_param('n_features', X_train.shape[1])

            # Log tags
            mlflow.set_tag('task', 'binary_classification')
            mlflow.set_tag('dataset', 'titanic')
            mlflow.set_tag('git.repo_url', repo_url)

            # Log train vs test comparison
            mlflow.log_metric('train_accuracy', float(train_accuracy))
            mlflow.log_metric('test_accuracy', float(accuracy))
            mlflow.log_metric('train_f1', float(train_f1))
            mlflow.log_metric('test_f1', float(f1))
            mlflow.log_metric('train_roc_auc', float(train_roc_auc))
            mlflow.log_metric('test_roc_auc', float(roc_auc))
            mlflow.log_metric('overfit_gap_f1', float(train_f1 - f1))
            mlflow.log_metric('overfit_gap_roc_auc', float(train_roc_auc - roc_auc))
    
            # Log holdout metrics
            mlflow.log_metric('accuracy', float(accuracy))
            mlflow.log_metric('f1_score', float(f1))
            mlflow.log_metric('roc_auc', float(roc_auc))
            mlflow.log_metric('log_loss', float(logloss))

            # Log confusion matrix
            mlflow.log_metric('true_negatives', cm[0][0])
            mlflow.log_metric('false_positives', cm[0][1])
            mlflow.log_metric('false_negatives', cm[1][0])
            mlflow.log_metric('true_positives', cm[1][1])

            # Log cross-validation metrics
            for fold, (f1,acc) in enumerate(zip(cv_f1_scores, cv_accuracy_scores)):
                mlflow.log_metric('cv_f1', f1, step=fold)
                mlflow.log_metric('cv_accuracy', acc, step=fold)
            mlflow.log_metric('cv_f1_mean', cv_f1_scores.mean())
            mlflow.log_metric('cv_f1_std', cv_f1_scores.std())
            mlflow.log_metric('cv_accuracy_mean', cv_accuracy_scores.mean())
            mlflow.log_metric('cv_accuracy_std', cv_accuracy_scores.std())

            mlflow_sklearn.log_model(model, name='model')

            print(f'{model_name}:')
            print(f'  Holdout  — Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}, Log Loss: {logloss:.3f}')
            print(f'  CV       — F1 Mean: {cv_f1_scores.mean():.3f}, F1 Std: {cv_f1_scores.std():.3f}')
            print(f'  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}')

    # Save best model with joblib for API use
    best_run_id = get_best_model()
    model_version = register_best_model(best_run_id)
    print(f'Best model saved from run: {best_run_id}')
    print(f'Registered as version: {model_version} in Production')

if __name__ == '__main__':
    train()