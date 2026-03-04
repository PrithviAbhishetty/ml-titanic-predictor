import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
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

def train():
    df = pd.read_csv('../data/titanic.csv')
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
    titanic_dataset = mlflow.data.from_pandas(
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

    best_model = None
    best_f1 = 0
    best_run_id = None

    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_proba = model.predict_proba(X_test)[:, 1]

            # Holdout metrics
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

            # Log holdout metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('roc_auc', roc_auc)
            mlflow.log_metric('log_loss', logloss)

            # Log confusion matrix
            mlflow.log_metric('true_negatives', cm[0][0])
            mlflow.log_metric('false_positives', cm[0][1])
            mlflow.log_metric('false_negatives', cm[1][0])
            mlflow.log_metric('true_positives', cm[1][1])

            # Log cross-validation metrics
            mlflow.log_metric('cv_f1_mean', cv_f1_scores.mean())
            mlflow.log_metric('cv_f1_std', cv_f1_scores.std())
            mlflow.log_metric('cv_accuracy_mean', cv_accuracy_scores.mean())
            mlflow.log_metric('cv_accuracy_std', cv_accuracy_scores.std())

            mlflow.sklearn.log_model(model, name='model')

            print(f'{model_name}:')
            print(f'  Holdout  — Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}, Log Loss: {logloss:.3f}')
            print(f'  CV       — F1 Mean: {cv_f1_scores.mean():.3f}, F1 Std: {cv_f1_scores.std():.3f}')
            print(f'  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}')

            if cv_f1_scores.mean() > best_f1:
                best_f1 = cv_f1_scores.mean()
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

    # Save best model with joblib for API use
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/best_model.joblib')
    print(f'\nBest model run ID: {best_run_id} with CV F1 Mean: {best_f1:.3f}')
    print(f'Best model saved to ../models/best_model.joblib')

if __name__ == '__main__':
    train()