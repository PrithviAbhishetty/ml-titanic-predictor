import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

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

    # Create MLflow experiment
    mlflow.set_experiment('titanic-survival')

    best_model = None
    best_f1 = 0
    best_run_id = None

    for model_name, (model, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            repo_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"]
            ).decode("utf-8").strip()

            # Log model name and hyperparameters
            mlflow.log_param('model', model_name)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log dataset
            mlflow.log_input(titanic_dataset, context='training')

            # Log dataset info
            mlflow.log_param('dataset', 'titanic')
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            mlflow.log_param('test_split', 0.2)
            mlflow.log_param('random_state', 42)
            mlflow.log_param('n_features', X_train.shape[1])

            # Log tags
            mlflow.set_tag('task', 'binary_classification')
            mlflow.set_tag('dataset', 'titanic')
            mlflow.set_tag("git.repo_url", repo_url)

            # Log metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1_score', f1)

            mlflow.sklearn.log_model(model, artifact_path='model')

            print(f'{model_name} - Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}')

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_run_id = mlflow.active_run().info.run_id

    print(f'\nBest model run ID: {best_run_id} with F1 Score: {best_f1:.3f}')

if __name__ == '__main__':
    train()