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
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    }

    mlflow.set_experiment('titanic-survival')

    best_model = None
    best_f1 = 0
    best_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            mlflow.log_param('model', model_name)
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