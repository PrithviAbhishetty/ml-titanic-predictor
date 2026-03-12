import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from train import preprocess

F1_THRESHOLD = 0.7

def validate():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    df = pd.read_csv(os.path.join(BASE_DIR, '../data/titanic.csv'))
    df = preprocess(df)

    X = df.drop(columns=['Survived'])
    y = df['Survived']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(os.path.join(BASE_DIR, '../models/best_model.joblib'))
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)

    print(f'Model F1 score: {f1:.3f}')
    if f1 < F1_THRESHOLD:
        print(f'FAILED: F1 score {f1:.3f} is below threshold of {F1_THRESHOLD}')
        sys.exit(1)
    
    print(f'PASSED: F1 score {f1:.3f} meets threshold of {F1_THRESHOLD}')

if __name__ == '__main__':
    validate()