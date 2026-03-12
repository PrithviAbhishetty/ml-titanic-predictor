import joblib
import pandas as pd
from api.schemas import PassengerInput

class ModelService:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def preprocess(self, passenger: PassengerInput) -> pd.DataFrame:
        data = {
            'Pclass': passenger.pclass,
            'Sex': 1 if passenger.sex == 'female' else 0,
            'Age': passenger.age,
            'SibSp': passenger.sibsp,
            'Parch': passenger.parch,
            'Fare': passenger.fare,
            'Embarked_Q': 1 if passenger.embarked == 'Q' else 0,
            'Embarked_S': 1 if passenger.embarked == 'S' else 0,
        }
        return pd.DataFrame([data])
    
    def predict(self, passenger: PassengerInput) -> tuple[bool, float]:
        df = self.preprocess(passenger)
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        return bool(prediction), round(float(probability), 4)