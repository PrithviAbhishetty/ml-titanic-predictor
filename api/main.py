from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PassengerInput, PredictionOutput
from api.model import ModelService
import os

app = FastAPI(
    title="Titanic Survival Predictor",
    description="Predicts whether a Titanic passenger would have survived based on their attributes.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://ml-titanic-predictor.*\.vercel\.app|http://localhost:5173|https://ml-titanic-predictor\.onrender\.com",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/best_model.joblib')
model_service = ModelService(model_path=MODEL_PATH)

@app.get("/health")
def health():
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    try:
        survived, probability = model_service.predict(passenger)
        return PredictionOutput(
            survived=survived,
            survival_probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))