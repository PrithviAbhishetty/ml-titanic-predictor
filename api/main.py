from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import PassengerInput, PredictionOutput
from model import ModelService

app = FastAPI(
    title="Titanic Survival Predictor",
    description="Predicts whether a Titanic passenger would have survived based on their attributes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService(model_path="../models/best_model.joblib")

@app.get("/health")
def health():
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