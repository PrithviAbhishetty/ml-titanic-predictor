from pydantic import BaseModel, Field
from typing import Literal

class PassengerInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: Literal['male', 'female'] = Field(..., description="Passenger sex")
    age: float = Field(..., gt=0, lt=120, description="Passenger age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare in pounds")
    embarked: Literal['S', 'C', 'Q'] = Field(..., description="Port of embarkation")

class PredictionOutput(BaseModel):
    survived: bool
    survival_probability: float
    