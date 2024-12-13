from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: List[float]

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    forecast: float

app = FastAPI()
model = joblib.load("model.joblib")

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    model = joblib.load("model.joblib")
    preds = model.predict([data.features]).flatten().tolist() # generamos la predicción
    prediction = ApiOutput(forecast=preds[0]) # estructuramos la salida del API.
    return prediction
