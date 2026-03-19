"""
FastAPI serving endpoint for Classification Engine.
POST features -> model prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Classification Engine API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
_model = None


class PredictionInput(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: float
    model: str = "LogisticRegression"


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "logreg_silicon.pkl")
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model": "LogisticRegression"}


@app.get("/info")
def info_endpoint():
    return {"project": "002_classification_engine", "description": "Classification Engine", "task": "classification"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        model = get_model()
        df = pd.DataFrame([input_data.features])
        pred = model.predict(df)[0]
        return PredictionResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
