
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path("artifacts/best_model.joblib")

app = FastAPI(
    title="Student MLOps Classification API",
    description=(
        "Educational machine-learning demo only. "
        "Not for medical diagnosis or clinical decisions."
    ),
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run python -m src.train first.",
        )

    expected = len(load_breast_cancer().feature_names)
    if len(request.features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features.",
        )

    model = joblib.load(MODEL_PATH)
    values = np.array(request.features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(values)[0])

    # scikit-learn target 0 = malignant, target 1 = benign.
    label = "malignant" if prediction == 0 else "benign"

    return {
        "prediction_code": prediction,
        "predicted_label": label,
        "notice": (
            "Educational dataset output only; "
            "not a medical diagnosis."
        ),
    }
