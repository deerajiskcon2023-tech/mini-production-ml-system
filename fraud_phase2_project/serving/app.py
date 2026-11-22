"""FastAPI application for online financial transaction fraud scoring."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from .schemas import TransactionFeatures, PredictionResponse, HealthResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "fraud_model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "artifacts" / "model_metadata.json"))

app = FastAPI(
    title="Financial Transaction Fraud Detection API",
    description="Scores a transaction and returns a fraud probability and decision.",
    version="1.0.0",
)

model = None
metadata = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
except Exception as exc:  # health endpoint reports unavailable model
    load_error = str(exc)

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok" if model is not None and metadata is not None else "degraded",
        "model_loaded": model is not None and metadata is not None,
        "model_name": metadata.get("model_name", "unknown") if metadata else "unknown",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    try:
        expected_features = metadata["features"]
        values = [float(getattr(transaction, name)) for name in expected_features]
        if not np.isfinite(values).all():
            raise HTTPException(status_code=422, detail="All feature values must be finite numbers.")
        model_input = pd.DataFrame([values], columns=expected_features)
        probability = float(model.predict_proba(model_input)[0][1])
        threshold = float(metadata["threshold"])
        predicted_class = int(probability >= threshold)
        return PredictionResponse(
            fraud_probability=probability,
            predicted_class=predicted_class,
            decision="fraud" if predicted_class == 1 else "legitimate",
            threshold=threshold,
            model_name=metadata["model_name"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc
