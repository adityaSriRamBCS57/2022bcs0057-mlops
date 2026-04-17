from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import json
import numpy as np
import os

STUDENT_NAME = "Aditya Sri Ram"
ROLL_NO = "2022BCS0057"

app = FastAPI(title="Iris MLOps API", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/best_scaler.pkl")
META_PATH = os.environ.get("META_PATH", "models/feature_meta.json")

model = None
scaler = None
feature_meta = None

@app.on_event("startup")
def load_model():
    global model, scaler, feature_meta
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        feature_meta = json.load(f)
    print("Model loaded successfully.")

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "model": feature_meta["model_type"] if feature_meta else "loading...",
        "features": feature_meta["features"] if feature_meta else []
    }

class PredictRequest(BaseModel):
    features: List[float]

IRIS_CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0].tolist()
    return {
        "prediction": int(pred),
        "class_name": IRIS_CLASSES[int(pred)],
        "probabilities": proba,
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }
