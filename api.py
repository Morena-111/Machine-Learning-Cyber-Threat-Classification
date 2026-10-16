"""FastAPI inference service for the trained cyber-threat classifier."""
import pickle
from pathlib import Path
from typing import Dict

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import AttentionLayer, SparseAUC, SparseMacroF1, SparseMacroRate

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Cyber Threat Classifier", version="1.0.0")
model = None
label_encoder = None
tokenizer = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, description="Threat intelligence text to classify")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


def load_artifacts():
    global model, label_encoder, tokenizer
    if model is None:
        try:
            custom = {"AttentionLayer": AttentionLayer, "SparseMacroF1": SparseMacroF1, "SparseMacroRate": SparseMacroRate, "SparseAUC": SparseAUC}
            model = tf.keras.models.load_model(ROOT / "models" / "best_model.h5", custom_objects=custom)
            with open(ROOT / "models" / "label_encoder.pkl", "rb") as handle:
                label_encoder = pickle.load(handle)
            with open(ROOT / "models" / "tokenizer.pkl", "rb") as handle:
                tokenizer = pickle.load(handle)
        except FileNotFoundError as error:
            raise HTTPException(503, "Model artifacts are unavailable. Run `python -m src.train` first.") from error


@app.get("/")
def root():
    return {"service": "Cyber Threat Classifier", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ready" if (ROOT / "models" / "best_model.h5").exists() else "model_not_trained"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    load_artifacts()
    probs = model.predict([request.text], verbose=0)[0]
    index = int(probs.argmax())
    return PredictionResponse(predicted_class=str(label_encoder.classes_[index]), confidence=float(probs[index]), all_probabilities={str(label): float(prob) for label, prob in zip(label_encoder.classes_, probs)})
