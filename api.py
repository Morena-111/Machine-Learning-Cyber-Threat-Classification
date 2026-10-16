"""
FastAPI inference service for the trained cyber-threat classifier.

Loads the model using rebuild + load_weights to bypass a Keras 3
TextVectorization deserialization bug.

Run with:
    uvicorn api:app --reload --port 8000
"""

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import build_model


ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Cyber Threat Classifier", version="1.0.0")

model = None
label_encoder = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, description="Threat intelligence text")


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


def load_artifacts():
    """Lazily load model artifacts (rebuild + load_weights pattern)."""
    global model, label_encoder
    if model is not None:
        return

    try:
        with open(ROOT / "models" / "label_encoder.pkl", "rb") as handle:
            label_encoder = pickle.load(handle)
        with open(ROOT / "models" / "tokenizer.pkl", "rb") as handle:
            vocabulary = pickle.load(handle)

        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=10000,
            output_mode="int",
            output_sequence_length=50,
            standardize="lower_and_strip_punctuation",
        )
        vectorizer.set_vocabulary(vocabulary)

        model = build_model(vectorizer, num_classes=len(label_encoder.classes_))
        model.load_weights(ROOT / "models" / "best_model.h5")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    except FileNotFoundError as error:
        raise HTTPException(
            503,
            "Model artifacts unavailable. Run `python -m src.train` first.",
        ) from error


@app.get("/")
def root():
    return {"service": "Cyber Threat Classifier", "docs": "/docs"}


@app.get("/health")
def health():
    exists = (ROOT / "models" / "best_model.h5").exists()
    return {"status": "ready" if exists else "model_not_trained"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    load_artifacts()
    probs = model.predict([request.text], verbose=0)[0]
    index = int(probs.argmax())
    return PredictionResponse(
        predicted_class=str(label_encoder.classes_[index]),
        confidence=float(probs[index]),
        all_probabilities={
            str(label): float(prob)
            for label, prob in zip(label_encoder.classes_, probs)
        },
    )