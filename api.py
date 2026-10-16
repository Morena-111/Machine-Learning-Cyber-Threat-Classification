"""
FastAPI inference service for the trained cyber-threat classifier.

This module exposes the trained BiLSTM + Attention model as an HTTP API so
that the Tkinter desktop app (app.py) — or any other client — can send a
piece of threat intelligence text and receive a predicted threat class with
confidence scores.

Design notes:
  - Model artifacts are loaded lazily on the first request (see load_artifacts).
    This keeps start-up fast and lets the app boot even if the model has not
    been trained yet, so `/docs` and `/health` remain reachable.
  - Custom Keras classes (AttentionLayer, SparseMacroF1, SparseMacroRate,
    SparseAUC) must be passed via custom_objects so Keras can rebuild the model.
  - Requests are validated by Pydantic so empty strings are rejected before
    they reach the model.

Run the server with:
    uvicorn api:app --reload --port 8000

Then test interactively at:
    http://localhost:8000/docs
"""

import pickle
from pathlib import Path
from typing import Dict

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import build_model


# Project root — api.py sits at the repo root, so .parent is the root itself.
ROOT = Path(__file__).resolve().parent

# FastAPI application. Title and version appear in the auto-generated /docs.
app = FastAPI(title="Cyber Threat Classifier", version="1.0.0")

# Lazy-loaded singletons. None until the first prediction request.
model = None
label_encoder = None
tokenizer = None


class PredictionRequest(BaseModel):
    """
    Request body schema for POST /predict.

    FastAPI uses this class to validate incoming JSON. Fields with min_length
    or other constraints are enforced before the handler runs, so bad requests
    fail with a clear 422 response instead of crashing inside the model.
    """

    text: str = Field(
        min_length=1,
        description="Threat intelligence text to classify",
    )


class PredictionResponse(BaseModel):
    """
    Response body schema for POST /predict.

    Includes the top-1 class, its confidence, and the full probability
    distribution across every known class — useful for the Tkinter app's
    probability bar chart and for the confidence-threshold warning.
    """

    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


def load_artifacts():
    """
    Lazily load the trained model, label encoder, and tokenizer into memory.

    Uses module-level globals so the artifacts are loaded only once per
    process. If the model file is missing (training has not yet run), raises
    HTTP 503 with a clear message so the client knows to retrain.

    Raises:
        HTTPException: 503 when model artifacts cannot be found on disk.
    """
    global model, label_encoder, tokenizer

    # Only load if not already loaded — subsequent calls are free.
    if model is None:
        try:
            with open(ROOT / "models" / "label_encoder.pkl", "rb") as handle:
                label_encoder = pickle.load(handle)

            with open(ROOT / "models" / "tokenizer.pkl", "rb") as handle:
                vocabulary = pickle.load(handle)

            # Rebuild the graph from source to avoid deserializing Keras 3's
            # generated TextVectorization NotEqual masking operation.
            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=10000,
                output_mode="int",
                output_sequence_length=50,
                standardize="lower_and_strip_punctuation",
            )
            vectorizer.set_vocabulary(vocabulary)
            model = build_model(vectorizer, num_classes=len(label_encoder.classes_))
            model.load_weights(ROOT / "models" / "best_model.keras")
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            tokenizer = vocabulary

        except FileNotFoundError as error:
            # Return a descriptive error so the caller knows what to do.
            raise HTTPException(
                503,
                "Model artifacts are unavailable. Run `python -m src.train` first.",
            ) from error


@app.get("/")
def root():
    """
    Root endpoint — quick sanity check that the service is up.

    Returns a small JSON object pointing to the interactive docs.
    """
    return {"service": "Cyber Threat Classifier", "docs": "/docs"}


@app.get("/health")
def health():
    """
    Health check endpoint.

    Reports "ready" if the model file exists on disk, otherwise
    "model_not_trained". Useful for uptime monitors and for the Tkinter app
    to detect a missing model before sending predictions.
    """
    return {
        "status": "ready"
        if (ROOT / "models" / "best_model.keras").exists()
        else "model_not_trained"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Classify a piece of threat intelligence text.

    Args:
        request: PredictionRequest containing the raw text to classify.

    Returns:
        PredictionResponse with predicted_class, confidence, and
        all_probabilities (a dict of class name -> probability).

    Notes:
        The model is trained to accept raw strings, so the tokenizer is only
        loaded for compatibility and potential future use — it is not applied
        here because TextVectorization is embedded inside the model.
    """
    # Ensure model + encoder are loaded before running inference.
    load_artifacts()

    # Model returns shape (1, num_classes); take the first (only) row.
    probs = model.predict([request.text], verbose=0)[0]

    # Highest-probability class index.
    index = int(probs.argmax())

    return PredictionResponse(
        predicted_class=str(label_encoder.classes_[index]),
        confidence=float(probs[index]),
        # Build the full distribution dict for the Tkinter UI bar chart.
        all_probabilities={
            str(label): float(prob)
            for label, prob in zip(label_encoder.classes_, probs)
        },
    )
