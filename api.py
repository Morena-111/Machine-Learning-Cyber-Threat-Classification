"""
FastAPI inference service for the trained cyber-threat classifier.

Loads the model using rebuild + load_weights to bypass a Keras 3
TextVectorization deserialization bug.

The model was trained on entity spans wrapped in [ENTITY] ... [/ENTITY]
markers. To make inference robust to plain text input, the /predict endpoint
automatically wraps raw input in those markers before prediction.

Run with:
    uvicorn api:app --reload --port 8000
"""

import pickle
from pathlib import Path
from typing import Dict

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from src.model import build_model


ROOT = Path(__file__).resolve().parent

# ============================================================================
# FastAPI app configuration
# ============================================================================
app = FastAPI(
    title="Cyber Threat Classifier API",
    description=(
        "**Entity-level Named Entity Recognition** for cyber threat intelligence text.\n\n"
        "Classifies entity mentions into **20 STIX 2.1 entity types** including "
        "`malware`, `threat-actor`, `vulnerability`, `IPV4`, `EMAIL`, and `DOMAIN`.\n\n"
        "### How to use\n"
        "Send a POST request to `/predict` with a JSON body containing a `text` field. "
        "The model automatically wraps the input in entity markers and returns the "
        "predicted class plus all 20 class probabilities."
    ),
    version="1.0.0",
    tags_metadata=[
        {
            "name": "Prediction",
            "description": "Classify cyber threat intelligence text into entity types.",
        },
        {
            "name": "System",
            "description": "Service health, metadata, and documentation.",
        },
    ],
)

model = None
label_encoder = None


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Threat intelligence text to classify",
        examples=[
            "The malware called Emotet",
            "CVE-2023-1234 is a critical vulnerability",
            "APT29 is a known threat actor",
            "The IP address 192.168.1.1 was in C2 traffic",
        ],
    )


class PredictionResponse(BaseModel):
    predicted_class: str = Field(description="Top-1 predicted entity class")
    confidence: float = Field(description="Probability of the top-1 class (0–1)")
    all_probabilities: Dict[str, float] = Field(
        description="Probability assigned to each of the 20 entity classes"
    )


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


# ============================================================================
# Custom Swagger UI CSS (Solid Grey Background)
# ============================================================================
CUSTOM_SWAGGER_CSS = """
<style>
    :root {
        --bg-page:      #e2e8f0; /* Solid grey page background */
        --bg-card:      #ffffff; /* White cards */
        --bg-muted:     #f8fafc; /* Very light grey for inputs/code */
        --text-primary: #0f172a; /* Dark slate */
        --text-muted:   #334155; /* Medium slate */
        --border:       #cbd5e1; /* Soft grey border */
        --accent:       #10b981; /* Emerald green */
        --accent-dark:  #059669;
        --purple:       #8b5cf6;
        --purple-light: #ede9fe;
    }

    /* Force background on all container levels */
    body, html, #swagger-ui, .swagger-container, .swagger-ui .wrapper {
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Header / Topbar */
    .swagger-ui .topbar {
        background-color: var(--bg-card) !important;
        border-bottom: 1px solid var(--border) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        padding: 10px 0 !important;
    }
    .swagger-ui .topbar .link span {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
    }
    .swagger-ui .topbar-wrapper img {
        display: none !important; /* Hide default FastAPI logo */
    }

    /* Page title */
    .swagger-ui .info .title {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    .swagger-ui .info .title small {
        background: var(--purple) !important;
        color: #fff !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        font-size: 0.75rem !important;
    }

    /* Operation tags */
    .swagger-ui .opblock-tag {
        background: transparent !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--border) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        padding: 16px 0 !important;
    }

    /* Operation cards */
    .swagger-ui .opblock {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-card) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        margin-bottom: 12px !important;
    }
    .swagger-ui .opblock.opblock-post {
        border-left: 4px solid var(--accent) !important;
        background: var(--bg-card) !important;
    }
    .swagger-ui .opblock.opblock-get {
        border-left: 4px solid var(--purple) !important;
        background: var(--bg-card) !important;
    }
    .swagger-ui .opblock .opblock-summary-method {
        border-radius: 4px !important;
        font-weight: 700 !important;
    }
    .swagger-ui .opblock.opblock-post .opblock-summary-method {
        background: var(--accent) !important;
        color: white !important;
    }
    .swagger-ui .opblock.opblock-get .opblock-summary-method {
        background: var(--purple) !important;
        color: white !important;
    }
    .swagger-ui .opblock .opblock-summary-path {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .swagger-ui .btn.execute {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .swagger-ui .btn.execute:hover {
        background-color: var(--accent-dark) !important;
    }
    .swagger-ui .btn.try-out__btn {
        border-color: var(--accent) !important;
        color: var(--accent-dark) !important;
        border-radius: 6px !important;
    }

    /* Code blocks & Try it out inputs */
    .swagger-ui .highlight-code pre,
    .swagger-ui .microlight,
    .swagger-ui textarea {
        background: var(--bg-muted) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    .swagger-ui input[type=text]:focus,
    .swagger-ui textarea:focus {
        border-color: var(--purple) !important;
        box-shadow: 0 0 0 3px var(--purple-light) !important;
        outline: none !important;
    }

    /* Tables */
    .swagger-ui table thead tr td,
    .swagger-ui table thead tr th {
        background: var(--bg-muted) !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--border) !important;
    }
    .swagger-ui table tbody tr td {
        border-bottom: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
    }
    
    /* Model schemas box at bottom */
    .swagger-ui .models {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
</style>
"""


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/", tags=["System"], summary="Service root")
def root():
    return {
        "service": "Cyber Threat Classifier",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/docs"],
        "classes": 20,
    }


@app.get("/health", tags=["System"], summary="Health check")
def health():
    exists = (ROOT / "models" / "best_model.h5").exists()
    return {
        "status": "ready" if exists else "model_not_trained",
        "model_file_present": exists,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify threat intelligence text",
    response_description="Predicted entity class and per-class probabilities",
)
def predict(request: PredictionRequest):
    """
    Classify a piece of threat intelligence text into one of 20 STIX 2.1 entity types.

    **How it works:**
    2. The text is passed through the trained model.
    3. The top class and full probability distribution are returned.
    """
    load_artifacts()

    # Wrap raw input in entity markers to match training distribution
    text = request.text.strip()
    if "[ENTITY]" not in text and "[/ENTITY]" not in text:
        text = f"[ENTITY] {text} [/ENTITY]"

    # Use tf.data.Dataset to avoid the Keras 3 "Unrecognized data type" bug
    ds = tf.data.Dataset.from_tensor_slices(
        tf.constant([text], dtype=tf.string)
    ).batch(1)
    probs = model.predict(ds, verbose=0)[0]
    index = int(probs.argmax())

    return PredictionResponse(
        predicted_class=str(label_encoder.classes_[index]),
        confidence=float(probs[index]),
        all_probabilities={
            str(label): float(prob)
            for label, prob in zip(label_encoder.classes_, probs)
        },
    )


# ============================================================================
# Override /docs to inject the custom bright theme
# ============================================================================
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    """Serve the Swagger UI with the custom grey theme."""
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} — Interactive Docs",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
    body = html.body.decode("utf-8")
    # Inject our CSS immediately after the <head> tag
    styled = body.replace("<head>", f"<head>{CUSTOM_SWAGGER_CSS}", 1)
    return HTMLResponse(content=styled)