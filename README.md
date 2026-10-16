# ITRI625 Cyber Threat Classification

TensorFlow/Keras BiLSTM-with-attention classifier for cybersecurity threat-intelligence (CTI) entity mentions. It performs entity classification (NER-style) across STIX 2.1 entity types and supports the rubric CSV splits and the repository's existing JSONL splits.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

The training command performs split-ID leakage checks on entity source documents, writes class-distribution plots, saves `models/best_model.h5`, a vectorizer vocabulary (`models/tokenizer.pkl`), entity-label encoder, CSV/TensorBoard logs, and reports the epochs completed. Evaluation creates the ROC, precision-recall, confusion-matrix, loss, report, and JSON metric artifacts. Run `python -c "from src.evaluate import create_shap_explanation; create_shap_explanation()"` after evaluation to create the optional SHAP image.

## Dataset

Place the three Kaggle CSVs in `data/` as `cyber-threat-intelligence-splited_train.csv`, `cyber-threat-intelligence-splited_validate.csv`, and `cyber-threat-intelligence-splited_test.csv`. If they are absent, the supplied `train.jsonl`, `validation.jsonl`, and `test.jsonl` splits are used. The entity pipeline uses every labelled entity span, preserves its source-document ID for leakage checks, wraps the target with `[ENTITY]` markers, and includes local context for classification.

## Run the desktop application

```powershell
uvicorn api:app --reload
# In another terminal
python app.py
```

Open `http://localhost:8000/docs` for interactive API documentation. The Tkinter client offers single-text and CSV batch prediction; batch input must contain a `text` column. Predictions under 60% confidence are clearly flagged for analyst review.

## Hugging Face Spaces deployment

Create a Docker Space, upload this repository including the trained `models/` artifacts, and set the Space SDK to Docker. The included `Dockerfile` serves FastAPI on port 7860, which Hugging Face Spaces expects. Build locally with `docker build -t cti-classifier .` and run with `docker run -p 7860:7860 cti-classifier`.

## Project layout

- `src/`: preprocessing, model, training and evaluation code
- `models/`: generated Keras model and reusable inference artifacts
- `logs/`: epoch CSV log and TensorBoard runs
- `results/`: high-DPI plots, reports, integrity checks, and test metrics
- `api.py` / `app.py`: FastAPI backend and Tkinter desktop client
- `THEORY_REPORT.md`: architecture, analysis, use-case, and cited theory
