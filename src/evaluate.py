"""
Evaluation workflow for the ITRI625 Cyber Threat Classification project.

This module loads the trained model and its preprocessing artifacts, runs the
model on the held-out test split, and produces the rubric-required outputs:

  - results/classification_report.csv   (per-class precision/recall/F1)
  - results/test_metrics.json           (aggregate test metrics)
  - results/roc_curve.png               (One-vs-Rest ROC curves)
  - results/pr_curve.png                (One-vs-Rest precision-recall curves)
  - results/confusion_matrix.png        (test-set confusion matrix)
  - results/loss_curve.png              (training vs validation loss)

An optional SHAP explanation (results/shap_explanation.png) can also be
generated to support the Explainable AI bonus requirement.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from src.data_loader import load_datasets
from src.model import AttentionLayer, SparseAUC, SparseMacroF1, SparseMacroRate
from src.preprocess import encode_labels, extract_entity_examples, prepare_text_data


# Project root — evaluate.py lives in src/, so parents[1] is the repo root.
ROOT = Path(__file__).resolve().parents[1]


def _save_figure(path):
    """
    Save the current Matplotlib figure to disk at high resolution and close it.

    Centralising this in a helper keeps every plot at the same DPI (300) and
    prevents figures from accumulating in memory across multiple plots.

    Args:
        path (str | Path): Destination file path (e.g. results/roc_curve.png).
    """
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curve(log_path, output):
    """
    Plot training vs validation loss from the CSV logger's output.

    The best validation epoch is marked with a vertical line so the effect of
    early stopping is visible in the figure (this supports Section 2.6).

    Args:
        log_path (str | Path): Path to the CSV written by the CSVLogger callback.
        output (str | Path):   Path where the PNG should be saved.
    """
    log = pd.read_csv(log_path)
    epochs = np.arange(1, len(log) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, log["loss"], label="Training loss")
    plt.plot(epochs, log["val_loss"], label="Validation loss")

    # Identify the epoch with the lowest validation loss (this is the epoch
    # whose weights are restored by EarlyStopping(restore_best_weights=True)).
    best = int(log["val_loss"].idxmin()) + 1
    plt.axvline(
        best,
        color="crimson",
        linestyle="--",
        label=f"Best epoch / early-stop restore: {best}",
    )

    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse categorical cross-entropy")
    plt.legend()
    _save_figure(output)


def evaluate_model():
    """
    Load the trained model and produce all test-set metrics and figures.

    Steps:
      1. Load the label encoder and test split from disk.
      2. Preprocess the test text into model-ready sequences.
      3. Load the saved Keras model (including custom layers/metrics).
      4. Predict class probabilities and derive hard labels.
      5. Write the classification report and aggregate metrics to results/.
      6. Generate ROC, PR, confusion matrix, and loss curve figures.

    Returns:
        dict: Aggregate test metrics (accuracy, macro precision/recall/F1, AUC).
    """
    import tensorflow as tf  # Imported here so the module can be inspected without TF installed.

    results, models, logs = ROOT / "results", ROOT / "models", ROOT / "logs"
    results.mkdir(exist_ok=True)

    # Reload the exact label encoder used during training so class indices align.
    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)

    # Load and preprocess the held-out entity examples with the training encoder.
    _, _, raw_test = load_datasets(ROOT / "data")
    test_examples = extract_entity_examples(raw_test)
    texts = prepare_text_data(test_examples)
    labels = encode_labels(test_examples, encoder)

    # Custom objects must be registered when loading the model, otherwise Keras
    # cannot reconstruct the AttentionLayer or the custom metric classes.
    custom = {
        "AttentionLayer": AttentionLayer,
        "SparseMacroF1": SparseMacroF1,
        "SparseMacroRate": SparseMacroRate,
        "SparseAUC": SparseAUC,
    }
    model = tf.keras.models.load_model(models / "best_model.keras", custom_objects=custom)

    # Probabilities are needed for ROC/PR curves; argmax gives hard class labels.
    probabilities = model.predict(np.asarray(texts, dtype=str), verbose=0)
    predicted = probabilities.argmax(axis=1)

    # Human-readable class names in the same order as the encoder's classes_.
    target_names = encoder.classes_.tolist()

    # Per-class precision/recall/F1 report, saved for Section 3.3 analysis.
    report = classification_report(
        labels,
        predicted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(results / "classification_report.csv")

    # Aggregate metrics used in the theory write-up.
    metrics = {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }

    # ---- ROC curves (One-vs-Rest for multi-class) ----
    # One-hot encode the true labels so each class can be treated as a binary problem.
    y_onehot = tf.keras.utils.to_categorical(labels, len(target_names))
    metrics["macro_auc_ovr"] = 0.0
    roc_class_count = 0

    plt.figure(figsize=(8, 7))
    for index, name in enumerate(target_names):
        # Skip classes that have only one unique value — roc_curve would fail.
        if y_onehot[:, index].min() == y_onehot[:, index].max():
            continue
        fpr, tpr, _ = roc_curve(y_onehot[:, index], probabilities[:, index])
        value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={value:.2f})")
        metrics["macro_auc_ovr"] += value
        roc_class_count += 1

    # Average AUC over all classes that produced a curve.
    metrics["macro_auc_ovr"] /= roc_class_count
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("One-vs-Rest ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    _save_figure(results / "roc_curve.png")

    # ---- Precision-Recall curves (One-vs-Rest) ----
    plt.figure(figsize=(8, 7))
    for index, name in enumerate(target_names):
        # Skip classes that have no positive/negative test entity examples.
        if y_onehot[:, index].min() == y_onehot[:, index].max():
            continue
        precision, recall, _ = precision_recall_curve(
            y_onehot[:, index], probabilities[:, index]
        )
        plt.plot(recall, precision, label=name)
    plt.title("One-vs-Rest precision-recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    _save_figure(results / "pr_curve.png")

    # ---- Confusion matrix ----
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predicted,
        display_labels=target_names,
        xticks_rotation=35,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Test-set confusion matrix")
    _save_figure(results / "confusion_matrix.png")

    # ---- Loss curve (from the CSV logger written during training) ----
    plot_loss_curve(logs / "training_log.csv", results / "loss_curve.png")

    # Persist the aggregate metrics for use in the theory report.
    (results / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def create_shap_explanation(max_samples=20):
    """
    Generate a SHAP text explanation for a small sample of test entity inputs.

    SHAP is model-agnostic here: we wrap the trained model in a callable that
    accepts raw text strings and returns predicted probabilities. The resulting
    text plot highlights which tokens drove the prediction — this satisfies the
    XAI bonus requirement (Section 2.7).

    Args:
        max_samples (int): Number of test entity inputs to explain. Keep this small
                           because SHAP is computationally expensive.
    """
    try:
        import shap
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Install the optional 'shap' dependency to generate this figure."
        ) from error

    models, results = ROOT / "models", ROOT / "results"

    custom = {
        "AttentionLayer": AttentionLayer,
        "SparseMacroF1": SparseMacroF1,
        "SparseMacroRate": SparseMacroRate,
        "SparseAUC": SparseAUC,
    }
    model = tf.keras.models.load_model(models / "best_model.keras", custom_objects=custom)

    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)

    _, _, raw_test = load_datasets(ROOT / "data")
    test_examples = extract_entity_examples(raw_test)
    texts = prepare_text_data(test_examples)

    # The explainer needs a callable that maps a list of strings to probabilities.
    explainer = shap.Explainer(
        lambda values: model.predict(np.asarray(values, dtype=str), verbose=0),
        shap.maskers.Text(),
    )
    values = explainer(texts[:max_samples])

    # Plot the first sample's class-0 explanation and save the figure to disk.
    shap.plots.text(values[0, :, 0], display=False)
    plt.savefig(results / "shap_explanation.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(evaluate_model())
