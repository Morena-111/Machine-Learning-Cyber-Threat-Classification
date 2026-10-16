"""
Evaluation workflow for the ITRI625 Cyber Threat Classification project.

Produces test metrics and the rubric-required figures:
  - results/classification_report.csv
  - results/test_metrics.json
  - results/roc_curve.png
  - results/pr_curve.png
  - results/confusion_matrix.png
  - results/loss_curve.png
  - results/shap_explanation.png
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from src.data_loader import load_datasets
from src.model import build_model
from src.preprocess import encode_labels, extract_entity_examples, prepare_text_data


ROOT = Path(__file__).resolve().parents[1]


def _save_figure(path):
    """Save current figure at 300 DPI and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _predict(model, texts, batch_size=32):
    """
    Predict on a list of raw strings using a tf.data.Dataset.

    Passing a NumPy array of Python strings to model.predict() triggers a
    Keras 3 bug ("Invalid dtype: str12256") when the model contains a
    TextVectorization layer. Wrapping the strings in a tf.data.Dataset with
    an explicit tf.string dtype bypasses that path.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(texts, dtype=tf.string)
    ).batch(batch_size)
    return model.predict(dataset, verbose=0)


def _rebuild_model(models_dir, encoder):
    """
    Rebuild the architecture and load weights only.

    This bypasses a Keras 3 bug where TextVectorization's internal NotEqual
    mask op cannot be deserialized from a saved model (.keras or .h5).
    """
    with open(models_dir / "tokenizer.pkl", "rb") as handle:
        vocabulary = pickle.load(handle)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=10000,
        output_mode="int",
        output_sequence_length=50,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.set_vocabulary(vocabulary)

    model = build_model(vectorizer, num_classes=len(encoder.classes_))
    model.load_weights(models_dir / "best_model.h5")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_loss_curve(log_path, output):
    """Plot train/val loss with early-stop epoch marker."""
    log = pd.read_csv(log_path)
    epochs = np.arange(1, len(log) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, log["loss"], label="Training loss")
    plt.plot(epochs, log["val_loss"], label="Validation loss")

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
    """Load model, predict on test split, produce metrics and figures."""
    results = ROOT / "results"
    models = ROOT / "models"
    logs = ROOT / "logs"
    results.mkdir(exist_ok=True)

    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)

    _, _, raw_test = load_datasets(ROOT / "data")
    test_examples = extract_entity_examples(raw_test)
    texts = prepare_text_data(test_examples)
    labels = encode_labels(test_examples, encoder)

    model = _rebuild_model(models, encoder)

    probabilities = _predict(model, texts)
    predicted = probabilities.argmax(axis=1)
    target_names = encoder.classes_.tolist()

    report = classification_report(
        labels,
        predicted,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(results / "classification_report.csv")

    metrics = {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }

    y_onehot = tf.keras.utils.to_categorical(labels, len(target_names))
    metrics["macro_auc_ovr"] = 0.0
    roc_class_count = 0

    plt.figure(figsize=(9, 8))
    for index, name in enumerate(target_names):
        if y_onehot[:, index].min() == y_onehot[:, index].max():
            continue
        fpr, tpr, _ = roc_curve(y_onehot[:, index], probabilities[:, index])
        value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={value:.2f})")
        metrics["macro_auc_ovr"] += value
        roc_class_count += 1

    metrics["macro_auc_ovr"] = (
        metrics["macro_auc_ovr"] / roc_class_count if roc_class_count else None
    )
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("One-vs-Rest ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8, loc="lower right")
    _save_figure(results / "roc_curve.png")

    plt.figure(figsize=(9, 8))
    for index, name in enumerate(target_names):
        if y_onehot[:, index].min() == y_onehot[:, index].max():
            continue
        precision, recall, _ = precision_recall_curve(
            y_onehot[:, index], probabilities[:, index]
        )
        plt.plot(recall, precision, label=name)
    plt.title("One-vs-Rest precision-recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8, loc="lower left")
    _save_figure(results / "pr_curve.png")

    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predicted,
        display_labels=target_names,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Test-set confusion matrix")
    _save_figure(results / "confusion_matrix.png")

    plot_loss_curve(logs / "training_log.csv", results / "loss_curve.png")

    (results / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def create_shap_explanation(max_samples=20):
    """Generate a SHAP text explanation on a small test sample."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("Install shap: pip install shap") from error

    models = ROOT / "models"
    results = ROOT / "results"

    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)

    model = _rebuild_model(models, encoder)

    _, _, raw_test = load_datasets(ROOT / "data")
    test_examples = extract_entity_examples(raw_test)
    texts = prepare_text_data(test_examples)

    # Use the same Dataset-based prediction path to avoid the dtype bug.
    explainer = shap.Explainer(
        lambda values: _predict(model, values, batch_size=16),
        shap.maskers.Text(),
    )
    values = explainer(texts[:max_samples])
    shap.plots.text(values[0, :, 0], display=False)
    plt.savefig(results / "shap_explanation.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Primary evaluation: metrics + all 4 plots + test_metrics.json
    metrics = evaluate_model()
    print("\n=== FINAL TEST SET METRICS ===")
    for key, value in metrics.items():
        print(f"  {key:20s}: {value}")

    # Bonus: SHAP XAI explanation for Section 2.7.
    # Failures here should not block the primary output, so we catch and report.
    print("\nGenerating SHAP explanation (this may take ~1 minute)...")
    try:
        create_shap_explanation(max_samples=10)
        print("SHAP explanation saved to results/shap_explanation.png")
    except Exception as error:
        print(f"SHAP generation failed: {error}")
        print("The primary evaluation outputs above are still valid.")