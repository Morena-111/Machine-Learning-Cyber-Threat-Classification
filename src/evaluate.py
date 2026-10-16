"""Evaluation, high-resolution figures, and optional SHAP explanation."""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, auc, classification_report,
                             precision_recall_curve, roc_curve)

from src.data_loader import load_datasets
from src.model import AttentionLayer, SparseAUC, SparseMacroF1, SparseMacroRate
from src.preprocess import prepare_document_dataset


ROOT = Path(__file__).resolve().parents[1]


def _save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curve(log_path, output):
    log = pd.read_csv(log_path)
    epochs = np.arange(1, len(log) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, log["loss"], label="Training loss")
    plt.plot(epochs, log["val_loss"], label="Validation loss")
    best = int(log["val_loss"].idxmin()) + 1
    plt.axvline(best, color="crimson", linestyle="--", label=f"Best epoch / early-stop restore: {best}")
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse categorical cross-entropy")
    plt.legend()
    _save_figure(output)


def evaluate_model():
    """Create metrics and plots from saved training artifacts."""
    import tensorflow as tf
    results, models, logs = ROOT / "results", ROOT / "models", ROOT / "logs"
    results.mkdir(exist_ok=True)
    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)
    _, _, raw_test = load_datasets(ROOT / "data")
    texts, labels, _, _ = prepare_document_dataset(raw_test, encoder=encoder)
    custom = {"AttentionLayer": AttentionLayer, "SparseMacroF1": SparseMacroF1, "SparseMacroRate": SparseMacroRate, "SparseAUC": SparseAUC}
    model = tf.keras.models.load_model(models / "best_model.keras", custom_objects=custom)
    probabilities = model.predict(np.asarray(texts, dtype=str), verbose=0)
    predicted = probabilities.argmax(axis=1)
    target_names = encoder.classes_.tolist()
    report = classification_report(labels, predicted, target_names=target_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(results / "classification_report.csv")
    metrics = {"accuracy": float(np.mean(predicted == labels)), "macro_precision": float(report["macro avg"]["precision"]), "macro_recall": float(report["macro avg"]["recall"]), "macro_f1": float(report["macro avg"]["f1-score"])}

    y_onehot = tf.keras.utils.to_categorical(labels, len(target_names))
    metrics["macro_auc_ovr"] = 0.0
    plt.figure(figsize=(8, 7))
    for index, name in enumerate(target_names):
        if y_onehot[:, index].min() == y_onehot[:, index].max():
            continue
        fpr, tpr, _ = roc_curve(y_onehot[:, index], probabilities[:, index])
        value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={value:.2f})")
        metrics["macro_auc_ovr"] += value
    metrics["macro_auc_ovr"] /= len(target_names)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("One-vs-Rest ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    _save_figure(results / "roc_curve.png")

    plt.figure(figsize=(8, 7))
    for index, name in enumerate(target_names):
        precision, recall, _ = precision_recall_curve(y_onehot[:, index], probabilities[:, index])
        plt.plot(recall, precision, label=name)
    plt.title("One-vs-Rest precision-recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    _save_figure(results / "pr_curve.png")

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(labels, predicted, display_labels=target_names, xticks_rotation=35, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test-set confusion matrix")
    _save_figure(results / "confusion_matrix.png")
    plot_loss_curve(logs / "training_log.csv", results / "loss_curve.png")
    (results / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def create_shap_explanation(max_samples=20):
    """Use SHAP's model-agnostic explainer on a small held-out text sample."""
    try:
        import shap
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install the optional 'shap' dependency to generate this figure.") from error
    models, results = ROOT / "models", ROOT / "results"
    custom = {"AttentionLayer": AttentionLayer, "SparseMacroF1": SparseMacroF1, "SparseMacroRate": SparseMacroRate, "SparseAUC": SparseAUC}
    model = tf.keras.models.load_model(models / "best_model.keras", custom_objects=custom)
    with open(models / "label_encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)
    _, _, raw_test = load_datasets(ROOT / "data")
    texts, _, _, _ = prepare_document_dataset(raw_test, encoder=encoder)
    explainer = shap.Explainer(lambda values: model.predict(np.asarray(values, dtype=str), verbose=0), shap.maskers.Text())
    values = explainer(texts[:max_samples])
    shap.plots.text(values[0, :, 0], display=False)
    plt.savefig(results / "shap_explanation.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print(evaluate_model())
