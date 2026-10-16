"""
Custom Keras callback for computing macro-averaged validation metrics.

Why this file exists:
    Keras's built-in metrics report per-batch averages that are not reliable
    for imbalanced, multi-class problems. This callback recomputes macro
    precision, recall, and F1 across the FULL validation set at the end of
    each epoch and injects them into Keras's `logs` dict so CSVLogger and
    TensorBoard record them automatically.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score


class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Compute macro-averaged precision, recall, and F1-score on the validation
    set after every training epoch.
    """

    def __init__(self, validation_dataset):
        """
        Args:
            validation_dataset: A tf.data.Dataset yielding (texts, labels)
                                batches. Texts must be raw strings so the
                                model's internal TextVectorization can run.
        """
        super().__init__()
        self.validation_dataset = validation_dataset
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        """Recompute validation metrics across the full validation set."""
        if logs is None:
            logs = {}

        y_true = []
        y_pred = []

        for batch_texts, batch_labels in self.validation_dataset:
            predictions = self.model.predict(batch_texts, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted_classes)

        precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

        # Inject into Keras logs so CSVLogger records them.
        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1

        print(
            f" - val_precision: {precision:.4f}"
            f" - val_recall: {recall:.4f}"
            f" - val_f1: {f1:.4f}"
        )