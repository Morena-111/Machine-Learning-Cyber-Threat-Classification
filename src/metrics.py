"""
Custom Keras callback for computing precision, recall, and F1-score on the
validation set after every epoch.

Why this file exists:
    Keras's built-in metrics list (accuracy, precision, recall) reports
    per-batch averages that are not reliable for imbalanced, multi-class
    problems. This callback instead recomputes macro-averaged precision,
    recall, and F1 across the ENTIRE validation set at the end of each epoch,
    which is the correct approach when classes are imbalanced.

    These values are:
      1. Appended to per-metric history lists (for plotting/analysis later).
      2. Injected into the `logs` dict so CSVLogger and TensorBoard record
         them alongside the built-in metrics.
"""

import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score


class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Compute macro-averaged precision, recall, and F1-score on the validation
    set after every training epoch.

    Macro averaging treats every class equally, regardless of frequency — the
    right choice when some threat categories are rare and we do not want them
    to be drowned out by the dominant classes.

    Attributes:
        validation_dataset: A tf.data.Dataset yielding (texts, labels) batches.
        precision_history:  List of precision values, one per epoch.
        recall_history:     List of recall values, one per epoch.
        f1_history:         List of F1 values, one per epoch.
    """

    def __init__(self, validation_dataset):
        """
        Store the validation dataset and initialise empty history lists.

        Args:
            validation_dataset: Iterable yielding (batch_texts, batch_labels).
                                Must be a tf.data.Dataset so batch_labels
                                supports `.numpy()`.
        """
        super().__init__()
        self.validation_dataset = validation_dataset

        # Per-epoch metric histories — used later for plotting and report tables.
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Recompute validation metrics across the full validation set.

        Called automatically by Keras after every epoch. We iterate over the
        validation dataset batch-by-batch, collect all true and predicted
        labels, then compute the metrics once on the full arrays.

        Args:
            epoch (int): Current epoch index (0-based), supplied by Keras.
            logs (dict | None): Keras-owned dict of metrics for this epoch.
                                We inject our custom metrics into it so other
                                callbacks (CSVLogger, TensorBoard) record them.
        """
        if logs is None:
            logs = {}

        # Accumulators for the full validation pass.
        y_true = []
        y_pred = []

        # Iterate over batches — this keeps memory usage low for large sets.
        for batch_texts, batch_labels in self.validation_dataset:
            predictions = self.model.predict(
                batch_texts,
                verbose=0  # Silence per-batch progress bar.
            )

            # Convert softmax probabilities to hard class indices.
            predicted_classes = np.argmax(
                predictions,
                axis=1
            )

            # batch_labels is a tf.Tensor — convert to NumPy to match predictions.
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted_classes)

        # Macro averaging: every class contributes equally to the score.
        # zero_division=0 avoids crashes when a class has no predictions.
        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        # Store for post-training analysis and plots.
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

        # Inject into Keras logs so other callbacks (CSVLogger, TensorBoard)
        # pick these up automatically under val_precision / val_recall / val_f1.
        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1

        # Print a compact line so the user sees progress during training.
        print(
            f" - val_precision: {precision:.4f}"
            f" - val_recall: {recall:.4f}"
            f" - val_f1: {f1:.4f}"
        )