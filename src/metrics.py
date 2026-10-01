import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score


class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    """
    Calculate validation precision, recall and F1-score
    after every training epoch.
    """

    def __init__(self, validation_dataset):
        super().__init__()
        self.validation_dataset = validation_dataset

        self.precision_history = []
        self.recall_history = []
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        y_true = []
        y_pred = []

        for batch_texts, batch_labels in self.validation_dataset:
            predictions = self.model.predict(
                batch_texts,
                verbose=0
            )

            predicted_classes = np.argmax(
                predictions,
                axis=1
            )

            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted_classes)

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

        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1

        print(
            f" - val_precision: {precision:.4f}"
            f" - val_recall: {recall:.4f}"
            f" - val_f1: {f1:.4f}"
        )