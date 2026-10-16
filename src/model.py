"""
Model definition: BiLSTM + Attention for multi-class entity classification.

This module builds the full Keras model including a custom attention pooling
layer, a TextVectorization-based input pipeline, and a softmax classifier.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="cyber_threat")
class AttentionLayer(tf.keras.layers.Layer):
    """
    Trainable temporal attention pooling for BiLSTM sequences.

    Learns a scalar attention weight per timestep and returns the
    weighted sum of the sequence. Supports an optional mask so padded
    timesteps are ignored.
    """

    def build(self, input_shape):
        """Create the layer's trainable parameters."""
        self.weight = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Apply attention pooling.

        Args:
            inputs: Tensor of shape (batch, timesteps, features).
            mask:   Optional boolean tensor marking valid timesteps.

        Returns:
            Tensor of shape (batch, features).
        """
        scores = tf.squeeze(
            tf.tanh(tf.matmul(inputs, self.weight) + self.bias), axis=-1
        )
        if mask is not None:
            scores += (1.0 - tf.cast(mask, scores.dtype)) * -1e9
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)


@tf.keras.utils.register_keras_serializable(package="cyber_threat")
class SparseMacroF1(tf.keras.metrics.Metric):
    """Streaming macro F1 that accepts sparse integer labels."""

    def __init__(self, num_classes, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.matrix = self.add_weight(
            name="confusion",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=self.dtype
        )
        self.matrix.assign_add(cm)

    def result(self):
        tp = tf.linalg.diag_part(self.matrix)
        precision = tf.math.divide_no_nan(tp, tf.reduce_sum(self.matrix, axis=0))
        recall = tf.math.divide_no_nan(tp, tf.reduce_sum(self.matrix, axis=1))
        return tf.reduce_mean(
            tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        )

    def reset_state(self):
        self.matrix.assign(tf.zeros_like(self.matrix))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


@tf.keras.utils.register_keras_serializable(package="cyber_threat")
class SparseMacroRate(SparseMacroF1):
    """Macro precision OR recall for sparse multiclass targets."""

    def __init__(self, num_classes, mode="precision", name="precision", **kwargs):
        self.mode = mode
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def result(self):
        tp = tf.linalg.diag_part(self.matrix)
        denominator = tf.reduce_sum(
            self.matrix, axis=0 if self.mode == "precision" else 1
        )
        return tf.reduce_mean(tf.math.divide_no_nan(tp, denominator))

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


@tf.keras.utils.register_keras_serializable(package="cyber_threat")
class SparseAUC(tf.keras.metrics.Metric):
    """One-vs-rest AUC adapter for sparse labels."""

    def __init__(self, num_classes, name="auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.auc = tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.one_hot(
            tf.cast(tf.reshape(y_true, [-1]), tf.int32), self.num_classes
        )
        self.auc.update_state(labels, y_pred, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


def create_vectorizer(train_texts, max_tokens=10000, sequence_length=50):
    """
    Create and adapt a TextVectorization layer using training text only.

    Fitting on training text alone prevents information from the validation
    or test splits leaking into the model's vocabulary.
    """
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(train_texts).batch(64)
    )
    return vectorizer


def build_model(vectorizer, num_classes):
    """
    Build the BiLSTM + Attention classifier.

    Architecture:
        Input (string) -> TextVectorization -> Embedding(128) ->
        BiLSTM(64) -> AttentionLayer -> Dropout(0.4) ->
        Dense(64, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)
    """
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")

    x = vectorizer(inputs)

    x = tf.keras.layers.Embedding(
        input_dim=vectorizer.vocabulary_size(),
        output_dim=128,
        mask_zero=True,
        name="embedding",
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True), name="bilstm"
    )(x)

    x = AttentionLayer(name="attention")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification"
    )(x)

    return tf.keras.Model(
        inputs=inputs, outputs=outputs, name="cyber_threat_classifier"
    )


def compile_model(model):
    """Compile with Adam and sparse categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model