import tensorflow as tf


def create_vectorizer(train_texts, max_tokens=10000, sequence_length=50):
    """
    Create and adapt a Keras TextVectorization layer
    using training text only.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
    )

    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(64))

    return vectorizer

def build_model(vectorizer, num_classes):
    """
    Build the neural network for cyber-threat entity classification.
    """

    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")

    x = vectorizer(inputs)

    x = tf.keras.layers.Embedding(
        input_dim=vectorizer.vocabulary_size(),
        output_dim=128,
        mask_zero=True,
        name="embedding"
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classification"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cyber_threat_entity_classifier"
    )

    return model

def compile_model(model):
    """
    Compile the cyber-threat classification model.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model