import tensorflow as tf
import pandas as pd


from src.data_loader import load_datasets
from src.preprocess import (
    extract_entity_examples,
    prepare_text_data,
    create_label_encoder,
    calculate_class_weights,
)
from src.model import create_vectorizer, build_model, compile_model
from src.metrics import ValidationMetricsCallback


def main():
    print("Loading datasets...")

    train_records, validation_records, test_records = load_datasets()

    print(f"Training documents: {len(train_records)}")
    print(f"Validation documents: {len(validation_records)}")
    print(f"Test documents: {len(test_records)}")

    # ---------------------------------------------------------
    # Prepare training data
    # ---------------------------------------------------------
    train_examples = extract_entity_examples(train_records)

    label_encoder = create_label_encoder(train_examples)

    train_texts = prepare_text_data(train_examples)
    train_labels = label_encoder.transform(
        [example["label"] for example in train_examples]
    )

    # ---------------------------------------------------------
    # Prepare validation data
    # ---------------------------------------------------------
    validation_examples = extract_entity_examples(validation_records)

    validation_texts = prepare_text_data(validation_examples)
    validation_labels = label_encoder.transform(
        [example["label"] for example in validation_examples]
    )

    # ---------------------------------------------------------
    # Prepare test data
    # ---------------------------------------------------------
    test_examples = extract_entity_examples(test_records)

    test_texts = prepare_text_data(test_examples)
    test_labels = label_encoder.transform(
        [example["label"] for example in test_examples]
    )

    print(f"\nTraining examples: {len(train_texts)}")
    print(f"Validation examples: {len(validation_texts)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # ---------------------------------------------------------
    # Create text vectorizer using training data only
    # ---------------------------------------------------------
    print("\nCreating text vectorizer...")

    vectorizer = create_vectorizer(train_texts)

    print(f"Vocabulary size: {vectorizer.vocabulary_size()}")

    # ---------------------------------------------------------
    # Build and compile model
    # ---------------------------------------------------------
    print("\nBuilding model...")

    model = build_model(
        vectorizer,
        num_classes=len(label_encoder.classes_)
    )

    model = compile_model(model)

    print("\nModel compiled successfully.")

    # ---------------------------------------------------------
    # Calculate class weights
    # ---------------------------------------------------------
    class_weights = calculate_class_weights(train_labels)

    print("\nClass weights calculated.")

    # ---------------------------------------------------------
    # Create TensorFlow datasets
    # ---------------------------------------------------------
    batch_size = 32

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_texts, train_labels)
    )

    train_dataset = (
        train_dataset
        .shuffle(len(train_texts), seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_texts, validation_labels)
    )

    validation_dataset = (
        validation_dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_texts, test_labels)
    )

    test_dataset = (
        test_dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\nTensorFlow datasets created successfully.")

    # ---------------------------------------------------------
    # Early stopping
    # ---------------------------------------------------------
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    validation_metrics = ValidationMetricsCallback(
            validation_dataset
        )

    print("\nEarly stopping configured.")

    
    
    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------
    print("\nStarting model training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            validation_metrics
        ],
        verbose=1
    )

    print("\nTraining completed.")

epochs_completed = len(history.history["loss"])

print(f"Epochs completed: {epochs_completed}")

# ---------------------------------------------------------
# Save training history
# ---------------------------------------------------------
history_data = {
    "epoch": range(1, epochs_completed + 1),
    "loss": history.history["loss"],
    "accuracy": history.history["accuracy"],
    "val_loss": history.history["val_loss"],
    "val_accuracy": history.history["val_accuracy"],
    "val_precision": validation_metrics.precision_history,
    "val_recall": validation_metrics.recall_history,
    "val_f1": validation_metrics.f1_history,
}

history_df = pd.DataFrame(history_data)

history_df.to_csv(
    "results/training_history.csv",
    index=False
)

print("\nTraining history saved to:")
print("results/training_history.csv")


if __name__ == "__main__":
    main()