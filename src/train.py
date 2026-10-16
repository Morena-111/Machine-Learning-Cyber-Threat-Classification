"""Train the BiLSTM-attention model on labelled CTI entity mentions."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data_loader import load_datasets
from src.preprocess import (
    calculate_class_weights,
    create_label_encoder,
    encode_labels,
    extract_entity_examples,
    prepare_text_data,
    save_label_encoder,
)
from src.model import create_vectorizer, build_model, compile_model
from src.callbacks import ValidationMetricsCallback


ROOT = Path(__file__).resolve().parents[1]


def _windows_safe_text(texts):
    """Replace characters outside cp1252 so TensorFlow does not crash."""
    return [t.encode("cp1252", errors="replace").decode("cp1252") for t in texts]


def _verify_entity_split_integrity(train_examples, validation_examples, test_examples):
    """Ensure no document appears in more than one split."""
    ids = {
        "train": {e["document_id"] for e in train_examples},
        "validation": {e["document_id"] for e in validation_examples},
        "test": {e["document_id"] for e in test_examples},
    }
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Entity-level data leakage detected: {overlaps}")
    return overlaps


def _save_distribution_plots(split_examples, results_dir):
    """Save class-count plots for the splits."""
    rows = [
        {"split": split, "label": e["label"]}
        for split, examples in split_examples.items()
        for e in examples
    ]
    counts = pd.DataFrame(rows).value_counts(["label", "split"]).unstack(fill_value=0)

    counts.plot(kind="bar", figsize=(13, 6))
    plt.title("Entity-label distribution by split")
    plt.xlabel("Entity type")
    plt.ylabel("Entity examples")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "split_distribution.png", dpi=300)
    plt.close()

    counts[["train"]].plot(kind="bar", figsize=(13, 6), legend=False)
    plt.title("Training entity-label distribution")
    plt.xlabel("Entity type")
    plt.ylabel("Entity examples")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "class_distribution.png", dpi=300)
    plt.close()


def run_training(epochs=30, batch_size=32):
    """Full training pipeline. Invoke with: python -m src.train"""
    results_dir = ROOT / "results"
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"
    for d in (results_dir, models_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_records, validation_records, test_records = load_datasets(ROOT / "data")
    print(f"Training documents: {len(train_records)}")
    print(f"Validation documents: {len(validation_records)}")
    print(f"Test documents: {len(test_records)}")

    train_examples = extract_entity_examples(train_records)
    label_encoder = create_label_encoder(train_examples)
    train_texts = _windows_safe_text(prepare_text_data(train_examples))
    train_labels = encode_labels(train_examples, label_encoder)

    validation_examples = extract_entity_examples(validation_records)
    validation_texts = _windows_safe_text(prepare_text_data(validation_examples))
    validation_labels = encode_labels(validation_examples, label_encoder)

    test_examples = extract_entity_examples(test_records)
    test_texts = _windows_safe_text(prepare_text_data(test_examples))
    test_labels = encode_labels(test_examples, label_encoder)

    print(f"\nTraining examples: {len(train_texts)}")
    print(f"Validation examples: {len(validation_texts)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    overlaps = _verify_entity_split_integrity(
        train_examples, validation_examples, test_examples
    )
    _save_distribution_plots(
        {
            "train": train_examples,
            "validation": validation_examples,
            "test": test_examples,
        },
        results_dir,
    )
    (results_dir / "split_integrity.json").write_text(
        json.dumps({"document_id_overlaps": overlaps}, indent=2),
        encoding="utf-8",
    )

    print("\nCreating text vectorizer...")
    vectorizer = create_vectorizer(train_texts)
    print(f"Vocabulary size: {vectorizer.vocabulary_size()}")

    print("\nBuilding model...")
    model = compile_model(
        build_model(vectorizer, num_classes=len(label_encoder.classes_))
    )
    print("Model compiled successfully.")

    class_weights = calculate_class_weights(train_labels)
    print("Class weights calculated.")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    train_dataset = (
        train_dataset.shuffle(len(train_texts), seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_texts, validation_labels)
    )
    validation_dataset = (
        validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    print("TensorFlow datasets created.")

    callbacks = [
        tf.keras.callbacks.CSVLogger(logs_dir / "training_log.csv"),
        tf.keras.callbacks.CSVLogger(results_dir / "training_log.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir / "tensorboard"),
        tf.keras.callbacks.ModelCheckpoint(
            models_dir / "best_model.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ValidationMetricsCallback(validation_dataset),
    ]
    print("Callbacks configured (including per-epoch macro metrics).")

    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )
    print("\nTraining completed.")
    print(f"Epochs completed: {len(history.history['loss'])} / {epochs}")

    # Save final model (ModelCheckpoint already saved the best epoch).
    model.save(models_dir / "best_model.h5")

    # Persist the artifacts needed at inference time.
    save_label_encoder(label_encoder, models_dir / "label_encoder.pkl")
    with open(models_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(vectorizer.get_vocabulary(), f)

    # Save the test split so evaluate.py can compute metrics deterministically.
    with open(models_dir / "test_split.pkl", "wb") as f:
        pickle.dump((test_texts, test_labels), f)

    return history


def main():
    run_training()


if __name__ == "__main__":
    main()