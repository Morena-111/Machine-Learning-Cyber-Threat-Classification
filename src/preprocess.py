import re
import ast
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def extract_entity_examples(records):
    """
    Convert document-level JSONL records into entity-level
    classification examples.
    """

    examples = []

    for record in records:
        text = record.get("text", "")
        entities = record.get("entities", [])

        for entity in entities:
            start = entity["start_offset"]
            end = entity["end_offset"]
            label = entity["label"]

            entity_text = text[start:end]

            # Skip invalid or empty entity spans
            if not entity_text.strip():
                continue

            examples.append(
                {
                    "document_id": record["id"],
                    "text": text,
                    "entity": entity_text,
                    "label": label,
                    "start_offset": start,
                    "end_offset": end,
                }
            )

    return examples


def create_model_input(example, context_window=150):
    """
    Create a focused text representation around the target entity.
    """

    text = example["text"]
    start = example["start_offset"]
    end = example["end_offset"]

    context_start = max(0, start - context_window)
    context_end = min(len(text), end + context_window)

    before = text[context_start:start]
    entity = text[start:end]
    after = text[end:context_end]

    model_input = (
        f"{before} [ENTITY] {entity} [/ENTITY] {after}"
    )

    return re.sub(r"\s+", " ", model_input).strip()

from sklearn.preprocessing import LabelEncoder


def extract_document_label(record):
    """Return the first available target label from CSV or JSONL records.

    CSV records store entity annotations in ``Entries`` as JSON; the supplied
    JSONL corpus stores them in ``entities``.  A document with no annotation is
    intentionally returned as ``None`` rather than guessed as benign.
    """
    entries = record.get("Entries", record.get("entities", []))
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except json.JSONDecodeError:
            try:
                entries = ast.literal_eval(entries)
            except (SyntaxError, ValueError):
                return None
    if not entries:
        return None
    first = entries[0]
    return first.get("label") if isinstance(first, dict) else None


def prepare_document_dataset(records, encoder=None, fit_encoder=False):
    """Create text/label arrays for document-level threat classification."""
    texts, labels, ids = [], [], []
    for record in records:
        label = extract_document_label(record)
        text = str(record.get("text", "")).strip()
        if label and text:
            texts.append(re.sub(r"\s+", " ", text))
            labels.append(label)
            ids.append(record.get("id"))
    if fit_encoder:
        encoder = LabelEncoder().fit(labels)
    if encoder is None:
        raise ValueError("An encoder is required unless fit_encoder=True.")
    known = set(encoder.classes_)
    keep = [i for i, label in enumerate(labels) if label in known]
    return ([texts[i] for i in keep], encoder.transform([labels[i] for i in keep]),
            [ids[i] for i in keep], encoder)


def save_label_encoder(encoder, path="models/label_encoder.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as output:
        pickle.dump(encoder, output)


def create_label_encoder(train_examples):
    """
    Create and fit a label encoder using only the training data.
    """

    labels = [example["label"] for example in train_examples]

    encoder = LabelEncoder()
    encoder.fit(labels)

    return encoder


def encode_labels(examples, encoder):
    """
    Convert string labels into numerical class IDs.
    """

    labels = [example["label"] for example in examples]

    return encoder.transform(labels)

def prepare_text_data(examples):
    """
    Create model input text from entity examples.
    """

    texts = [
        create_model_input(example)
        for example in examples
    ]

    return texts

def prepare_dataset(records, encoder):
    """
    Convert raw JSONL records into model texts and encoded labels.
    """

    examples = extract_entity_examples(records)
    texts = prepare_text_data(examples)
    labels = encode_labels(examples, encoder)

    return examples, texts, labels

def calculate_class_weights(labels):
    """
    Calculate balanced class weights from the training labels.
    """

    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }
