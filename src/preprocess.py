import re
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