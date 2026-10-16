"""
Text preprocessing and label-encoding utilities for the CTI classifier.

This module bridges the raw dataset (CSV or JSONL) and the model. It provides
two layers of preprocessing:

  Document-level (used by the training/evaluation pipeline):
      prepare_document_dataset, extract_document_label,
      save_label_encoder, calculate_class_weights

  Entity-level (used by src/train.py and src/evaluate.py):
      extract_entity_examples, create_model_input, prepare_text_data,
      prepare_dataset, create_label_encoder, encode_labels

Key design choices:
  - The label encoder is fit on TRAINING data only (no leakage).
  - Documents without a usable label are dropped silently but transparently.
  - Whitespace is normalised so the vectorizer sees consistent text.
  - Both CSV ("Entries" as JSON string) and JSONL ("entities" as list)
    data shapes are supported.
"""

import re
import ast
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Entity-level utilities (used by src/train.py and src/evaluate.py)
# ---------------------------------------------------------------------------

def extract_entity_examples(records):
    """
    Convert document-level records into entity-level classification examples.

    Handles both CSV and JSONL data shapes:
      - CSV:  labels live inside an ``Entries`` column as a JSON string.
      - JSONL: labels live directly under an ``entities`` key as a list.

    Each document may contain zero or more annotated entities. This function
    emits one example per (document, entity) pair.

    Args:
        records (list[dict]): Raw CSV or JSONL records.

    Returns:
        list[dict]: One dict per entity with keys document_id, text, entity,
                    label, start_offset, end_offset.
    """
    examples = []

    for record in records:
        text = record.get("text", "")
        # Prefer "entities" (JSONL) but fall back to "Entries" (CSV).
        entities = record.get("entities", record.get("Entries", []))

        # CSV stores entities as a JSON-encoded string — parse it if needed.
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except json.JSONDecodeError:
                try:
                    entities = ast.literal_eval(entities)
                except (SyntaxError, ValueError):
                    continue

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            start = entity.get("start_offset", 0)
            end = entity.get("end_offset", 0)
            label = entity.get("label", None)

            if label is None or end <= start:
                continue

            # Slice the entity's surface form out of the document text.
            entity_text = text[start:end]

            # Skip invalid or empty entity spans — some annotations produce
            # zero-length ranges after trimming.
            if not entity_text.strip():
                continue

            examples.append({
                "document_id": record.get("id"),
                "text": text,
                "entity": entity_text,
                "label": label,
                "start_offset": start,
                "end_offset": end,
            })

    return examples


def create_model_input(example, context_window=150):
    """
    Create a focused text representation around the target entity.

    Produces a string of the form "<before> [ENTITY] <entity> [/ENTITY] <after>"
    so a token-based model can attend to the entity while still seeing its
    surrounding context. The window size controls how much context is kept on
    each side.

    Args:
        example (dict): One entry produced by extract_entity_examples.
        context_window (int): Characters of context to keep on each side.

    Returns:
        str: Whitespace-normalised context string with entity markers.
    """
    text = example["text"]
    start = example["start_offset"]
    end = example["end_offset"]

    # Clamp the window so we never index outside the original text.
    context_start = max(0, start - context_window)
    context_end = min(len(text), end + context_window)

    before = text[context_start:start]
    entity = text[start:end]
    after = text[end:context_end]

    model_input = f"{before} [ENTITY] {entity} [/ENTITY] {after}"

    # Collapse runs of whitespace so the vectorizer sees a clean string.
    return re.sub(r"\s+", " ", model_input).strip()


def create_label_encoder(train_examples):
    """
    Create and fit a LabelEncoder using only the training data.

    Fitting on training data alone prevents label-class information from
    leaking across splits.

    Args:
        train_examples (list[dict]): Entity examples from extract_entity_examples.

    Returns:
        sklearn.preprocessing.LabelEncoder: Fitted encoder.
    """
    labels = [example["label"] for example in train_examples]
    return LabelEncoder().fit(labels)


def encode_labels(examples, encoder):
    """
    Convert string labels into numerical class IDs.

    Args:
        examples (list[dict]): Entity examples with a "label" key.
        encoder: A fitted LabelEncoder.

    Returns:
        np.ndarray: Integer class IDs, one per example.
    """
    labels = [example["label"] for example in examples]
    return encoder.transform(labels)


def prepare_text_data(examples):
    """
    Create model input text from entity examples.

    Args:
        examples (list[dict]): Entity examples from extract_entity_examples.

    Returns:
        list[str]: One context-window string per example.
    """
    return [create_model_input(example) for example in examples]


def prepare_dataset(records, encoder):
    """
    Convert raw records into model texts and encoded labels.

    End-to-end helper for the entity-level pipeline.

    Args:
        records (list[dict]): Raw CSV or JSONL records.
        encoder: A fitted LabelEncoder.

    Returns:
        tuple: (examples, texts, labels).
    """
    examples = extract_entity_examples(records)
    texts = prepare_text_data(examples)
    labels = encode_labels(examples, encoder)
    return examples, texts, labels


# ---------------------------------------------------------------------------
# Document-level utilities (kept for compatibility)
# ---------------------------------------------------------------------------

def extract_document_label(record):
    """
    Return the first available target label from a CSV or JSONL record.

    A document with no annotation is returned as ``None`` rather than being
    silently relabelled as "benign" — this keeps the pipeline honest about
    unlabeled examples.

    Args:
        record (dict): One row from the CSV or JSONL source.

    Returns:
        str | None: The first entity's label, or None if unavailable.
    """
    # Prefer "Entries" (CSV) but fall back to "entities" (JSONL).
    entries = record.get("Entries", record.get("entities", []))

    # CSV stores the list as a JSON-encoded string — decode it if needed.
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
    """
    Create text/label arrays for document-level classification.

    Note: this is the document-level path. The active training pipeline uses
    entity-level extraction (extract_entity_examples) instead.

    Args:
        records (list[dict]): Raw CSV or JSONL rows.
        encoder: A fitted LabelEncoder, or None if fit_encoder=True.
        fit_encoder (bool): If True, fit a new encoder on this data.

    Returns:
        tuple: (texts, labels, ids, encoder)
    """
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

    return (
        [texts[i] for i in keep],
        encoder.transform([labels[i] for i in keep]),
        [ids[i] for i in keep],
        encoder,
    )


def save_label_encoder(encoder, path="models/label_encoder.pkl"):
    """
    Persist a fitted LabelEncoder to disk for reuse at inference time.

    Args:
        encoder: A fitted LabelEncoder.
        path (str | Path): Destination pickle path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as output:
        pickle.dump(encoder, output)


def calculate_class_weights(labels):
    """
    Calculate balanced class weights from the training labels.

    Balanced class weights scale the loss contribution of each class inversely
    to its frequency, so rare threat types are not drowned out by dominant
    classes during training.

    Args:
        labels (array-like): Integer class IDs from the training split.

    Returns:
        dict: Mapping of {class_id (int): weight (float)}.
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}