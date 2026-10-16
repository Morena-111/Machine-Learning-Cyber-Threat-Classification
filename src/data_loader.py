import json
from pathlib import Path

import pandas as pd


def load_jsonl(file_path):
    """Load a JSONL file and return a list of records."""
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_datasets(data_dir="data"):
    """Load the training, validation, and test datasets."""

    # Prefer the rubric's CSV names when supplied, but keep compatibility with
    # the repository's existing JSONL dataset.
    data_path = Path(data_dir)
    csv_names = {
        "train": "cyber-threat-intelligence-splited_train.csv",
        "validation": "cyber-threat-intelligence-splited_validate.csv",
        "test": "cyber-threat-intelligence-splited_test.csv",
    }
    if all((data_path / name).exists() for name in csv_names.values()):
        return tuple(pd.read_csv(data_path / name).to_dict("records") for name in csv_names.values())

    train_data = load_jsonl(data_path / "train.jsonl")
    validation_data = load_jsonl(data_path / "validation.jsonl")
    test_data = load_jsonl(data_path / "test.jsonl")

    return train_data, validation_data, test_data
