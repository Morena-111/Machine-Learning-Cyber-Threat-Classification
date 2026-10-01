import json


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

    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    validation_data = load_jsonl(f"{data_dir}/validation.jsonl")
    test_data = load_jsonl(f"{data_dir}/test.jsonl")

    return train_data, validation_data, test_data