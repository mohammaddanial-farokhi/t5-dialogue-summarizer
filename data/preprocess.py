from datasets import DatasetDict
from data.load_dataset import load_model_dataset
from models.base_model import load_base_model
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
CACHE_DIR = CURRENT_DIR / "cached_tokenized_dataset"


def prepare_tokenized_dataset(force_recompute=False):

    if not force_recompute and os.path.exists(CACHE_DIR):
        print("Loading tokenized dataset from cache...")
        tokenized_datasets = DatasetDict.load_from_disk(CACHE_DIR)
        _, tokenizer = load_base_model()

        print(f"Training shape: {tokenized_datasets['train'].shape}")
        print(f"Validation shape: {tokenized_datasets['validation'].shape}")
        print(f"Test shape: {tokenized_datasets['test'].shape}")

        return tokenized_datasets, tokenizer

    print("Tokenizing dataset from scratch...")
    dataset = load_model_dataset()
    _, tokenizer = load_base_model()

    def tokenize_prompt(example):
        start_prompt = "Summarize the following conversation.\n\n"
        end_prompt = "\n\nSummary: "
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example["input_ids"] = tokenizer(
            prompt, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        ).input_ids
        example["labels"] = tokenizer(
            example["summary"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        ).input_ids
        return example

    tokenized_datasets = dataset.map(tokenize_prompt, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["id", "topic", "dialogue", "summary"])

    tokenized_datasets.save_to_disk(CACHE_DIR)
    print(f"Tokenized dataset saved to {CACHE_DIR}")

    return tokenized_datasets, tokenizer
