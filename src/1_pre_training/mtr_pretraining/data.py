import torch
from loguru import logger
from nlp import load_dataset
from torch.utils.data import Dataset


def preprocess(line, tokenizer, block_size):
    def _clean_property(x):
        return 0.0 if x == "" or "inf" in str(x) else float(x)

    smiles = line["smile"]
    labels = line["labels"]

    batch_encoding = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )
    batch_encoding["label"] = [_clean_property(x) for x in labels]
    batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}

    return batch_encoding


class RegressionTextDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        logger.info("Initializing dataset...")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        logger.info("Inferring JSONL structure from first line...")
        self.dataset = load_dataset("json", data_files=file_path)["train"]
        self.num_labels = len(self.dataset[0]["labels"])
        logger.info("Loaded Dataset")
        self.len = len(self.dataset)
        logger.info(f"Number of lines: {self.len}")
        logger.info(f"Block size: {self.block_size}")

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return preprocess(self.dataset[i], self.tokenizer, self.block_size)
