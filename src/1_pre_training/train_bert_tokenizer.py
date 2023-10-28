import os
from pathlib import Path

from tokenizers import BertWordPieceTokenizer

# Model and tokenizer paths
MODEL_NAME = "smole-bert"
TOKENIZER_PATH = str(Path(__file__).parent.resolve() / MODEL_NAME)
VOCAB_SIZE = 4096
DATASET_PATH = str(Path(__file__).parent.resolve() / "data" / "guacamol_v1_train.smiles")


def train_tokenizer(dataset_path):
    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False
    )
    # Customize training
    tokenizer.train(
        files=dataset_path,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    tokenizer.save_model(TOKENIZER_PATH, MODEL_NAME)


if __name__ == "__main__":
    train_tokenizer(dataset_path=DATASET_PATH)
