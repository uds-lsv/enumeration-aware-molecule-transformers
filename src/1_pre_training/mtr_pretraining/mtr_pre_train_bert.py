from pathlib import Path

import torch
from transformers import BertConfig, TrainingArguments
from utils import DatasetArguments, create_trainer

# change the following configuration vars to your own path, if necessary
DATASET_PATH = str(Path(__file__).parent.parent.resolve() / "data")
NORMALIZATION_PATH = "normalization_values.json"
TRAIN_MTR_FILENAME = "guacamol_v1_train_mtr.jsonl"
VALID_MTR_FILENAME = "guacamol_v1_valid_mtr.jsonl"
MOLDEL_NAME = "smole-bert-mtr"
VOCAB_SIZE = 4096
MODEL_AND_TOKENIZER_PATH = str(Path(__file__).parent.parent.resolve() / MOLDEL_NAME)


if __name__ == "__main__":
    torch.manual_seed(0)

    model_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=768,
        intermediate_size=2048,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        is_gpu=torch.cuda.is_available(),
    )

    dataset_args = DatasetArguments(
        dataset_path=f"{DATASET_PATH}/{TRAIN_MTR_FILENAME}",
        normalization_path=f"{DATASET_PATH}/{NORMALIZATION_PATH}",
        eval_path=f"{DATASET_PATH}/{VALID_MTR_FILENAME}",
        tokenizer_path=f"{MODEL_AND_TOKENIZER_PATH}/{MOLDEL_NAME}-vocab.txt",
        tokenizer_max_length=128,
        mlm_probability=0.15,
    )
    training_args = TrainingArguments(
        output_dir=MODEL_AND_TOKENIZER_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # fp16 only works on CUDA devices
    )
    trainer = create_trainer(model_config, training_args, dataset_args)
    trainer.train()

    # save model post training
    trainer.save_model(MODEL_AND_TOKENIZER_PATH)
    trainer.model.save_pretrained(MODEL_AND_TOKENIZER_PATH)
