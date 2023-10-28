from pathlib import Path

from transformers import DataCollatorForLanguageModeling  # noqa
from transformers import Trainer  # noqa
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, LineByLineTextDataset, TrainingArguments  # noqa

MOLDEL_NAME = "smole-bert"
VOCAB_SIZE = 4096
MODEL_AND_TOKENIZER_PATH = str(Path(__file__).parent.parent.resolve() / MOLDEL_NAME)
DATASET_PATH = str(Path(__file__).parent.parent.resolve() / "data")
TRAIN_DATASET_FILENAME = "guacamol_v1_train.smiles"
VALID_DATASET_FILENAME = "guacamol_v1_valid.smiles"


def get_init_model():
    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=768,
        intermediate_size=2048,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    return BertForMaskedLM(config)


# load model and tokenizer
tokenizer = BertTokenizer(f"{MODEL_AND_TOKENIZER_PATH}/{MOLDEL_NAME}-vocab.txt")
model = get_init_model()


# load dataset
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{DATASET_PATH}/{TRAIN_DATASET_FILENAME}",
    block_size=128,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{DATASET_PATH}/{VALID_DATASET_FILENAME}",
    block_size=128,
)

# create data collator for Mask Language Modelling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# training arguments for pretraining
training_args = TrainingArguments(
    output_dir=MODEL_AND_TOKENIZER_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

# create PyTorch Lightning trainer
trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=val_dataset
)

# pretrain the model
trainer.train()


# save model to disk once training is done
trainer.save_model(MODEL_AND_TOKENIZER_PATH)
tokenizer.save_pretrained(MODEL_AND_TOKENIZER_PATH)
