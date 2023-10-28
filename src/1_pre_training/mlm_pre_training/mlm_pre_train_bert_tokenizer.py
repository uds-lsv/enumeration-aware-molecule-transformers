import pandas as pd
from tqdm import tqdm
import time
from transformers import (
    BertTokenizer,
    BertConfig,
    LineByLineTextDataset,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
import os


TOKENIZER_PATH = "./smole-bert"
VOCAB_SIZE = 4096


def train_tokenizer(dataset_path, tokenizer_name="simo-bert"):
    # paths = [str(x) for x in Path(".").glob("**/*.txt")]
    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )
    # Customize training
    tokenizer.train(
        files=dataset_path,
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        limit_alphabet=1000, wordpieces_prefix='##',
        special_tokens=[
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    )

    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    tokenizer.save_model(TOKENIZER_PATH, 'simo-bert')


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
        num_labels=3,
    )
    model = BertForMaskedLM(config)
    return model


## load model and tokenizer
#train_tokenizer("./guacamol_v1_train.txt")
tokenizer = BertTokenizer(
 "./smole-bert/simo-bert-vocab.txt"
)

#tokenizer = BertTokenizer(tokenizer)#.from_pretrained("/smole-bert/")
# model = BertForMaskedLM.from_pretrained("shahrukhx01/chemical-bert-uncased")
model = get_init_model()


## load dataset
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./guacamol_v1_train_mini.txt",
    block_size=128,
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./guacamol_v1_valid.txt",
    block_size=128,
)

## create data collator for Mask Language Modelling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

## training arguments for pretraining
training_args = TrainingArguments(
    output_dir="./smole-bert",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

## create PyTorch Lightning trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    progress_bar_refresh_rate=25,
    val_check_interval=1,
)

## pretrain the model
trainer.train()


## save model to disk once training is done
trainer.save_model("./smole-bert")
tokenizer.save_pretrained("./smole-bert")
