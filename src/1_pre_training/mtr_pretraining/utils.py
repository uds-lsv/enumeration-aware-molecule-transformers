import json
from dataclasses import dataclass

from bert_for_regression import BertForRegression
from data import RegressionTextDataset
from data_collators import multitask_data_collator
from transformers import BertTokenizer, Trainer


@dataclass
class DatasetArguments:
    dataset_path: str
    normalization_path: str
    eval_path: str
    tokenizer_path: str
    tokenizer_max_length: int
    mlm_probability: float


def create_trainer(
    config,
    training_args,
    dataset_args: DatasetArguments,
    pretrained_model=None,
):
    tokenizer = BertTokenizer.from_pretrained(
        dataset_args.tokenizer_path,
    )
    dataset_class = RegressionTextDataset
    dataset = dataset_class(
        tokenizer=tokenizer,
        file_path=dataset_args.dataset_path,
        block_size=dataset_args.tokenizer_max_length,
    )

    with open(dataset_args.normalization_path) as f:
        normalization_values = json.load(f)

    config.num_labels = dataset.num_labels
    config.norm_mean = normalization_values["mean"]
    config.norm_std = normalization_values["std"]
    model = BertForRegression

    data_collator = multitask_data_collator

    if pretrained_model:
        model = model.from_pretrained(pretrained_model, config=config, use_auth_token=True)
    else:
        model = model(config=config)

    train_dataset, eval_dataset = get_dataset_splits(dataset_args, tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def get_dataset_splits(dataset_args: DatasetArguments, tokenizer):
    train_dataset = RegressionTextDataset(
        tokenizer=tokenizer,
        file_path=dataset_args.dataset_path,
        block_size=dataset_args.tokenizer_max_length,
    )
    eval_dataset = RegressionTextDataset(
        tokenizer=tokenizer,
        file_path=dataset_args.eval_path,
        block_size=dataset_args.tokenizer_max_length,
    )

    return train_dataset, eval_dataset
