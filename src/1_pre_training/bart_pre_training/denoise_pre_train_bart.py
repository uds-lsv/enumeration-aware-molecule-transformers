import logging
import math
from pathlib import Path
import random
import os

import datasets
from datasets import Dataset, DatasetDict
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
    BartTokenizer
)
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args

DATASETS_PATH = str(Path(__file__).parent.parent.resolve() / "data")

logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "guacamol_data": ("source", "target"),
}


def main():  # sourcery skip: invert-any-all
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    raw_datasets = DatasetDict(**{
            "train":  Dataset.from_pandas(pd.read_json(f"{DATASETS_PATH}/guacamol_v1_train_guacomol.json")),
            "validation": Dataset.from_pandas(pd.read_json(f"{DATASETS_PATH}/guacamol_v1_valid_guacamol.json")),
        })

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=".ckpt" in args.model_name_or_path,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    #  Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name)
    print(f"dataset_columns: {dataset_columns}")
    if args.text_column is None:
        text_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # This function is copied from modeling_bart.py
    def shift_tokens_right(input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def add_noise(input_ids):
        if args.masking_noise:
            input_ids = add_masks(input_ids)
        if args.insertion_noise:
            input_ids = add_insertion_noise(input_ids)
        if args.rolling_noise:
            input_ids = add_rolling_noise(input_ids)

        return input_ids

    def add_masks(encoded_texts):
        # 30% BART masking
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        # Do not mask special tokens
        inp_mask[encoded_texts <= 4] = False

        # Prepare input
        encoded_texts_masked = np.copy(encoded_texts)
        # Set input to [MASK] which is the last token for the 90% of tokens
        # This means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
        encoded_texts_masked[
            inp_mask_2mask
        ] = tokenizer.mask_token_id  # mask token is the last in the dict

        for idx, row in enumerate(encoded_texts_masked):
            tmp = np.array([tokenizer.pad_token_id] * len(row))
            dupIdx = np.r_[[0], np.nonzero(np.diff((row-1)+(row == 4).astype(int)))[0] + 1]
            tmp[:len(dupIdx)] = row[dupIdx]
            encoded_texts_masked[idx, :] = tmp
        return encoded_texts_masked

    def add_rolling_noise(input_ids):
        input_ids = torch.Tensor(input_ids)
        offset = torch.randint(1, max(1, input_ids.size(-1) - 1) + 1, (input_ids.size(0),))

        output_ids = torch.full_like(input_ids, fill_value=tokenizer.pad_token_id)
        for seq_idx in range(output_ids.size(0)):
            output_ids[seq_idx] = torch.cat(
                (
                    input_ids[seq_idx, 0:1],
                    input_ids[seq_idx, offset[seq_idx]:-1],
                    input_ids[seq_idx, 1:offset[seq_idx]],
                    input_ids[seq_idx, -1:],
                ),
                dim=0,
            )
        return output_ids.long()

    def add_insertion_noise(input_ids, p=0.20, random_ratio=0.1):
        input_ids = torch.LongTensor(input_ids)
        """As currently implemented, all sequences in this batch will get the same number of added noise (num_noise).
        In fairseq, this is implemented on the dataset-level, meaning that every sequence in a batch may have a
        different number of added noise.
        In addition, because we are now in the data collator, we have to already account for sequence length. In
        Fairseq, the dataset can output any longer length, which can be truncated later. Here, we have to truncate
        directly at the end, after inserting noise. This means, however, that it is possible that a sequence does not
        end with </s>.
        """

        if p == 0.0:
            return input_ids

        seq_num_tokens = input_ids.size(1)
        num_noise = int(math.ceil(seq_num_tokens * p))
        all_results = torch.full(
            (input_ids.size(0), seq_num_tokens + num_noise), fill_value=tokenizer.pad_token_id
        )
        for seq_id, sequence in enumerate(input_ids):
            # -2 and + 1 to avoid targetting first and last item?
            noise_indices = torch.randperm(seq_num_tokens + num_noise - 2)[:num_noise] + 1
            noise_mask = torch.zeros(size=(seq_num_tokens + num_noise,), dtype=torch.bool)
            noise_mask[noise_indices] = 1

            result = torch.LongTensor(seq_num_tokens + num_noise).fill_(-1)

            num_random = int(math.ceil(num_noise * random_ratio))
            result[noise_indices[num_random:]] = tokenizer.mask_token_id
            result[noise_indices[:num_random]] = torch.randint(low=1, high=len(tokenizer), size=(num_random,))

            result[~noise_mask] = sequence

            assert (result >= 0).all()
            all_results[seq_id] = result

        all_results = all_results[:, :seq_num_tokens]
        return all_results.long()

    def preprocess_function_modified(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = tokenizer.prepare_seq2seq_batch(
            src_texts=inputs,
            tgt_texts=targets,
            max_length=args.max_length,
            return_tensors='pt',
            padding="max_length"
        )

        decoder_inputs = shift_tokens_right(inputs['labels'], tokenizer.pad_token_id)
        inputs["decoder_input_ids"] = decoder_inputs.tolist()
        inputs["input_ids"] = add_noise(inputs["input_ids"]).tolist()
        inputs["labels"] = inputs["labels"].tolist()
        inputs["attention_mask"] = inputs["attention_mask"].tolist()
        return inputs

    processed_datasets = raw_datasets.map(
        preprocess_function_modified,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        torch.cuda.empty_cache()
        logger.info(f"Starting epoch: {epoch}")
        logger.info("Starting training...")
        model.train()
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        # break
        torch.cuda.empty_cache()
        logger.info("Starting validation...")
        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length
            if args is not None
            else config.max_length,
            "num_beams": args.num_beams,
        }
        for step, batch in enumerate(tqdm1(eval_dataloader)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                    )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            if step > args.max_train_steps:
                break
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)
        torch.cuda.empty_cache()
        try:
            if args.output_dir is not None:
                save_path = os.path.join(args.output_dir, f"model_{epoch}")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                logger.info(f"Saving model at epoch {epoch}...")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    save_path, save_function=accelerator.save
                )
                tokenizer.save_pretrained(save_path)
        except Exception:
            logger.info("Error saving model.")


if __name__ == "__main__":
    main()
