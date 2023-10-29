# Domain Adaptation with SBERT and CBERT
In this document we layout the critical steps to perform molecular domain adaptation for both CBERT and SBERT architectures.
## Domain Adaptation with SBERT

1. Assuming that you have already installed all the required dependencies and pretrained a molecular language model. If not please see [this guide](../1_pre_training/README.md).
2. Now for SBERT domain adaptation you'd need to prepare the domain adaptation dataset from the MoleculeNet benchmark among `("muv", "bace", "bbbp", "tox21", "clintox")`. You can prepare the dataset by running the the `src/2_domain_adaptation/data/prepare_data_sbert.py` script:
```bash
# for BBBP
python prepare_data_sbert.py bbbp

# for Clintox
python prepare_data_sbert.py clintox

# for Tox21
python prepare_data_sbert.py tox21

# for Bace
python prepare_data_sbert.py bace

# for MUV
python prepare_data_sbert.py muv
```
3. Now run the domain adaptation script using `src/2_domain_adaptation/sbert/sbert_domain_adaptation.py`:
```bash
# for BBBP
python sbert_domain_adaptation.py bbbp <PATH_TO_PRETRAINED_MODEL_DIRECTORY> <NUM_EPOCHS> <BATCH_SIZE>

# for Clintox
python sbert_domain_adaptation.py clintox <PATH_TO_PRETRAINED_MODEL_DIRECTORY> <NUM_EPOCHS> <BATCH_SIZE>

# for Tox21
python sbert_domain_adaptation.py tox21 <PATH_TO_PRETRAINED_MODEL_DIRECTORY> <NUM_EPOCHS> <BATCH_SIZE>

# for Bace
python sbert_domain_adaptation.py bace <PATH_TO_PRETRAINED_MODEL_DIRECTORY> <NUM_EPOCHS> <BATCH_SIZE>

# for MUV
python sbert_domain_adaptation.py muv <PATH_TO_PRETRAINED_MODEL_DIRECTORY> <NUM_EPOCHS> <BATCH_SIZE>

```

## Domain Adaptation with CBERT

1. For CBERT domain adaptation you'd need to prepare the domain adaptation dataset from the MoleculeNet benchmark among `("muv", "bace", "bbbp", "tox21", "clintox")`. You can prepare the dataset by running the the `src/2_domain_adaptation/data/prepare_data_cbert.py` script:
```bash
# for BBBP
python prepare_data_cbert.py bbbp

# for Clintox
python prepare_data_cbert.py clintox

# for Tox21
python prepare_data_cbert.py tox21

# for Bace
python prepare_data_cbert.py bace

# for MUV
python prepare_data_cbert.py muv
```
2. For performing the domain adaptation step you can the following command, for instance on the BBBP dataset. Please make sure to update the option params corresponding to dataset and model name or path before running the following step. Run the script `src/2_domain_adaptation/cbert/train.py` as follows:
```bash
python train.py \
--model_name_or_path shahrukhx01/smole-bert \
--train_file ../data/bbbp_cbert_domain_adaptation.csv \
--output_dir ../models/cbert_domain_adapted_bbbp \
--normalization_path ../data/bbbp_cbert_domain_adaptation_normalization.json \
--num_train_epochs 10 \
--per_device_train_batch_size 32 \
--learning_rate 3e-5 \
--max_seq_length 128 \
--evaluation_strategy steps \
--metric_for_best_model stsb_spearman \
--load_best_model_at_end \
--eval_steps 500 \
--pooler_type cls \
--mlp_only_train \
--overwrite_output_dir \
--temp 0.05 \
--do_train \
--do_eval \
--do_mtr \
--do_mlm \
--cl_weight 0.33 \
--mlm_weight 0.33 \
--mtr_weight 0.33
```

To disable any of the domain adaptation objective remove the corresponding option flag and set its weight to zero, for instance for disabling MLM you'd remove flag option `--do_mlm` and set `--mlm_weight 0.0` whereby set the other two objective weights to `--mtr_weight 0.50` and `--cl_weight 0.50`

## Final Pre-trained Model
At the end training you can located the domain adapted models for SBERT and CBERT in `src/2_domain_adaptation/models` with naming format `sbert_domain_adapted_{dataset_name}` and `cbert_domain_adapted_{dataset_name}` respectively. For instance for CBERT domain adapted on BBBP dataset the resulting model would be at the path `src/2_domain_adaptation/models/cbert_domain_adapted_bbbp`.
