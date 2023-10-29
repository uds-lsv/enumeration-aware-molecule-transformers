# Domain Adaptation with SBERT and CBERT

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
