# Pre-training Molecular BERT Model with MLM or MTR
1. Firstly, install the required dependencies using `pip install -r requirements.txt`, you can find the `requirements.txt` file [here](../../requirements.txt).
2. Download the Guacamol dataset using the script found in `src/1_pre_training/data/guacamol_dataset_downloader.py` using:
```bash
python guacamol_dataset_downloader.py
```
3. Subsequently train the tokenizer using the script found in `src/1_pre_training/train_bert_tokenizer.py` after adjusting the `MODEL_NAME` variable in the script and then execute:
```bash
python train_bert_tokenizer.py
```

To reproduce our pre-training step please follow the steps below:

## MLM Pre-training
- Make sure you have adjusted tokenizer name and trained it using the instructions above.
- Finally start pre-training the BERT model using the script in `src/1_pre_training/mlm_pre_training/mlm_pre_train_bert.py` using:
```bash
python mlm_pre_train_bert.py
```
## MTR Pre-training
- Make sure you have adjusted tokenizer name and trained it using the instructions above.
- In addition to downloading and training the tokenizer, for MTR you also need to preprare physicochemical properties as labels and compute normalization values (mean and std) on the extracted properties. You can achieve this by executing the script `src/1_pre_training/data/prepare_mtr_dataset.py`
```bash
python prepare_mtr_dataset.py
```
- Then, start pre-training the BERT model with Multi-task Regression objective using the script in `src/1_pre_training/mtr_pretraining/mtr_pre_train_bert.py` using:
```bash
python mtr_pre_train_bert.py
```
## Final Pre-trained Model
At the end training you can located the pre-trained model for MLM and MTR pre-training in `src/1_pre_training/smole-bert` and/or `src/1_pre_training/smole-bert-mtr` respectively for the given pre-training objective.
