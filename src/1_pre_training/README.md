# Pre-training Molecular BERT Model with MLM or MTR
To reproduce our pre-training step please follow the steps below:
## MLM Pre-training
1. Install the required dependencies using `pip install -r requirements.txt`, you can find the `requirements.txt` file [here](../../requirements.txt).
2. Download the Guacamol dataset using the script found in `src/1_pre_training/data/guacamol_dataset_downloader.py` using:
```bash
python guacamol_dataset_downloader.py
```
3. Subsequently train the tokenizer using the script found in `src/1_pre_training/train_bert_tokenizer.py` using:
```bash
python train_bert_tokenizer.py
```
4. Finally start pre-training the BERT model using the script in `src/1_pre_training/mlm_pre_training/mlm_pre_train_bert.py` using:
```bash
python mlm_pre_train_bert.py
```
