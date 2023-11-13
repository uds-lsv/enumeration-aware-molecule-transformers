# Download Guacamol dataset from Figshare
from urllib.request import urlretrieve

from loguru import logger
from tqdm import tqdm

# Guacamol dataset download Urls
dataset_name = "guacamol_v1"
train_url = "https://huggingface.co/shahrukhx01/guacamol-bart-dataset/resolve/main/guacamol_v1_train.json?download=true"
valid_url = "https://huggingface.co/shahrukhx01/guacamol-bart-dataset/raw/main/guacamol_v1_valid.json"


if __name__ == "__main__":
    for url, split_name in tqdm([(train_url, "train_guacomol.json"), (valid_url, "valid_guacamol.json")]):
        logger.info(f"Downloading {split_name} from {url}")
        urlretrieve(url, f"{dataset_name}_{split_name}")
