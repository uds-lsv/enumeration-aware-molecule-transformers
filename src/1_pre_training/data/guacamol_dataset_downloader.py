# Download Guacamol dataset from Figshare
from urllib.request import urlretrieve

from loguru import logger
from tqdm import tqdm

# Guacamol dataset download Urls
dataset_name = "guacamol_v1"
train_url = "https://ndownloader.figshare.com/files/13612760"
valid_url = "https://ndownloader.figshare.com/files/13612766"
# uncomment if you need test set
# test_url = "https://ndownloader.figshare.com/files/13612757"


if __name__ == "__main__":
    for url, split_name in tqdm([(train_url, "train.smiles"), (valid_url, "valid.smiles")]):
        logger.info(f"Downloading {split_name} from {url}")
        urlretrieve(url, f"{dataset_name}_{split_name}")
