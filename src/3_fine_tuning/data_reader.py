from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from deepchem.molnet import load_bace_classification, load_muv, load_tox21

DATASET_PATH = str(Path(__file__).parent.resolve() / "data")


@dataclass
class Dataset:
    smiles: list
    y: list


class DataReader:
    """
    This class reads molecular smiles datasets from file system and
    tracks the mappings of relevant depedent and independent features
    """

    def __init__(self, dataset_name: str):
        dataset = dataset_name.lower().strip()
        match dataset:
            case "bace":
                train_dataset = load_bace_classification(reload=False)[1][0]
                valid_dataset = load_bace_classification(reload=False)[1][1]
                test_dataset = load_bace_classification(reload=False)[1][2]
                self.train_dataset = Dataset(smiles=train_dataset.ids, y=train_dataset.y)
                self.valid_dataset = Dataset(smiles=valid_dataset.ids, y=valid_dataset.y)
                self.test_dataset = Dataset(smiles=test_dataset.ids, y=test_dataset.y)
            case "bbbp":
                """
                loading bbbp dataset via deepchem throws an error hence, loading from saved CSVs extracted
                via deepchem, splits are `scaffold`
                """

                def load_csv(file_path) -> Dataset:
                    df = pd.read_csv(file_path)
                    return Dataset(smiles=df["smiles"].tolist(), y=df["p_np"].tolist())

                self.train_dataset = load_csv(f"{DATASET_PATH}/bbbp/train.csv")
                self.valid_dataset = load_csv(f"{DATASET_PATH}/bbbp/valid.csv")
                self.test_dataset = load_csv(f"{DATASET_PATH}/bbbp/test.csv")

            case "muv":
                train_dataset = load_muv()[1][0]
                valid_dataset = load_muv()[1][1]
                test_dataset = load_muv()[1][2]
                self.train_dataset = Dataset(smiles=train_dataset.ids, y=train_dataset.y)
                self.valid_dataset = Dataset(smiles=valid_dataset.ids, y=valid_dataset.y)
                self.test_dataset = Dataset(smiles=test_dataset.ids, y=test_dataset.y)
            case "tox21":
                train_dataset = load_tox21(reload=False)[1][0]
                valid_dataset = load_tox21(reload=False)[1][1]
                test_dataset = load_tox21(reload=False)[1][2]
                self.train_dataset = Dataset(smiles=train_dataset.ids, y=train_dataset.y)
                self.valid_dataset = Dataset(smiles=valid_dataset.ids, y=valid_dataset.y)
                self.test_dataset = Dataset(smiles=test_dataset.ids, y=test_dataset.y)
            case "clintox":

                def load_csv(file_path) -> Dataset:
                    tasks = ["FDA_APPROVED", "CT_TOX"]
                    df = pd.read_csv(file_path)
                    return Dataset(smiles=df["smiles"].tolist(), y=df[tasks].values.tolist())

                self.train_dataset = load_csv(f"{DATASET_PATH}/clintox/train.csv")
                self.valid_dataset = load_csv(f"{DATASET_PATH}/clintox/valid.csv")
                self.test_dataset = load_csv(f"{DATASET_PATH}/clintox/test.csv")
            case _:
                raise ValueError(f"Dataset {dataset_name} not supported")
