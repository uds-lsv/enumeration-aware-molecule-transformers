from dataclasses import dataclass

import pandas as pd
from deepchem.molnet import load_bace_classification, load_muv, load_tox21


@dataclass
class Dataset:
    smiles: list
    y: list


class DataReader:
    """
    This class reads molecular smiles datasets from file system and
    tracks the mappings of relevant depedent and independent features
    """

    def __init__(self, dataset_name: str, smiles_column: str, sample_size: int = -1):
        dataset = dataset_name.lower().strip()
        match dataset:
            case "bace":
                _dataset = load_bace_classification(reload=True)[1][0]
                self.dataset = Dataset(smiles=_dataset.ids, y=_dataset.y)
            case "bbbp":
                """
                loading bbbp dataset via deepchem throws an error hence, loading from saved CSVs extracted
                via deepchem, splits are `scaffold`
                """

                def load_csv(file_path) -> Dataset:
                    df = pd.read_csv(file_path)
                    return Dataset(smiles=df["smiles"].tolist(), y=df["p_np"].tolist())

                self.dataset = load_csv("bbbp/train.csv")

            case "muv":
                _dataset = load_muv()[1][0]
                self.dataset = Dataset(smiles=_dataset.ids, y=_dataset.y)
            case "tox21":
                _dataset = load_tox21()[1][0]
                self.dataset = Dataset(smiles=_dataset.ids, y=_dataset.y)
            case "clintox":

                def load_csv(file_path) -> Dataset:
                    tasks = ["FDA_APPROVED", "CT_TOX"]
                    df = pd.read_csv(file_path)
                    return Dataset(smiles=df["smiles"].tolist(), y=df[tasks].values.tolist())

                self.dataset = load_csv("clintox/train.csv")
            case _:
                raise ValueError(f"Dataset {dataset_name} not supported")
