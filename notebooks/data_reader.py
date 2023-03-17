import pandas as pd


class DataReader:
    """
    This class reads molecular smiles datasets from file system and
    tracks the mappings of relevant depedent and independent features
    """

    def __init__(self, file_path, smiles_column, sample_size=-1, labels_column = None):
        if sample_size > -1:
            self.dataset = pd.read_csv(file_path).sample(sample_size)
        else:
            self.dataset = pd.read_csv(file_path)
        if labels_column:
            self.dataset = self.dataset[self.dataset[labels_column].notna()]
        self.smiles_column = smiles_column
