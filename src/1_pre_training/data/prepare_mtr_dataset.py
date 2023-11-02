import json

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from tqdm import tqdm

TRAIN_FILENAME = "guacamol_v1_train.smiles"
VALID_FILENAME = "guacamol_v1_valid.smiles"
# Path to normalization values (mean, std) for physicochemical properties
NORMALIZATION_PATH = "normalization_values.json"
# Path to the labeled dataset
TRAIN_MTR_FILENAME = "guacamol_v1_train_mtr.jsonl"
VALID_MTR_FILENAME = "guacamol_v1_valid_mtr.jsonl"


class PhysicoChemcialPropertyExtractor:
    """Computes RDKit properties on-the-fly."""

    def __init__(self):
        super().__init__()

        self.descriptors = [name for name, _ in Chem.Descriptors.descList]
        self.descriptors.remove("Ipc")
        self.calculator = MolecularDescriptorCalculator(self.descriptors)
        self.num_labels = len(self.descriptors)
        logger.info(f"Number of physicochemical properties: {self.num_labels}")

    def __len__(self):
        return self.len

    def compute_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_descriptors = np.full(shape=(self.num_labels), fill_value=0.0)
        else:
            mol_descriptors = np.array(list(self.calculator.CalcDescriptors(mol)))
            mol_descriptors = np.nan_to_num(mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        assert mol_descriptors.size == self.num_labels

        return mol_descriptors


def extract_physicochemical_props(dataset_path, mtr_path, normalization_path=None):
    """Extracts physicochemical properties from the dataset and saves them to a JSONL file.

    Args:
        dataset_path (str): Path to the dataset.
        mtr_path (str): Path to save the labeled dataset.
        normalization_path (str, optional): Path to save the normalization values (mean and std). Defaults to None.
    """
    with open(dataset_path) as train_dataset_file:
        smiles = train_dataset_file.readlines()
    assert len(smiles), "No smiles found in dataset"

    physicochemical_fingerprint_extractor = PhysicoChemcialPropertyExtractor()
    physicochemical_fingerprints = [
        physicochemical_fingerprint_extractor.compute_descriptors(smile) for smile in tqdm(smiles)
    ]
    logger.info(f"Finished computing physicochemical properties for {len(physicochemical_fingerprints)} molecules")
    physicochemical_fingerprints_np = np.array(physicochemical_fingerprints)

    if normalization_path:
        # Compute mean and std
        mean = np.mean(physicochemical_fingerprints_np, axis=0)
        std = np.std(physicochemical_fingerprints_np, axis=0)

        logger.info(
            f"Computed normalization values (mean and std) of {len(physicochemical_fingerprints[0])}                   "
            f"  physicochemical properties. Number of molecules: {len(physicochemical_fingerprints_np)}"
        )

        # dump the output as json to be used for in pre-training
        with open(normalization_path, "w") as normalization_file:
            json.dump({"mean": list(mean), "std": list(std)}, normalization_file)
        logger.info(f"Saved normalization values to {normalization_path}")

    with open(mtr_path, "w") as labeled_dataset_file:
        for smile, physicochemical_property in zip(smiles, physicochemical_fingerprints):
            json.dump({"smile": smile, "labels": physicochemical_property.tolist()}, labeled_dataset_file)
            labeled_dataset_file.write("\n")
    logger.info(f"Saved labeled dataset to {mtr_path}")


if __name__ == "__main__":
    logger.info("Extracting physicochemical properties from the Train dataset.")
    extract_physicochemical_props(TRAIN_FILENAME, TRAIN_MTR_FILENAME, normalization_path=NORMALIZATION_PATH)

    logger.info("Extracting physicochemical properties from the Validation dataset.")
    extract_physicochemical_props(VALID_FILENAME, VALID_MTR_FILENAME, normalization_path=None)
