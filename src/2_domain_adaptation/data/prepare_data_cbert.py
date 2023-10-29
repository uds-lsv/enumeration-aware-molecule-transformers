import json

import numpy as np
import typer
from data_reader import DataReader
from enumeration import SmilesEnumerator
from loguru import logger
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from tqdm import tqdm
from typing_extensions import Annotated


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


def create_domain_adaptation_dataset_cbert(
    dataset_name: Annotated[str, typer.Argument()], n_enumerations: Annotated[int, typer.Argument()] = 2
):
    smiles_column = "smiles"
    data_reader = DataReader(dataset_name, smiles_column=smiles_column)
    smiles_enumerator = SmilesEnumerator(canonical=False, enum=True)
    target_dataset_filename = f"{dataset_name}_cbert_domain_adaptation.csv"
    smiles_enumerator.enumerate_smiles_hard_neg(
        data_reader, smiles_column, random_pairs=True, replication_count=n_enumerations
    ).to_csv(target_dataset_filename, index=False, encoding="utf-8-sig")
    logger.info(f"Saved enumerated smiles to {target_dataset_filename} for CBERT domain adaptation")

    physicochemical_fingerprint_extractor = PhysicoChemcialPropertyExtractor()
    physicochemical_fingerprints = [
        physicochemical_fingerprint_extractor.compute_descriptors(smile)
        for smile in tqdm(getattr(data_reader.dataset, smiles_column))
    ]
    logger.info(f"Finished computing physicochemical properties for {len(physicochemical_fingerprints)} molecules")
    physicochemical_fingerprints_np = np.array(physicochemical_fingerprints)

    # Compute mean and std
    mean = np.mean(physicochemical_fingerprints_np, axis=0)
    std = np.std(physicochemical_fingerprints_np, axis=0)

    logger.info(
        f"Computed normalization values (mean and std) of {len(physicochemical_fingerprints[0])}                   "
        f"  physicochemical properties. Number of molecules: {len(physicochemical_fingerprints_np)}"
    )
    normalization_target_file_name = f"{dataset_name}_cbert_domain_adaptation_normalization.json"
    # dump the output as json to be used for in pre-training
    with open(normalization_target_file_name, "w") as normalization_file:
        json.dump({"mean": list(mean), "std": list(std)}, normalization_file)
    logger.info(f"Saved normalization values to {normalization_target_file_name}")


if __name__ == "__main__":
    typer.run(create_domain_adaptation_dataset_cbert)
