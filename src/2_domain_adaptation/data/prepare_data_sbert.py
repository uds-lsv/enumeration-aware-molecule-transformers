import pandas as pd
import typer
from data_reader import DataReader
from enumeration import SmilesEnumerator
from loguru import logger
from typing_extensions import Annotated


def perform_augmentation(dataset_name: Annotated[str, typer.Argument()]):
    smiles_column = "smiles"
    data_reader = DataReader(dataset_name, smiles_column=smiles_column)
    smiles_enumerator = SmilesEnumerator(canonical=False, enum=True)
    (
        enumerated_smiles,
        original_smiles,
        is_enumerated,
    ) = smiles_enumerator.enumerate_smiles(data_reader, smiles_column, random_pairs=True)
    enumerated_smiles_df = pd.DataFrame(
        {
            "smiles_a": original_smiles,
            "smiles_b": enumerated_smiles,
            "is_enumerated": is_enumerated,
        }
    )
    logger.info(f"Enumerated smiles: {len(enumerated_smiles_df)}")
    target_dataset_filename = f"{dataset_name}_sbert_domain_adaptation.csv"
    enumerated_smiles_df.to_csv(
        target_dataset_filename,
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(f"Saved enumerated smiles to {target_dataset_filename} for SBERT domain adaptation")


if __name__ == "__main__":
    typer.run(perform_augmentation)
