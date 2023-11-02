import os
from pathlib import Path

import pandas as pd
import typer
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from typing_extensions import Annotated

DATASETS_PATH = str(Path(__file__).parent.parent.resolve() / "data")
MODELS_SAVE_PATH = str(Path(__file__).parent.parent.resolve() / "models")
SUPPORTED_DATASETS = ["muv", "bace", "bbbp", "tox21", "clintox"]


class SentenceBERTDataloader:
    def load_data(self, dataset_df, train_batch_size=64, loss_type="multiple_negs_ranking"):
        # Read train data
        train_samples_MultipleNegativesRankingLoss = []
        train_samples_ConstrativeLoss = []

        for idx, row in dataset_df.iterrows():
            if loss_type == "multiple_negs_ranking":
                train_samples_MultipleNegativesRankingLoss.extend(
                    (
                        InputExample(texts=[row["smiles_a"], row["smiles_b"]], label=1),
                        InputExample(texts=[row["smiles_b"], row["smiles_a"]], label=1),
                    )
                )
            else:
                train_samples_ConstrativeLoss.append(
                    InputExample(
                        texts=[row["smiles_a"], row["smiles_b"]],
                        label=int(row["is_enumerated"]),
                    )
                )
        if loss_type == "multiple_negs_ranking":
            return DataLoader(
                train_samples_MultipleNegativesRankingLoss,
                shuffle=True,
                batch_size=train_batch_size,
            )
        else:
            return DataLoader(
                train_samples_ConstrativeLoss,
                shuffle=True,
                batch_size=train_batch_size,
            )


class SSBERT:
    def __init__(self, model_name_path, model_save_path="sbert_domain_adapted") -> None:
        word_embedding_model = models.Transformer(model_name_path, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Increasing the batch size improves the performance for MultipleNegativesRankingLoss.
        # Choose it as large as possible
        # I achieved the good results with a batch size of 300-350 (requires about 30 GB of GPU memory)
        self.train_batch_size = 64

        # As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
        self.distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

        # Negative pairs should have a distance of at least 0.5
        self.margin = 0.5

        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

    def train(self, num_epochs, dataloaders):
        train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(self.model)

        train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(
            model=self.model, distance_metric=self.distance_metric, margin=self.margin
        )

        # Train the model
        self.model.fit(
            train_objectives=[
                (
                    dataloaders["train_dataloader_mul_negs_ranking"],
                    train_loss_MultipleNegativesRankingLoss,
                ),
                (
                    dataloaders["train_dataloader_contrastive_loss"],
                    train_loss_ConstrativeLoss,
                ),
            ],
            epochs=num_epochs,
            warmup_steps=1000,
            output_path=self.model_save_path,
        )


def train_sbert(
    dataset_name: Annotated[str, typer.Argument()],
    pretrained_model_name_or_path: Annotated[str, typer.Argument()],
    epochs: Annotated[int, typer.Argument()],
    batch_size: Annotated[int, typer.Argument()] = 100,
):
    sbert_dataloader = SentenceBERTDataloader()
    _dataset = dataset_name.lower().strip()
    assert _dataset in SUPPORTED_DATASETS, f"Dataset {_dataset} not supported, please choose from {SUPPORTED_DATASETS}"
    _dataset_path = f"{DATASETS_PATH}/{_dataset}_sbert_domain_adaptation.csv"
    assert os.path.exists(
        _dataset_path
    ), f"Dataset {_dataset_path} not found, please run        `python prepare_data_sbert.py {_dataset}`"
    df = pd.read_csv(_dataset_path)

    # create dataloaders
    train_dataloader_ConstrativeLoss = sbert_dataloader.load_data(
        df, loss_type="constrastive_loss", train_batch_size=batch_size
    )
    train_dataloader_MultipleNegativesRankingLoss = sbert_dataloader.load_data(df, train_batch_size=batch_size)
    dataloaders = {
        "train_dataloader_mul_negs_ranking": train_dataloader_MultipleNegativesRankingLoss,
        "train_dataloader_contrastive_loss": train_dataloader_ConstrativeLoss,
    }

    # train model
    model_save_name = f"sbert_domain_adapted_{dataset_name}"
    model = SSBERT(
        model_name_path=pretrained_model_name_or_path, model_save_path=f"{MODELS_SAVE_PATH}/{model_save_name}"
    )
    model.train(num_epochs=epochs, dataloaders=dataloaders)


if __name__ == "__main__":
    typer.run(train_sbert)
