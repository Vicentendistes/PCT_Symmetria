from pathlib import Path
from typing import Callable, List, Optional

import lightning
from torch.utils.data import DataLoader

from src.dataset.SymDataset import SymDataset
from src.dataset.SymDatasetBatcher import SymDatasetBatcher
from src.dataset.SymDatasetItem import SymDatasetItem
from src.dataset.transforms.AbstractTransform import AbstractTransform
from src.dataset.transforms.IdentityTransform import IdentityTransform


def custom_collate_fn(item_list: List[SymDatasetItem]) -> SymDatasetBatcher:
    return SymDatasetBatcher(item_list)


class SymDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            dataset_path: str = "/path/to/dataset",
            predict_data_path: str = "/path/to/predict_data",
            batch_size: int = 2,
            transform: AbstractTransform = IdentityTransform(),
            shape_excluded: Optional[List[str]] = None,
            perturbation_excluded: Optional[List[str]] = None,
            does_predict_has_ground_truths: bool = False,
            collate_function: Callable = custom_collate_fn,
            shuffle: bool = True,
            n_workers: int = 1,
    ):
        """
        Data module designed to load Shrec2023 symmetry dataset.
        :param dataset_path: Path to dataset, it must contain points and planes split in train/valid/test sets.
        :param predict_data_path: Path to predict it must contain points and can contain planes
        :param does_predict_has_ground_truths : Boolean flag to indicate if the predict data has
        ground truths.
        :param batch_size: Batch size used on all dataloaders.
        :param transform: Transform applied to all dataloaders.
        :param collate_function: Function used to batch the items from symmetry dataset.
        :param shuffle: True if you want to shuffle the train dataloader every epoch.
        :param n_workers: Amount of workers used for loading data into RAM.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.predict_data_path = predict_data_path
        self.does_predict_has_ground_truths = does_predict_has_ground_truths
        self.batch_size = batch_size
        self.transform = transform
        self.collate_function = collate_function
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.shape_excluded = [] if shape_excluded is None else shape_excluded
        self.perturbation_excluded = [] if perturbation_excluded is None else perturbation_excluded

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SymDataset(
                data_source_path=Path(self.dataset_path) / 'train',
                transform=self.transform,
                has_ground_truth=True,
                shape_excluded=self.shape_excluded,
                perturbation_excluded=self.perturbation_excluded,
            )
            self.valid_dataset = SymDataset(
                data_source_path=Path(self.dataset_path) / 'valid',
                transform=self.transform,
                has_ground_truth=True,
                shape_excluded=self.shape_excluded,
                perturbation_excluded=self.perturbation_excluded,
            )

        if stage == "test":
            self.test_dataset = SymDataset(
                data_source_path=Path(self.dataset_path) / 'test',
                transform=self.transform,
                has_ground_truth=True,
                shape_excluded=self.shape_excluded,
                perturbation_excluded=self.perturbation_excluded,
            )

        if stage == "predict":
            self.predict_dataset = SymDataset(
                data_source_path=self.predict_data_path,
                transform=self.transform,
                has_ground_truth=self.does_predict_has_ground_truths,
                shape_excluded=self.shape_excluded,
                perturbation_excluded=self.perturbation_excluded,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )
