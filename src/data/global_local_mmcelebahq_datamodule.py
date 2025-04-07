from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from src.data.components.global_local_mask_mmcelebahq import GlobalLocalMaskMMCelebAHQ
from torchvision.transforms import transforms
from matplotlib import pyplot as plt


class GlobalLocalMaskMMCelebaqHQDataModule(LightningDataModule):

    def __init__(
        self,
        root: str = "data/mmcelebahq",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[GlobalLocalMaskMMCelebAHQ] = None
        self.val_dataset: Optional[GlobalLocalMaskMMCelebAHQ] = None
        self.test_dataset: Optional[GlobalLocalMaskMMCelebAHQ] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        GlobalLocalMaskMMCelebAHQ(
            root=self.hparams.root,
            drop_image_prob=self.hparams.drop_image_prob,
            drop_text_prob=self.hparams.drop_text_prob,
            phase="train",
        )
        GlobalLocalMaskMMCelebAHQ(
            root=self.hparams.root,
            drop_image_prob=self.hparams.drop_image_prob,
            drop_text_prob=self.hparams.drop_text_prob,
            phase="val",
        )

    def collate_fn(self, examples):
        instance_prompt_ids = [torch.tensor(example["instance_prompt_ids"]) for example in examples]
        instance_prompt_ids = torch.stack(instance_prompt_ids)
        image = [example["image"] for example in examples]
        image = torch.stack(image)
        condition = [example["condition"] for example in examples]
        condition = torch.stack(condition)

        return {
            "instance_prompt_ids": instance_prompt_ids,
            "image": image,
            "condition": condition,
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:

            self.train_dataset = GlobalLocalMaskMMCelebAHQ(
                root=self.hparams.root,
                drop_image_prob=self.hparams.drop_image_prob,
                drop_text_prob=self.hparams.drop_text_prob,
                phase="train",
            )

            self.val_dataset = GlobalLocalMaskMMCelebAHQ(
                root=self.hparams.root,
                drop_image_prob=self.hparams.drop_image_prob,
                drop_text_prob=self.hparams.drop_text_prob,
                phase="val",
            )
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
