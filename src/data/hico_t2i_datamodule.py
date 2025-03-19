from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from lightning import LightningDataModule
from src.data.components.hico_t2i_dataset import HicoT2IDataset
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader


class HicoDataModule(LightningDataModule):

    def __init__(
        self,
        root="data/mmcelebahq",
        text_file="data/mmcelebahq/text.json",
        batch_size: int = 1,
        num_workers: int = 0,
        tokenizer_id="checkpoints/stablev15",
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `MNISTDataModule`."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: HicoT2IDataset = None
        self.val_dataset: HicoT2IDataset = None
        self.test_dataset: HicoT2IDataset = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        HicoT2IDataset(
            root=self.hparams.root,
            text_file=self.hparams.text_file,
            split="train",
            tokenizer_id=self.hparams.tokenizer_id,
        )

        HicoT2IDataset(
            root=self.hparams.root,
            text_file=self.hparams.text_file,
            tokenizer_id=self.hparams.tokenizer_id,
            split="val",
        )

    def setup(self, stage: Optional[str] = None) -> None:
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
            self.train_dataset = HicoT2IDataset(
                root=self.hparams.root,
                text_file=self.hparams.text_file,
                split="train",
                tokenizer_id=self.hparams.tokenizer_id,
            )

            self.val_dataset = self.test_dataset = HicoT2IDataset(
                root=self.hparams.root,
                text_file=self.hparams.text_file,
                tokenizer_id=self.hparams.tokenizer_id,
                split="val",
            )

    def collate_fn(self, examples):

        return {
            "instance_images": torch.stack(
                [example["instance_images"] for example in examples]
            ),
            "cls_mask": torch.stack(
                [torch.tensor(example["cls_mask"]) for example in examples]
            ),
            # "cls": torch.stack([torch.tensor(example["cls"]) for example in examples]),
            "cls_text": torch.stack([example["cls_text"] for example in examples]),
            "instance_prompt_ids": torch.stack(
                [example["instance_prompt_ids"] for example in examples]
            ),
        }

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=True,
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
            collate_fn=self.collate_fn,
            shuffle=False,
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
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
