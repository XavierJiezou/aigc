from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from lightning import LightningDataModule
from src.data.components import MMCelebahq
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader


class CelebahqDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_path="data/mmcelebahq",
        face_cache="data/mmcelebahq/face.zip",
        mask_cache="data/mmcelebahq/mask.zip",
        text_cache="data/mmcelebahq/text.json",
        size=512,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `MNISTDataModule`."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[MMCelebahq] = None
        self.val_dataset: Optional[MMCelebahq] = None
        self.test_dataset: Optional[MMCelebahq] = None

        self.batch_size_per_device = batch_size
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "checkpoints",
            subfolder="tokenizer",
        )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MMCelebahq(
            dataset_path=self.hparams.dataset_path,
            face_cache=self.hparams.face_cache,
            mask_cache=self.hparams.mask_cache,
            text_cache=self.hparams.text_cache,
            size=self.hparams.size,
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
            self.train_dataset = MMCelebahq(
                dataset_path=self.hparams.dataset_path,
                face_cache=self.hparams.face_cache,
                mask_cache=self.hparams.mask_cache,
                text_cache=self.hparams.text_cache,
                size=self.hparams.size,
                split="train",
            )

            self.val_dataset = self.test_dataset = MMCelebahq(
                dataset_path=self.hparams.dataset_path,
                face_cache=self.hparams.face_cache,
                mask_cache=self.hparams.mask_cache,
                text_cache=self.hparams.text_cache,
                size=self.hparams.size,
                split="val",
            )

    def collate_fn(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        instance_masks = [example["instance_masks"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        instance_masks = torch.stack(instance_masks)
        instance_masks = instance_masks.to(
            memory_format=torch.contiguous_format
        ).float()

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        batch = {
            "instance_prompt_ids": input_ids,
            "instance_images": pixel_values,
            "instance_masks": instance_masks,
        }
        return batch

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


if __name__ == "__main__":
    dataloader = CelebahqDataModule()
    dataloader.setup()
    tokenizer = CLIPTokenizer.from_pretrained(
        "checkpoints",
        subfolder="tokenizer",
    )
    from matplotlib import pyplot as plt

    for batch in dataloader.train_dataloader():

        plt.figure(figsize=(12, 8))

        instance_prompt_ids = batch["instance_prompt_ids"][0]
        original_text = tokenizer.decode(
            instance_prompt_ids, skip_special_tokens=True
        )
        print(original_text)
        image = batch["instance_images"][0]
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        mask = batch["instance_masks"]

        mask = mask.squeeze().detach().numpy().astype(np.uint8)

        palette = np.array(
        [
            (0, 0, 0),
            (204, 0, 0),
            (76, 153, 0),
            (204, 204, 0),
            (51, 51, 255),
            (204, 0, 204),
            (0, 255, 255),
            (51, 255, 255),
            (102, 51, 0),
            (255, 0, 0),
            (102, 204, 0),
            (255, 255, 0),
            (0, 0, 153),
            (0, 0, 204),
            (255, 51, 153),
            (0, 204, 204),
            (0, 51, 0),
            (255, 153, 51),
            (0, 204, 0),
        ],
        dtype=np.uint8,
    )

        color_mask = palette[mask]
        color_mask = Image.fromarray(color_mask)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(original_text,fontsize=8)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(color_mask)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"dataloader_show.png")
        break
