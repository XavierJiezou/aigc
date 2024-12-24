import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    """`LightningDataModule` for the CIFAR-10 dataset."""

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (50_000, 5_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        size=256,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CIFAR10DataModule`."""
        super().__init__()

        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(size,size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # CIFAR-10 normalization
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return 10

    def prepare_data(self) -> None:
        """Download data if needed."""
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training or testing."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Save the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the datamodule state."""
        pass

def show(dataloader:DataLoader,num_samples: int = 5) -> None:
    """Display a few random images from the CIFAR-10 dataset."""
    # Select a random batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Convert the images from Tensor to NumPy for displaying
    images = images[:num_samples]  # Limit the number of samples to show
    labels = labels[:num_samples]

    # Convert the images to the original format for display
    images = images / 2 + 0.5  # Undo normalization
    images = images.permute(0, 2, 3, 1)  # Change shape from (N, C, H, W) to (N, H, W, C)

    # Create a plot with the images
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 6))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].numpy())
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("cifar10.png")

    plt.close()


if __name__ == "__main__":
    # Create the data module instance
    data_module = CIFAR10DataModule(batch_size=32)

    # Prepare data and setup
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.train_dataloader()
    print(len(dataloader))
    # Show some random CIFAR-10 images
    show(data_module.train_dataloader(),num_samples=5)
