import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from src.utils.misc.find_num_worker_default import find_num_worker_default


class CIFAR10DataModule(L.LightningDataModule):
    """CIFAR10 DataModule."""

    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        train_ratio: float = 0.9,
        seed: int = 42,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
    ):
        """
        Initializes the CIFAR10DataModule.

        Args:
            batch_size (int): The batch size for data loaders.
            data_dir (str): The directory to store CIFAR10 dataset.
            train_ratio (float, optional): The ratio of the dataset to use for training. Defaults to 0.9.
            seed (int, optional): The seed for random operations. Defaults to 42.
            num_workers (int | None, optional): The number of worker processes. Defaults to None.
            pin_memory (bool | None, optional): Whether to pin memory for CUDA tensors. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.seed = seed
        self.train_ratio = train_ratio
        if num_workers is None:
            self.num_workers = find_num_worker_default(num_workers)
        else:
            self.num_workers = num_workers
        if pin_memory is None:
            self.pin_memory = True if torch.cuda.device_count() > 0 else False
        else:
            self.pin_memory = pin_memory

        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Downloads the CIFAR10 dataset."""
        try:
            CIFAR10(self.data_dir, train=True, download=True)
            self.class_to_idx = CIFAR10(self.data_dir, train=False, download=True).class_to_idx
        except Exception as e:
            raise RuntimeError(f"Failed to prepare CIFAR10 dataset: {e}")

    def setup(self, stage: str = None) -> None:
        """Apply transforms and plits the dataset."""
        # transforms
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        # split dataset
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=transform_train)
            train_size = int(50000 * self.train_ratio)
            val_size = 50000 - train_size
            self.cifar10_train, self.cifar10_val = random_split(
                dataset=cifar10_full,
                lengths=[train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=transform_test)

        if stage == "predict" or stage is None:
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=transform_test)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for training data."""
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for validation data."""
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for test data."""
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns a DataLoader for prediction data."""
        return DataLoader(
            self.cifar10_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
