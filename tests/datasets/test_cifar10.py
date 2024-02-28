from pathlib import Path

import pytest
import torch

from src.datasets.cifar10 import CIFAR10DataModule


class TestCIFAR10DataModule:
    """Tests for the `CIFAR10DataModule` class."""

    @pytest.mark.parametrize("batch_size", [32, 128])
    def test_cifar10_datamodule(self, batch_size: int) -> None:
        """Tests `CIFAR10DataModule` to verify that it can be downloaded correctly, that the
        necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch
        sizes correctly match.

        :param batch_size: Batch size of the data to be loaded by the dataloader.
        """
        data_dir = "data/cifar10"

        # Instantiate datamodule
        dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size)
        dm.prepare_data()

        # Check if data was downloaded
        assert Path(data_dir).exists()

        # Check if dataloaders were created
        dm.setup()
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        # Check if batch sizes and dtypes match
        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert len(x) == batch_size
        assert len(y) == batch_size
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
