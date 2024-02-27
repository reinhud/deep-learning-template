from pathlib import Path

import pytest
import torch

from src.datasets.cifar10 import CIFAR10DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_cifar10_datamodule(batch_size: int) -> None:
    """Tests `CIFAR10DataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data"

    dm = CIFAR10DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert Path(data_dir, "cifar10").exists()

    dm.setup()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
