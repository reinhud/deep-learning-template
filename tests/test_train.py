import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.pytest_helpers.run_if import RunIf


class TestTrain:
    """Tests the training script train.py."""

    def test_train_fast_dev_run(self, cfg_train: DictConfig) -> None:
        """Run for 1 train, val, and test step.

        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        HydraConfig().set_config(cfg_train)

        with open_dict(cfg_train):
            cfg_train.trainer.fast_dev_run = True
            cfg_train.trainer.accelerator = "cpu"

        train(cfg_train)

    @RunIf(min_gpus=1)
    def test_train_fast_dev_run_gpu(self, cfg_train: DictConfig) -> None:
        """Run for 1 train, val, and test step on GPU.

        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        HydraConfig().set_config(cfg_train)

        with open_dict(cfg_train):
            cfg_train.trainer.fast_dev_run = True
            cfg_train.trainer.accelerator = "gpu"

        train(cfg_train)

    @RunIf(min_gpus=1)
    @pytest.mark.slow
    def test_train_epoch_gpu_amp(self, cfg_train: DictConfig) -> None:
        """Train 1 epoch on GPU with mixed-precision.

        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        HydraConfig().set_config(cfg_train)

        with open_dict(cfg_train):
            cfg_train.trainer.accelerator = "gpu"
            cfg_train.trainer.precision = 16

        train(cfg_train)

    @pytest.mark.slow
    def test_train_epoch_double_val_loop(self, cfg_train: DictConfig) -> None:
        """Train 1 epoch with validation loop twice per epoch.

        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        HydraConfig().set_config(cfg_train)

        with open_dict(cfg_train):
            cfg_train.trainer.val_check_interval = 0.5

        train(cfg_train)

    @pytest.mark.slow
    def test_train_resume(self, tmp_path: Path, cfg_train: DictConfig) -> None:
        """Run 1 epoch, finish, and resume for another epoch.

        :param tmp_path: The temporary logging path.
        :param cfg_train: A DictConfig containing a valid training configuration.
        """
        with open_dict(cfg_train):
            cfg_train.trainer.max_epochs = 1

        HydraConfig().set_config(cfg_train)

        metric_dict_1, _ = train(cfg_train)

        # Ensure that the checkpoint was created
        files = os.listdir(cfg_train.paths.ckpt_path)
        assert "best_model.ckpt" in files

        with open_dict(cfg_train):
            cfg_train.trainer.max_epochs = 2

        # Resume training
        metric_dict_2, _ = train(cfg_train)

        files = os.listdir(cfg_train.paths.ckpt_path)
        assert "best_model.ckpt" in files
