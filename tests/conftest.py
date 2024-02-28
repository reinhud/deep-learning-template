"""This file prepares config fixtures for other tests."""

# TODO: Add tests for util functions
# TODO: add tests for running dvc experiments
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # Set defaults for all tests
        with open_dict(cfg):
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.dataset.num_workers = 0
            cfg.dataset.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["paths.ckpt_path=."])

        # Set defaults for all tests
        with open_dict(cfg):
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.dataset.num_workers = 0
            cfg.dataset.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.loggers = None

    return cfg


@pytest.fixture(scope="function")
def test_logger_config(tmp_path) -> DictConfig:
    """Fixture to generate a test logger configuration."""
    return OmegaConf.create(
        {
            "csv_logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(tmp_path),
                "name": "pytest_testing",
                "version": None,
                "prefix": "",
                "flush_logs_every_n_steps": 100,
            }
        }
    )


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path, test_logger_config: DictConfig) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg.
    Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    # Update the output, log paths and logger
    with open_dict(cfg):
        cfg.paths.output_path = str(tmp_path)
        cfg.paths.log_path = str(tmp_path)
        cfg.loggers = test_logger_config

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path, test_logger_config: DictConfig) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg.
    Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    # Update the output, log paths and logger
    with open_dict(cfg):
        cfg.paths.output_path = str(tmp_path)
        cfg.paths.log_path = str(tmp_path)
        cfg.loggers = test_logger_config

    yield cfg

    GlobalHydra.instance().clear()
