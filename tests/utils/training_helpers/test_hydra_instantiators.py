import pytest
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import OmegaConf

from src.utils.training_helpers.hydra_instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)


class TestCallbackInstatiators:
    """Tests for the instantiation functions."""

    @pytest.fixture
    def callbacks_config(self):
        """Fixture providing sample callback configurations."""
        return OmegaConf.create(
            {
                "callback1": {"_target_": "src.callbacks.progress_bar.CustomRichProgressBar"},
                "callback2": {"_target_": "src.callbacks.model_summary.CustomRichModelSummary"},
            }
        )

    @pytest.mark.parametrize(
        "config",
        [
            (None),  # No configuration provided
            ({}),  # Empty configuration
        ],
    )
    def test_instantiate_callbacks_empty_configs(self, config):
        """Test instantiate_callbacks function with empty configurations.

        :param config: Empty callback configuration.
        """
        callbacks = instantiate_callbacks(config)
        assert not callbacks

    @pytest.mark.parametrize(
        "config",
        [
            ({"invalid_key": {"_target_": "module.Callback"}}),  # Invalid key in configuration
            ({"callback1": "invalid_target"}),  # Invalid target value
        ],
    )
    def test_instantiate_callbacks_invalid_configs(self, config):
        """Test instantiate_callbacks function with invalid configurations.

        :param config: Invalid callback configuration.
        """
        with pytest.raises((TypeError, KeyError, ValueError)):
            instantiate_callbacks(config)

    def test_instantiate_callbacks(self, callbacks_config):
        """Test instantiate_callbacks function.

        :param callbacks_config: Sample callback configurations.
        """
        callbacks = instantiate_callbacks(callbacks_config)
        assert len(callbacks) == 2
        assert all(isinstance(cb, Callback) for cb in callbacks)


class TestLoggerInstantiators:
    """Tests for the logging instantiation functions."""

    @pytest.fixture
    def loggers_config(self, tmp_path):
        """Fixture providing sample logger configurations."""
        return OmegaConf.create(
            {
                "logger1": {"_target_": "lightning.pytorch.loggers.CSVLogger", "save_dir": f"{tmp_path}"},
                "logger2": {"_target_": "dvclive.lightning.DVCLiveLogger", "dir": f"{tmp_path}"},
            }
        )

    @pytest.mark.parametrize(
        "config",
        [
            (None),  # No configuration provided
            ({}),  # Empty configuration
        ],
    )
    def test_instantiate_loggers_empty_configs(self, config):
        """Test instantiate_loggers function with empty configurations.

        :param config: Empty logger configuration.
        """
        loggers = instantiate_loggers(config)
        assert not loggers

    @pytest.mark.parametrize(
        "config",
        [
            ({"invalid_key": {"_target_": "module.Logger"}}),  # Invalid key in configuration
            ({"logger1": "invalid_target"}),  # Invalid target value
        ],
    )
    def test_instantiate_loggers_invalid_configs(self, config):
        """Test instantiate_loggers function with invalid configurations.

        :param config: Invalid logger configuration.
        """
        with pytest.raises((TypeError, KeyError, ValueError)):
            instantiate_loggers(config)

    def test_instantiate_loggers(self, loggers_config):
        """Test instantiate_loggers function.

        :param loggers_config: Sample logger configurations.
        """
        loggers = instantiate_loggers(loggers_config)
        assert len(loggers) == 2
        assert all(isinstance(lg, Logger) for lg in loggers)
