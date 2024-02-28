from unittest.mock import MagicMock

import psutil

from src.utils.misc.find_num_worker_default import find_num_worker_default


class TestFindNumWorkerDefault:
    """Test the `find_num_worker_default` function."""

    def test_config_value_provided(self):
        """Test when a specific value is provided in the configuration."""
        config_value = 8
        assert find_num_worker_default(config_value) == config_value

    def test_cpu_cores_logical(self, monkeypatch):
        """Test when CPU cores are retrieved logically."""
        monkeypatch.setattr(psutil, "cpu_count", MagicMock(return_value=4))
        assert find_num_worker_default(None) == 4

    def test_cpu_cores_calculation(self, monkeypatch):
        """Test the calculation of default workers based on CPU cores."""
        monkeypatch.setattr(psutil, "cpu_count", MagicMock(return_value=10))
        assert find_num_worker_default(None) == 8

    def test_exception_handling(self, monkeypatch):
        """Test exception handling."""
        monkeypatch.setattr(psutil, "cpu_count", MagicMock(side_effect=Exception))
        assert find_num_worker_default(None) == 4
