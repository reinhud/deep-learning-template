from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.utils.training_helpers.apply_extra_utilities import apply_extra_utilities


@pytest.fixture
def apply_extra_utilities_mocks(monkeypatch):
    """Fixture to provide mocks for apply_extra_utilities function.

    This fixture creates mocks for the enforce_tags, print_config_tree, and filterwarnings
    functions used in apply_extra_utilities. These mocks are then patched into the
    apply_extra_utilities function.

    :param monkeypatch: The pytest monkeypatch fixture.
    :return: Tuple: Mock objects for enforce_tags, print_config_tree, and filterwarnings.
    """
    mock_enforce_tags = MagicMock()
    mock_print_config_tree = MagicMock()
    mock_filterwarnings = MagicMock()

    monkeypatch.setattr("src.utils.training_helpers.apply_extra_utilities.enforce_tags", mock_enforce_tags)
    monkeypatch.setattr(
        "src.utils.training_helpers.apply_extra_utilities.print_config_tree", mock_print_config_tree
    )
    monkeypatch.setattr("warnings.filterwarnings", mock_filterwarnings)

    return mock_enforce_tags, mock_print_config_tree, mock_filterwarnings


class TestApplyExtraUtilities:
    """Unit tests for apply_extra_utilities.py."""

    def test_apply_extra_utilities_no_extras_config(self, cfg_train: DictConfig, apply_extra_utilities_mocks):
        """Test apply_extra_utilities when no `extras` config is provided.

        This test verifies the behavior of apply_extra_utilities when no 'extras' configuration is
        provided. It checks that none of the utility functions are called.

        :param cfg_train: A DictConfig containing training configuration.
        :param apply_extra_utilities_mocks: Mock objects for enforce_tags, print_config_tree, and
            filterwarnings.
        """
        cfg = OmegaConf.create({})

        mock_enforce_tags, mock_print_config_tree, mock_filterwarnings = apply_extra_utilities_mocks

        apply_extra_utilities(cfg)

        mock_filterwarnings.assert_not_called()
        mock_enforce_tags.assert_not_called()
        mock_print_config_tree.assert_not_called()

    def test_apply_extra_utilities_ignore_warnings(self, cfg_train: DictConfig, apply_extra_utilities_mocks):
        """Test apply_extra_utilities when `ignore_warnings` is set to True.

        This test verifies the behavior of apply_extra_utilities when 'ignore_warnings'
        configuration is set to True. It checks that filterwarnings is called while enforce_tags
        and print_config_tree are not called.

        :param cfg_train: A DictConfig containing training configuration.
        :param apply_extra_utilities_mocks: Mock objects for enforce_tags, print_config_tree, and
            filterwarnings.
        """
        cfg = OmegaConf.create({"extras": {"ignore_warnings": True}})

        mock_enforce_tags, mock_print_config_tree, mock_filterwarnings = apply_extra_utilities_mocks

        apply_extra_utilities(cfg)

        mock_filterwarnings.assert_called_once()
        mock_enforce_tags.assert_not_called()
        mock_print_config_tree.assert_not_called()

    def test_apply_extra_utilities_enforce_tags_called(
        self, cfg_train: DictConfig, apply_extra_utilities_mocks
    ):
        """Test apply_extra_utilities when `enforce_tags` is set to True.

        This test verifies the behavior of apply_extra_utilities when 'enforce_tags' configuration
        is set to True. It checks that enforce_tags is called while filterwarnings and
        print_config_tree are not called.

        :param cfg_train: A DictConfig containing training configuration.
        :param apply_extra_utilities_mocks: Mock objects for enforce_tags, print_config_tree, and
            filterwarnings.
        """
        cfg = OmegaConf.create({"extras": {"enforce_tags": True}})

        mock_enforce_tags, mock_print_config_tree, mock_filterwarnings = apply_extra_utilities_mocks
        apply_extra_utilities(cfg)

        mock_filterwarnings.assert_not_called()
        mock_enforce_tags.assert_called_once()
        mock_print_config_tree.assert_not_called()

    def test_apply_extra_utilities_print_config_called(
        self, cfg_train: DictConfig, apply_extra_utilities_mocks
    ):
        """Test apply_extra_utilities when `print_config` is set to True.

        This test verifies the behavior of apply_extra_utilities when 'print_config' configuration
        is set to True. It checks that print_config_tree is called while filterwarnings and
        enforce_tags are not called.

        :param cfg_train: A DictConfig containing training configuration.
        :param apply_extra_utilities_mocks: Mock objects for enforce_tags, print_config_tree, and
            filterwarnings.
        """
        cfg = OmegaConf.create({"extras": {"print_config": True}})

        mock_enforce_tags, mock_print_config_tree, mock_filterwarnings = apply_extra_utilities_mocks
        apply_extra_utilities(cfg)

        mock_filterwarnings.assert_not_called()
        mock_enforce_tags.assert_not_called()
        mock_print_config_tree.assert_called_once()

    def test_apply_extra_utilities_all_called(self, cfg_train: DictConfig, apply_extra_utilities_mocks):
        """Test apply_extra_utilities when all `extras` are set to True.

        This test verifies the behavior of apply_extra_utilities when all 'extras' configuration
        options are set to True. It checks that all utility functions are called.

        :param cfg_train: A DictConfig containing training configuration.
        :param apply_extra_utilities_mocks: Mock objects for enforce_tags, print_config_tree, and
            filterwarnings.
        """
        cfg = OmegaConf.create(
            {"extras": {"ignore_warnings": True, "enforce_tags": True, "print_config": True}}
        )

        mock_enforce_tags, mock_print_config_tree, mock_filterwarnings = apply_extra_utilities_mocks
        apply_extra_utilities(cfg)

        mock_filterwarnings.assert_called_once()
        mock_enforce_tags.assert_called_once()
        mock_print_config_tree.assert_called_once()
