# TODO: fix this, doesn't work due to rank_zero_only decorator
'''import functools
from unittest import mock

import pytest
from omegaconf import OmegaConf

from src.utils.training_helpers.enforce_tags import enforce_tags

from lightning_utilities.core.rank_zero import rank_zero_only


@pytest.fixture
def mock_rank_zero_only():
    """Fixture to mock the rank_zero_only decorator."""
    with patch("src.utils.training_helpers.enforce_tags.rank_zero_only") as mock_decorator:
        # Define the behavior of the mock decorator
        def mock_wrapper(fn):
            # Define the wrapped function
            def wrapped_fn(*args, **kwargs):
                # Set the rank attribute
                wrapped_fn.rank = 0
                # Call the input function
                return fn(*args, **kwargs)

            return wrapped_fn

        # Set the return value of the mock decorator to the mock_wrapper function
        mock_decorator.return_value = mock_wrapper
        yield mock_decorator


@pytest.fixture
def cfg_with_tags():
    """Fixture providing a DictConfig object with existing tags."""
    return OmegaConf.create({"tags": ["tag1", "tag2"]})


@pytest.fixture
def cfg_without_tags():
    """Fixture providing a DictConfig object without existing tags."""
    return OmegaConf.create({})


# Apply the mock_rank_zero_only decorator to your test function
@pytest.mark.usefixtures("mock_rank_zero_only")
def test_enforce_tags_with_existing_tags(cfg_with_tags, tmp_path, mock_rank_zero_only):
    """Test enforce_tags function when existing tags are present in the config.

    :param cfg_with_tags: A DictConfig object with existing tags.
    :param tmp_path: A temporary directory path.
    """
    enforce_tags(cfg_with_tags)
    assert cfg_with_tags.tags == ["tag1", "tag2"]


def test_enforce_tags_without_existing_tags(cfg_without_tags, tmp_path):
    """Test enforce_tags function when no tags are present in the config.

    :param cfg_without_tags: A DictConfig object without existing tags.
    :param tmp_path: A temporary directory path.
    """
    with patch("src.utils.training_helpers.enforce_tags.Prompt.ask", return_value="new_tag1, new_tag2"):
        enforce_tags(cfg_without_tags)
    assert cfg_without_tags.tags == ["new_tag1", "new_tag2"]


def test_enforce_tags_save_to_file(cfg_without_tags, tmp_path):
    """Test enforce_tags function to save tags to a file.

    :param cfg_without_tags: A DictConfig object without existing tags.
    :param tmp_path: A temporary directory path.
    """
    enforce_tags(cfg_without_tags, save_to_file=True)
    tags_log_file = tmp_path / "log_path" / "tags.log"
    assert tags_log_file.exists()
    with open(tags_log_file, "r") as f:
        content = f.read()
        assert "new_tag1" in content
        assert "new_tag2" in content'''
