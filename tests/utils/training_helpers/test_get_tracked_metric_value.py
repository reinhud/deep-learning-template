from typing import Any, Dict

import pytest
import torch

from src.utils.training_helpers.get_tracked_metric_value import get_tracked_metric_value


class TestGetTrackedMetricValue:
    """Tests for the get_tracked_metric_value function."""

    @pytest.mark.parametrize(
        "metric_dict, metric_name, expected_value",
        [
            ({}, None, None),  # When metric_name is None
            ({"metric1": 10}, "invalid_metric", None),  # When metric_name is not found
            ({"metric1": 10}, "metric1", 10.0),  # When metric_name is found
            ({"metric1": torch.tensor(10)}, "metric1", 10.0),  # When metric_value is a tensor
            (
                {"metric1": torch.tensor(10.5)},
                "metric1",
                10.5,
            ),  # When metric_value is a tensor with float value
            ({"metric1": torch.tensor(10)}, "metric1", 10.0),  # When metric_value is a tensor with int value
            (
                {"metric1": torch.tensor([10])},
                "metric1",
                10.0,
            ),  # When metric_value is a tensor with single value
            ({"metric1": 10.5}, "metric1", 10.5),  # When metric_value is a float
            ({"metric1": 10}, "metric1", 10.0),  # When metric_value is an int
            ({"metric1": "invalid_value"}, "metric1", None),  # When metric_value is not numeric
        ],
    )
    def test_get_tracked_metric_value(
        self, metric_dict: Dict[str, Any], metric_name: str, expected_value: Any
    ):
        """Test get_tracked_metric_value function with various scenarios.

        :param metric_dict: Dictionary containing metric values.
        :param metric_name: Name of the metric to retrieve.
        :param expected_value: Expected value of the retrieved metric.
        """
        assert get_tracked_metric_value(metric_dict, metric_name) == expected_value

    def test_get_tracked_metric_value_raises_value_error(self):
        """Test if ValueError is raised when metric_value is a tensor with multiple values."""
        with pytest.raises(RuntimeError):
            get_tracked_metric_value({"metric1": torch.tensor([10, 20])}, "metric1")
