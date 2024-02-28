import json

import pytest

from src.utils.logging.base_logger import BaseLogger
from src.utils.training_helpers.merge_test_metrics import merge_test_metrics

log = BaseLogger(__name__)


class TestMergeTestMetrics:
    """Tests for the merge_test_metrics function."""

    def test_merge_test_metrics_valid_paths(self, tmp_path):
        """Test merge_test_metrics with valid input and output file paths.

        This test checks whether the merge_test_metrics function correctly generates a CSV file
        from a JSON file containing test metrics, excluding 'predictions' and 'targets'.

        :param tmp_path: pytest temporary directory path fixture.
        """
        input_json = {"test": {"metric1": 10, "metric2": 20}}
        input_json_path = tmp_path / "input.json"
        output_csv_path = tmp_path / "output.csv"

        # Write sample JSON data to a file
        with open(input_json_path, "w") as json_file:
            json.dump(input_json, json_file)

        # Call the function under test
        merge_test_metrics(input_json_path, output_csv_path)

        # Assert that the output CSV file was created
        assert output_csv_path.exists()

    def test_merge_test_metrics_non_existing_json(self, tmp_path):
        """Test merge_test_metrics with non-existing JSON file path.

        This test checks whether the merge_test_metrics function raises a FileNotFoundError when
        provided with a non-existing JSON file path.

        :param tmp_path: pytest temporary directory path fixture.
        """
        output_csv_path = tmp_path / "output.csv"

        # Call the function under test with a non-existing JSON file path
        merge_test_metrics("non_existing", output_csv_path)

        # Assert that the output CSV file was not created
        assert not output_csv_path.exists()

    def test_merge_test_metrics_invalid_json_content(self, tmp_path):
        """Test merge_test_metrics with JSON file containing unexpected content.

        This test checks whether the merge_test_metrics function raises a KeyError when provided
        with a JSON file containing unexpected content.

        :param tmp_path: pytest temporary directory path fixture.
        """
        invalid_json_content = {"some_key": "some_value"}
        input_json_path = tmp_path / "input.json"
        output_csv_path = tmp_path / "output.csv"

        # Write invalid JSON content to a file
        with open(input_json_path, "w") as json_file:
            json.dump(invalid_json_content, json_file)

        # Call the function under test
        merge_test_metrics(input_json_path, output_csv_path)

        # Assert that the output CSV file was not created
        assert not output_csv_path.exists()
