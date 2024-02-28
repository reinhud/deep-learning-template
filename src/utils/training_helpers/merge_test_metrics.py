import csv
import json
import os

from src.utils.logging.base_logger import BaseLogger

log = BaseLogger(__name__)


def merge_test_metrics(metrics_file_path: str | os.PathLike, output_file_path: str | os.PathLike) -> None:
    """Create a CSV file from a JSON file containing test metrics, excluding 'predictions' and
    'targets'.

    Args:
        metrics_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output CSV file.
    """
    try:
        # Load the JSON data
        with open(metrics_file_path) as json_file:
            data = json.load(json_file)
            test_metrics = data["test"]

        # Write data to CSV file
        with open(output_file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Write header
            column_names = ("metric", "metric_value")
            writer.writerow(column_names)

            # Write each metric and its value, excluding 'predictions' and 'targets'
            for metric, value in test_metrics.items():
                if metric not in ["predictions", "targets", "test_probability_predicted_class"]:
                    writer.writerow([metric, value])
    except Exception as e:
        log.error(f"The file {metrics_file_path} was not found. Error: {e}")
