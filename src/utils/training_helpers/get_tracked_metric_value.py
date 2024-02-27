from typing import Any, Dict, Optional

from src.utils.logging.base_logger import BaseLogger

log = BaseLogger(__name__)


def get_tracked_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        log.error(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
        return None

    # Retrieve and log metric value
    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved value for optimized metric '{metric_name}': {round(metric_value, 4)}")

    return metric_value
