import psutil

from src.utils.logging.base_logger import BaseLogger

log = BaseLogger(__name__)


def find_num_worker_default(config_value: int | None):
    """Determine the optimal number of workers for data loading based on system resources.

    If a specific value is provided in the configuration, it will be used. Otherwise, the function
    calculates a reasonable default based on the number of available CPU cores. The default is set
    to either 4 or 80% of the available CPU cores (whichever is higher), subtracting one to avoid
    potential resource contention.

    :param config_value: An optional integer specifying the desired number of workers. If provided,
        this value will be used directly.
    :return: The calculated number of workers to be used for data loading.
    """
    if config_value is not None:
        return config_value
    else:
        log.info("Finding default number of workers for data loading.")
        try:
            # CPU cores
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

            # Set workers as a percentage of available CPU cores
            default_workers = max(4, min(cpu_cores - 1, int(cpu_cores * 0.8)))
            log.info(f"Available CPU cores: {cpu_cores}.")
            log.info(f"Setting number of workers to: {default_workers} for data loading.")

            return default_workers

        except Exception as e:
            log.info(f"Error while fetching system information: {e}")
            return 4
