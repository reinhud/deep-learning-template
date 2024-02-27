from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig

from src.utils.logging.base_logger import BaseLogger

log = BaseLogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)    # noqa: E501
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)    # noqa: E501
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # Things to do if exception occurs
        except Exception as ex:
            # Save exception to `.log` file
            log.exception("")

            # Some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # Things to always do after either success or exception
        finally:
            # Display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_path}")

        return metric_dict, object_dict

    return wrap
