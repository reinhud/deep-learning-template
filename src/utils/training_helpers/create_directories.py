from pathlib import Path
from typing import List

from src.utils.logging.base_logger import BaseLogger

log = BaseLogger(__name__)


def create_directories(paths: List[str]) -> None:
    """Create the specified directories if they don't already exist.

    :param path_list: A list of paths to create.
    """
    log.info(f"Checking if directories: {paths} exist.")
    for path in paths:
        if path is not None:
            log.info(f"Creating directory: {path}.")
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            log.error("Received None for a path which is not expected. Check your configuration.")
