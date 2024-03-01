import logging
from typing import Optional

from src.utils.logging.logging_formatter import base_logging_formatter


class BaseLogger(logging.Logger):
    """Python logger with custom format that outputs to console and file."""

    def __init__(
        self,
        name: str = __name__,
        level: int = logging.DEBUG,
        file: Optional[str] = "output/logs/base_logger.log",
    ):
        """Initializes a python logger with a custom format that outputs to console and file.

        :param name: The name of the logger.
        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param file: The file to write logs to. Default is `None`.
        :param log_format: The format string for log messages. Default is `None`.
        """
        super().__init__(name, level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(base_logging_formatter)

        # Add console handler to the logger
        self.addHandler(console_handler)

        # Create file handler if file is provided
        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setLevel(level)
            # file_handler.setFormatter(base_logging_formatter)

            # aAd file handler to the logger
            self.addHandler(file_handler)
