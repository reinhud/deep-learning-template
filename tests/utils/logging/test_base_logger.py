import os

import pytest

from src.utils.logging.base_logger import BaseLogger

log_file = "test_logs.log"


@pytest.fixture
def base_logger():
    """Fixture to set up and tear down the BaseLogger for testing.

    This fixture creates an instance of BaseLogger with a test file path. It yields the logger
    object to the tests and ensures that the test log file is removed after each test.

    :yield: BaseLogger: The BaseLogger object for testing.
    """
    # Set up the logger with a test file path
    logger = BaseLogger(file=log_file)

    yield logger  # Provide the logger object to the tests

    # Teardown: Remove the test log file after each test
    if os.path.exists(log_file):
        os.remove(log_file)


class TestBaseLogger:
    """Tests for the BaseLogger class."""

    def test_logger_output(self, base_logger):
        """Test the logging functionality of BaseLogger.

        This test verifies that the BaseLogger correctly writes log messages to both the console
        and the log file. It logs messages at different severity levels (debug, info, warning,
        error, critical) and checks if these messages are present in the log file.

        :param base_logger: BaseLogger: An instance of BaseLogger created for testing.
        """
        # Test if the logger writes to both console and file
        base_logger.debug("Debug message")
        base_logger.info("Info message")
        base_logger.warning("Warning message")
        base_logger.error("Error message")
        base_logger.critical("Critical message")

        # Read the contents of the log file
        with open(log_file) as f:
            log_content = f.read()

        # Check if log messages are present in the file
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
        assert "Critical message" in log_content
