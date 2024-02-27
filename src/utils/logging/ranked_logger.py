import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_zero_only

from src.utils.logging.base_logger import BaseLogger


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    # TODO: Fix this to also show the correct filename and linenumber where the call is originally from and
    # not only this file and line number

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ):
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param logger: The logger object to adapt.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process.
                                Default is `False`.
        """
        logger = BaseLogger(name=name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level, msg, *args, rank=None, **kwargs):
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param args: Additional args to pass to the underlying logging function.
        :param rank: The rank to log at.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None or current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
