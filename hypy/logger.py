"""Logging helpers used across the package."""

import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str = __name__, level: int = logging.INFO, verbose=True, log_file=None
) -> logging.Logger:
    """Return a configured logger instance.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default __name__
    level : int, optional
        Logging level, by default logging.INFO
    verbose : bool, optional
        Whether to output log messages to the console, by default True
    log_file : _type_, optional
        Path to the log file, by default None

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    under_pytest = "PYTEST_CURRENT_TEST" in os.environ

    if under_pytest:
        # Let pytest capture via its own handler/formatting
        logger.propagate = True
    else:
        # Standalone app/CLI: attach our console handler if none
        if not logger.handlers:
            if verbose:
                ch = logging.StreamHandler()
                ch.setLevel(logging.NOTSET)  # let logger level decide
                ch.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(ch)
                logger.propagate = False
            if log_file:
                fh = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
                fh.setLevel(logging.NOTSET)
                fh.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(fh)
                logger.propagate = False

    return logger


logger = get_logger(log_file="logs/log", level=logging.DEBUG)
