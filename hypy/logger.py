# logging_utils.py
import logging
import os


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    under_pytest = "PYTEST_CURRENT_TEST" in os.environ

    if under_pytest:
        # Let pytest capture via its own handler/formatting
        logger.propagate = True
    else:
        # Standalone app/CLI: attach our console handler if none
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.NOTSET)  # let logger level decide
            ch.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(ch)
            logger.propagate = False

    return logger
