import logging

disable_loggers = ["matplotlib.font_manager", "matplotlib"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
