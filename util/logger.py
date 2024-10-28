import logging
import os


def get_logger(logfile: str, level: str = "DEBUG") -> logging.Logger:
    os.makedirs(os.path.split(logfile)[0], exist_ok=True)
    logger = logging.getLogger(logfile)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile, mode="a+", encoding="utf-8")
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    console_handler.setLevel(level)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
