import logging
from pathlib import Path
import sys


def get_logger(name: str = "exp", log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)
    logger.addHandler(stream_h)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(Path(log_dir)/f"{name}.log")
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)
    return logger
