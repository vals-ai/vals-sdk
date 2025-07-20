import inspect
import logging
from typing import NamedTuple


class FileContent(NamedTuple):
    content: str
    extension: str
    type: str


MAX_LLM_LOG_LENGTH = 50


def truncate_str(s: str, max_len: int = MAX_LLM_LOG_LENGTH) -> str:
    if len(s) <= max_len:
        return s
    half = (max_len - 1) // 2
    return s[:half] + "…" + s[-half:]


def get_logger(name: str | None = None):
    if not name:
        caller = inspect.stack()[1]
        module = inspect.getmodule(caller[0])
        name = module.__name__ if module else "__main__"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
