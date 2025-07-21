import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic.main import BaseModel


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


def deep_model_dump(obj: object) -> object:
    if isinstance(obj, BaseModel):
        return deep_model_dump(obj.model_dump())

    if isinstance(obj, Mapping):
        return {k: deep_model_dump(v) for k, v in obj.items()}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [deep_model_dump(v) for v in obj]

    return obj
