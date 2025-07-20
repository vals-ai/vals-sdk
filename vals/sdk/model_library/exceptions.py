import re

import backoff
from backoff._typing import Details
from typing_extensions import override

from vals.sdk.model_library.utils import get_logger

logger = get_logger(__name__)


class DoNotRetryException(Exception):
    def __init__(self, message: str, exception: Exception | None = None):
        self.message: str = message
        self.exception: Exception | None = exception
        super().__init__(message)
        if exception:
            self.__cause__: BaseException | None = exception

    @override
    def __str__(self):
        if self.exception:
            return f"{self.message} (caused by: {self.exception})"
        return self.message


class MaxOutputTokensExceededError(DoNotRetryException):
    """Raised when the output exceeds the allowed max output tokens limit."""

    DEFAULT_MESSAGE: str = (
        "Output exceeded your 'Max Output Tokens' limit. "
        "Consider increasing the limit in 'Run Suite' > 'Model Parameters'."
    )

    def __init__(self, message: str | None = None):
        super().__init__(message or MaxOutputTokensExceededError.DEFAULT_MESSAGE)


INVALID_KEYWORDS = [
    "unrecognized",
    "invalid",
    "unexpected",
    "unknown field",
    "extra inputs are not permitted",
]

INVALID_KEYWORDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(keyword) for keyword in INVALID_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_invalid_argument_error(e: Exception) -> bool:
    return bool(INVALID_KEYWORDS_RE.search(str(e)))


RETRY_MAX_TRIES: int | None = None
RETRY_BASE: float = 5.0
RETRY_FACTOR: float = 1.2
RETRY_MAX_BACKOFF_WAIT: float = 60.0
RETRY_MAX_TOTAL_TIME: float = 600.0  # 10 minutes total max retry time


def on_backoff(details: Details) -> None:
    exception = details.get("exception")
    tries = details.get("tries", 0)
    elapsed = details.get("elapsed", 0.0)
    wait = details.get("wait", 0.0)

    logger.warning(
        f"[Retrying] Exception: {exception} | Attempt: {tries} | "
        + f"Elapsed: {elapsed:.1f}s | Next wait: {wait:.1f}s"
    )


def giveup(e: Exception) -> bool:
    return isinstance(e, DoNotRetryException) or is_invalid_argument_error(e)


def on_giveup(details: Details) -> None:
    exception: Exception | None = details.get("exception", None)
    if not exception:
        return

    logger.error(f"Giving up after retries. Final exception: {exception}")

    if is_invalid_argument_error(exception):
        raise DoNotRetryException("Invalid argument detected; not retrying.", exception)
    elif isinstance(exception, DoNotRetryException):
        raise exception


def retry_llm_call(retry_failed_calls_indefinitely: bool = False):
    max_tries = None if retry_failed_calls_indefinitely else RETRY_MAX_TRIES
    max_time = None if retry_failed_calls_indefinitely else RETRY_MAX_TOTAL_TIME

    return backoff.on_exception(
        wait_gen=lambda: backoff.expo(
            base=RETRY_BASE,
            factor=RETRY_FACTOR,
            max_value=RETRY_MAX_BACKOFF_WAIT,
        ),
        exception=Exception,
        max_tries=max_tries,
        max_time=max_time,
        giveup=giveup,
        on_backoff=on_backoff,
        on_giveup=on_giveup,
        jitter=backoff.full_jitter,
    )
