import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from pprint import pformat
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    NamedTuple,
    TypeVar,
)

from attr import dataclass
from pydantic.fields import Field
from pydantic.main import BaseModel
from typing_extensions import TypedDict, override

from vals.sdk.model_library.exceptions import retry_llm_call
from vals.sdk.model_library.utils import truncate_str

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95


class FileBase(BaseModel):
    type: Literal["image", "file"]
    name: str
    mime: str


class FileWithBase64(FileBase):
    append_type: Literal["base64"] = "base64"
    base64: str


class FileWithUrl(FileBase):
    append_type: Literal["url"] = "url"
    url: str


class FileWithId(FileBase):
    append_type: Literal["file_id"] = "file_id"
    file_id: str


FileInput = Annotated[
    FileWithBase64 | FileWithUrl | FileWithId,
    Field(discriminator="append_type"),
]


class Files(NamedTuple):
    type: Literal["file", "image"]
    append_type: Literal["base64", "url", "file_id"]
    name: str
    mime: str
    base64: str


class QueryResultMetadata(TypedDict, total=False):
    in_tokens: int
    out_tokens: int
    duration_seconds: float | None
    reasoning: str | None
    reasoning_tokens: int | None


QueryResult = tuple[str, QueryResultMetadata | None]

T = TypeVar("T")


@dataclass
class LLMConfig:
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    reasoning: bool = False
    supports_metadata: bool = True
    supports_images: bool = False
    supports_files: bool = False
    supports_batch: bool = False
    retry_failed_calls_indefinitely: bool = False
    native: bool = True


class LLM(ABC):
    """
    Base class for all LLMs
    LLM call errors should be raised as exceptions
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig | None = None,
    ):
        self.provider: str = provider
        self.model_name: str = model_name

        config = config or LLMConfig()
        self._config: LLMConfig = config  # use only for logging

        self.max_tokens: int = config.max_tokens
        self.temperature: float = config.temperature
        self.top_p: float = config.top_p

        self.reasoning: bool = config.reasoning

        self.supports_metadata: bool = config.supports_metadata
        self.supports_files: bool = config.supports_files
        self.supports_images: bool = config.supports_images
        self.supports_batch: bool = config.supports_batch

        self.native: bool = config.native

        self.batch: LLMBatchMixin | None = None
        self.delete_batch: LLMBatchMixin | None = None

        self.duration_seconds: float | None = None
        self.logger: logging.Logger = logging.getLogger(
            f"llm.{provider}.{model_name}(native:{self.native})"
        )
        self.custom_retrier: (
            Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]
            | None
        ) = retry_llm_call(
            retry_failed_calls_indefinitely=config.retry_failed_calls_indefinitely
        )

    @override
    def __repr__(self):
        attrs = vars(self).copy()
        attrs.pop("logger", None)
        attrs.pop("custom_retrier", None)
        attrs.pop("_config", None)
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2)}\n)"

    @abstractmethod
    def get_client(self) -> object:
        """Return the instance of the appropriate SDK client."""
        ...

    async def timed_query(
        self,
        prompt: str,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        """
        Time the query
        """
        start = time.perf_counter()
        result, metadata = await self._query_impl(
            prompt, images=images, files=files, **kwargs
        )
        duration = time.perf_counter() - start

        if metadata is not None:
            metadata["duration_seconds"] = duration

        return result, metadata

    async def query(
        self,
        prompt: str,
        *,
        images: list[FileInput] | None = None,
        files: list[FileInput] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        """Query the model. Log, time, and retry"""
        short_prompt = truncate_str(repr(prompt))
        short_kwargs = {k: truncate_str(repr(v)) for k, v in kwargs.items()}

        images = images or []
        files = files or []

        self.logger.info(
            f"Calling query with prompt: {short_prompt} --- "
            + f"{len(images)} images, {len(files)} files, "
            + f"kwargs: {short_kwargs}"
        )

        if self.custom_retrier:
            return await self.custom_retrier(self.timed_query)(
                prompt, images=images, files=files, **kwargs
            )  # type: ignore
        else:
            return await self.timed_query(prompt, images=images, files=files, **kwargs)

    @abstractmethod
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        """
        Query the model with a prompt
        Optional images, files, and arguments
        Images and files should be preprocessed according to what the model supports:
            - base64
            - url
            - file_id
        """
        ...


class BatchResult(BaseModel):
    custom_id: str
    llm_output: str
    live_metadata: QueryResultMetadata | None = None
    error_message: str | None = None


class LLMBatchMixin(ABC):
    @abstractmethod
    def create_batch_query_request(
        self,
        custom_id: str,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        """Return a single query request.

        The batch api sends out a batch of query requests to various endpoints.

        For example OpenAI sends can send requests to /v1/responses or /v1/chat/completions endpoints.

        This method creates a query request for methods such as these.
        """
        ...

    @abstractmethod
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        """
        Batch query the model
        Returns:
            str: batch_id
        Raises:
            Exception: If failed to batch query
        """
        ...

    @abstractmethod
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        """
        Returns results for batch
        Raises:
            Exception: If failed to get results
        """
        ...

    @abstractmethod
    async def get_batch_progress(self, batch_id: str) -> int:
        """
        Returns number of completed requests for batch
        Raises:
            Exception: If failed to get progress
        """
        ...

    @abstractmethod
    async def cancel_batch_request(self, batch_id: str) -> None:
        """
        Cancels batch
        Raises:
            Exception: If failed to cancel
        """
        ...

    @abstractmethod
    async def get_batch_status(
        self,
        batch_id: str,
    ) -> str:
        """
        Returns batch status
        Raises:
            Exception: If failed to get status
        """
        ...

    @classmethod
    @abstractmethod
    def is_status_completed(
        cls,
        batch_status: str,
    ) -> bool:
        """
        Returns if batch status is completed

        A completed state is any state that is final and not in-progress
        Example: failed | cancelled | expired | completed

        An incompleted state is any state that is not completed
        Example: in_progress | pending | running
        """
        ...

    @classmethod
    @abstractmethod
    def is_status_failed(
        cls,
        batch_status: str,
    ) -> bool:
        """Returns if batch status is failed"""
        ...

    @classmethod
    @abstractmethod
    def is_status_cancelled(
        cls,
        batch_status: str,
    ) -> bool:
        """Returns if batch status is cancelled"""
        ...
