from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from typing import Any, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.moderation_create_response import ModerationCreateResponse
from openai.types.responses.response import Response
from pydantic import BaseModel
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    BatchResult,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    LLMBatchMixin,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import (
    DoNotRetryException,
    MaxOutputTokensExceededError,
)

PydanticT = TypeVar("PydanticT", bound=BaseModel)


class OpenAIBatchMixin(LLMBatchMixin):
    COMPLETED_BATCH_STATUSES: list[str] = [
        "failed",
        "completed",
        "expired",
        "cancelled",
    ]

    def __init__(self, openai: OpenAIModel):
        self._root: OpenAIModel = openai
        self._client: AsyncOpenAI = self._root.get_client()

    @override
    def create_batch_query_request(
        self,
        custom_id: str,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": self._root.query_body(prompt, images=images, files=files, **kwargs),
        }

    @override
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        """Sends a batch api query and returns batch id."""
        input_jsonl_str = "\n".join(json.dumps(req) for req in requests)
        input_jsonl_bytes = io.BytesIO(input_jsonl_str.encode("utf-8"))
        input_jsonl_bytes.name = batch_name

        batch_input_file = await self._client.files.create(
            file=input_jsonl_bytes, purpose="batch"
        )

        # TODO: Parameterize completion window
        completion_window = "24h"

        batch = await self._client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
            metadata={"description": batch_name},
        )
        return batch.id

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        batch = await self._client.batches.retrieve(batch_id)

        if not batch:
            raise Exception(f"Couldn't retrieve batch results for batch {batch_id}.")

        batch_results: list[BatchResult] = []

        if batch.output_file_id:
            successful_responses = await self._client.files.content(
                batch.output_file_id
            )
            successful_results: list[dict[str, Any]] = [
                json.loads(line) for line in successful_responses.iter_lines() if line
            ]
            for result in successful_results:
                id = cast(str, result["response"]["body"]["id"])
                response: Response = await self._client.responses.retrieve(id)

                metadata: QueryResultMetadata | None = None
                if response.usage:
                    metadata = QueryResultMetadata(
                        in_tokens=response.usage.input_tokens,
                        out_tokens=response.usage.output_tokens,
                    )

                batch_results.append(
                    BatchResult(
                        custom_id=cast(str, result["custom_id"]),
                        llm_output=response.output_text,
                        live_metadata=metadata,
                    )
                )

        if batch.error_file_id:
            failed_responses = await self._client.files.content(batch.error_file_id)
            failed_results: list[dict[str, Any]] = [
                json.loads(line) for line in failed_responses.iter_lines() if line
            ]
            for result in failed_results:
                error_message = cast(
                    str, result["response"]["body"]["error"]["message"]
                )
                batch_results.append(
                    BatchResult(
                        custom_id=cast(str, result["custom_id"]),
                        llm_output=error_message,
                        error_message=error_message,
                    )
                )

        return batch_results

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        batch = await self._client.batches.retrieve(batch_id)
        if batch and batch.request_counts:
            completed = batch.request_counts.completed
        else:
            self._root.logger.error(f"Couldn't retrieve {batch_id}")
            completed = 0
        return completed

    @override
    async def cancel_batch_request(self, batch_id: str):
        self._root.logger.info(f"Cancelling {batch_id}")
        _ = await self._client.batches.cancel(batch_id)

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        batch = await self._client.batches.retrieve(batch_id)
        self._root.logger.info(f"Batch: {batch}")
        return batch.status

    @override
    @classmethod
    def is_status_completed(cls, batch_status: str) -> bool:
        return batch_status in OpenAIBatchMixin.COMPLETED_BATCH_STATUSES

    @override
    @classmethod
    def is_status_failed(cls, batch_status: str) -> bool:
        return batch_status == "failed"

    @override
    @classmethod
    def is_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == "cancelled"


class OpenAIModel(LLM):
    MAX_CHARACTERS: int = 1048576
    TIMEOUT: int = 30 * 60

    @override
    def get_client(self) -> AsyncOpenAI:
        # client is created for every instance of OpenAIModel
        return self._openai_client

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        *,
        config: LLMConfig | None = None,
        custom_client: AsyncOpenAI | None = None,
        use_completions: bool = False,
        supports_temperature: bool = True,
    ):
        super().__init__(model_name, provider, config=config)
        self.supports_temperature: bool = supports_temperature
        self.use_completions: bool = use_completions

        # allow custom client to act as delegate for other providers (native)
        self._openai_client: AsyncOpenAI = custom_client or AsyncOpenAI(
            api_key=sdk.model_library_settings.OPENAI_KEY
        )

        # batch client
        self.supports_batch: bool = self.supports_batch and not custom_client
        self.batch: LLMBatchMixin | None = (
            OpenAIBatchMixin(self) if self.supports_batch else None
        )

        if "o1-mini" in self.model_name:
            self.use_completions = True

        if (
            "o1" in self.model_name
            or "o3" in self.model_name
            or "o4" in self.model_name
        ):
            self.supports_temperature = False

    # supported formats: png, jpeg, webp, gif
    # max 500 images
    def _append_images(
        self,
        images: list[FileInput],
    ) -> Iterator[dict[str, Any]]:
        """Append images to the request body"""
        if not images:
            return iter(())
        for image in images:
            base_dict: dict[str, Any]
            if self.use_completions:
                base_dict = {
                    "type": "image_url",
                    "image_url": {
                        "detail": "auto",
                    },
                }
                match image:
                    case FileWithBase64():
                        base_dict["image_url"]["url"] = (
                            f"data:image/{image.mime};base64,{image.base64}"
                        )
                    case FileWithUrl():
                        base_dict["image_url"]["url"] = image.url
                    case FileWithId():
                        raise Exception("Completions endpoint does not support file_id")
            else:
                base_dict = {
                    "type": "input_image",
                    "detail": "auto",
                }
                match image:
                    case FileWithBase64():
                        base_dict["image_url"] = (
                            f"data:image/{image.mime};base64,{image.base64}"
                        )
                    case FileWithUrl():
                        base_dict["image_url"] = image.url
                    case FileWithId():
                        base_dict["file_id"] = image.file_id
            yield base_dict

    def _append_files(
        self,
        files: list[FileInput],
    ) -> Iterator[dict[str, Any]]:
        """Append files to the request body"""
        if not files:
            return iter(())
        for file in files:
            base_dict: dict[str, Any]
            if self.use_completions:
                base_dict = {
                    "type": "file",
                    "file": {},
                }
                match file:
                    case FileWithBase64():
                        base_dict["file"]["file_data"] = file.base64
                    case FileWithUrl():
                        raise Exception("Completions endpoint does not support url")
                    case FileWithId():
                        base_dict["file"]["file_id"] = file.file_id
            else:
                base_dict = {
                    "type": "input_file",
                }
                match file:
                    case FileWithBase64():
                        base_dict["file_data"] = file.base64
                    case FileWithUrl():
                        base_dict["file_url"] = file.url
                    case FileWithId():
                        base_dict["file_id"] = file.file_id
            yield base_dict

    async def _query_completions(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        """Completions endpoint"""

        content_user: list[dict[str, Any]] = []
        content_user.extend(self._append_images(images))
        content_user.extend(self._append_files(files))
        content_user.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = []
        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_user})

        body: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": OpenAIModel.TIMEOUT,
            "messages": messages,
        }
        body.update(kwargs)

        if not self.supports_temperature:
            del body["temperature"]

        completion: ChatCompletion = await self.get_client().chat.completions.create(
            **body, stream=False
        )  # type: ignore

        if not completion.choices or not completion.choices[0].message.content:
            raise Exception("Model returned no completions")
        if completion.choices[0].finish_reason == "length":
            raise MaxOutputTokensExceededError()

        metadata = None
        if completion.usage:
            metadata = QueryResultMetadata(
                in_tokens=completion.usage.prompt_tokens,
                out_tokens=completion.usage.completion_tokens,
            )
        text = completion.choices[0].message.content

        if self.reasoning and metadata and self.provider != "openai":
            think_content_list: list[str] = re.findall(
                r"<think>(.*?)</think>", text, flags=re.DOTALL
            )
            if len(think_content_list) == 1:
                think_content = think_content_list[0]
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            else:
                think_content = "Error: multiple or no reasoning tokens found"

            metadata["reasoning"] = think_content

        return text, metadata

    def query_body(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        content_user: list[dict[str, Any]] = []
        content_user.extend(self._append_images(images))
        content_user.extend(self._append_files(files))
        content_user.append({"type": "input_text", "text": prompt})

        input: list[dict[str, Any]] = []
        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                input.append(
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                )
        input.append({"role": "user", "content": content_user})

        body: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "input": input,
        }
        _ = kwargs.pop("stream", None)
        body.update(kwargs)

        if not self.supports_temperature:
            del body["temperature"]
        return body

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        if len(prompt) >= OpenAIModel.MAX_CHARACTERS:
            raise DoNotRetryException(
                f"Input is too long - OpenAI models supports a maximum of {OpenAIModel.MAX_CHARACTERS} characters"
            )

        if self.use_completions:
            return await self._query_completions(
                prompt, images=images, files=files, **kwargs
            )

        body = self.query_body(prompt, images=images, files=files, **kwargs)

        response: Response = await self.get_client().responses.create(
            **body, stream=False, timeout=OpenAIModel.TIMEOUT
        )  # type: ignore

        if (
            response.incomplete_details
            and response.incomplete_details.reason == "max_output_tokens"
        ):
            raise MaxOutputTokensExceededError()

        metadata = None
        if response.usage:
            metadata = QueryResultMetadata(
                in_tokens=response.usage.input_tokens,
                out_tokens=response.usage.output_tokens,
            )

        return response.output_text, metadata

    async def query_json(
        self,
        prompt: str,
        pydantic_model: type[PydanticT],
        **kwargs: object,
    ) -> PydanticT:
        """Query the model with JSON response format using Pydantic model.

        Args:
            prompt: The input prompt
            pydantic_model: Pydantic model class defining the expected response structure
            parameters: Additional parameters for the query
            max_tokens: Maximum number of tokens in the response

        Returns:
            Parsed Pydantic model instance

        Raises:
            ValueError: If the model response is empty, invalid, or refused
        """

        messages: list[dict[str, str]] = []
        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": OpenAIModel.TIMEOUT,
            "messages": messages,
        }
        body.update(kwargs)

        if not self.supports_temperature:
            del body["temperature"]

        response = await self.get_client().beta.chat.completions.parse(
            response_format=pydantic_model,
            **body,
        )  # type: ignore

        parsed: PydanticT | None = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Model returned empty response")

        return parsed

    async def get_embedding(self, text: str) -> list[float]:
        """Query OpenAI's Embedding endpoint"""
        try:
            response: CreateEmbeddingResponse = (
                await self.get_client().embeddings.create(
                    input=text,
                    model="text-embedding-3-small",
                )
            )
        except Exception as e:
            raise Exception("Failed to query OpenAI Embedding endpoint") from e

        if not response.data:
            raise Exception("No embeddings returned from OpenAI")

        return response.data[0].embedding

    async def moderate_content(self, text: str) -> ModerationCreateResponse:
        """Query OpenAI's Moderation endpoint"""
        try:
            return await self.get_client().moderations.create(input=text)
        except Exception as e:
            raise Exception("Failed to query OpenAI Moderation endpoint") from e
