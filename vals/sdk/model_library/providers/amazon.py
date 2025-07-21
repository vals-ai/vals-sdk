import asyncio
import base64
from collections.abc import Iterator
from typing import Any, cast

import boto3
from botocore.client import BaseClient
from typing_extensions import override

from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import MaxOutputTokensExceededError


class AmazonModel(LLM):
    _client: BaseClient | None = None

    @override
    def get_client(self) -> BaseClient:
        if not AmazonModel._client:
            AmazonModel._client = cast(BaseClient, boto3.client("bedrock-runtime"))  # type: ignore
        return AmazonModel._client

    def _append_images(
        self,
        images: list[FileInput],
    ) -> Iterator[dict[str, Any]]:
        """Append images to the request body"""
        if not images:
            return iter(())
        for image in images:
            match image:
                case FileWithBase64():
                    image_bytes = base64.b64decode(image.base64)
                    yield {
                        "image": {
                            "format": image.mime,
                            "source": {"bytes": image_bytes},
                        },
                    }
                case _:
                    raise Exception("Unsupported image type")

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#
    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        content_user: list[dict[str, Any]] = []
        content_user.extend(self._append_images(images))
        content_user.append({"text": prompt})

        body: dict[str, Any] = {
            "modelId": self.model_name,
            "messages": [{"role": "user", "content": content_user}],
        }

        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                body["system"] = [{"text": system_prompt}]

        inference: dict[str, Any] = {
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
        }
        inference.update(kwargs)

        response: dict[str, Any] = await asyncio.to_thread(
            self.get_client().converse,
            **body,
            inferenceConfig=inference,
        )

        if response["stopReason"] == "max_tokens":
            raise MaxOutputTokensExceededError()

        text_response = response["output"]["message"]["content"][0]["text"]
        return text_response, QueryResultMetadata(
            in_tokens=response["usage"]["inputTokens"],
            out_tokens=response["usage"]["outputTokens"],
        )
