from typing import Any, Literal, cast

from ai21 import AsyncAI21Client
from ai21.models.chat import ChatMessage
from ai21.models.chat.chat_completion_response import ChatCompletionResponse
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import MaxOutputTokensExceededError


class AI21LabsModel(LLM):
    _client: AsyncAI21Client | None = None

    @override
    def get_client(self) -> AsyncAI21Client:
        if not AI21LabsModel._client:
            AI21LabsModel._client = AsyncAI21Client(
                api_key=sdk.model_library_settings.AI21LABS_KEY,
            )
        return AI21LabsModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["ai21labs"] = "ai21labs",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        messages: list[ChatMessage] = []

        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model": self.model_name,
            "messages": messages,
        }
        body.update(kwargs)

        response: ChatCompletionResponse = (
            await self.get_client().chat.completions.create(**body, stream=False)
        )

        if not response or not response.choices or not response.choices[0].message:
            raise Exception("Model returned no completions")

        if response.choices[0].finish_reason == "length":
            raise MaxOutputTokensExceededError()

        metadata = None
        if response.usage:
            metadata = QueryResultMetadata(
                in_tokens=response.usage.prompt_tokens,
                out_tokens=response.usage.completion_tokens,
            )
        text = cast(str, response.choices[0].message.content)

        return text, metadata
