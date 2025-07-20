import re
from collections.abc import Callable, Iterator
from typing import Any, Literal

from anthropic import AsyncAnthropic
from openai._client import AsyncOpenAI
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import MaxOutputTokensExceededError
from vals.sdk.model_library.openai.openai import OpenAIModel


class AnthropicModel(LLM):
    _client: AsyncAnthropic | None = None

    @override
    def get_client(self) -> AsyncAnthropic:
        if not AnthropicModel._client:
            AnthropicModel._client = AsyncAnthropic(
                api_key=sdk.model_library_settings.ANTHROPIC_KEY,
            )
        return AnthropicModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["anthropic"] = "anthropic",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.custom_retrier: Callable[..., Any] | None = None  # streams instead

        # https://docs.anthropic.com/en/api/openai-sdk
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=AsyncOpenAI(
                    api_key=sdk.model_library_settings.ANTHROPIC_KEY,
                    base_url="https://api.anthropic.com/v1/",
                ),
                use_completions=True,
            )
        )

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
                    yield {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image.mime}",
                            "data": image.base64,
                        },
                    }
                case _:
                    raise Exception("Unsupported image type")

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            self.delegate.model_name = self.model_name.replace("-thinking", "")
            return await self.delegate.query(
                prompt, images=images, files=files, **kwargs
            )

        system_prompt: str | None = None
        if "system_prompt" in kwargs:
            _system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if _system_prompt.strip():
                system_prompt = _system_prompt

        content: list[dict[str, Any]] = []
        content.extend(self._append_images(images))
        content.append({"type": "text", "text": prompt})

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        body.update(kwargs)

        max_tokens: int = body["max_tokens"]
        if self.reasoning or self.model_name.endswith("-thinking"):
            if max_tokens < 1024:
                max_tokens = 2048
            budget_tokens = int(max(max_tokens * 0.75, 1024))
            if system_prompt and system_prompt.startswith("$$$THINKING:"):
                thinking_pattern = r"\$\$\$THINKING:(\d+)\$\$\$\s*(.*)"
                match = re.match(thinking_pattern, system_prompt, re.DOTALL)
                if match:
                    system_prompt = match.group(2).strip()
                    budget_tokens = int(match.group(1))

            self.logger.debug(f"Using {budget_tokens} as reasoning token budget")

            body["temperature"] = 1
            body["model"] = self.model_name.replace("-thinking", "")
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        if system_prompt:
            body["system"] = system_prompt

        in_tokens: int = 0
        out_tokens: int = 0
        output_text: str = ""
        reasoning_text: str = ""

        async with self.get_client().messages.stream(**body) as stream:
            async for chunk in stream:
                if chunk.type == "message_stop":
                    if chunk.message.stop_reason == "max_tokens":
                        await stream.close()
                        raise MaxOutputTokensExceededError()
                elif chunk.type == "message_start":
                    in_tokens += chunk.message.usage.input_tokens
                    out_tokens += chunk.message.usage.output_tokens
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        output_text += chunk.delta.text
                    if chunk.delta.type == "thinking_delta":
                        reasoning_text += chunk.delta.thinking
                elif chunk.type == "message_delta":
                    out_tokens += chunk.usage.output_tokens

        metadata = QueryResultMetadata(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )
        if reasoning_text:
            metadata["reasoning"] = reasoning_text

        return output_text, metadata
