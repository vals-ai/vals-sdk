from collections.abc import Iterator
from typing import Any, Literal

from openai import AsyncOpenAI
from typing_extensions import override
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import Content, image, system, user

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.openai.openai import OpenAIModel


class XAIModel(LLM):
    _client: AsyncClient | None = None

    @override
    def get_client(self) -> AsyncClient:
        if not XAIModel._client:
            XAIModel._client = AsyncClient(
                api_key=sdk.model_library_settings.XAI_KEY,
            )
        return XAIModel._client

    @override
    def __init__(
        self,
        model_name: str,
        provider: Literal["xai"] = "xai",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=AsyncOpenAI(
                    api_key=sdk.model_library_settings.XAI_KEY,
                    base_url=(
                        "https://us-west-1.api.x.ai/v1"
                        if "grok-3-mini-reasoning" in self.model_name
                        else "https://api.x.ai/v1"
                    ),
                ),
                use_completions=True,
            )
        )

    def _append_images(
        self,
        images: list[FileInput],
    ) -> Iterator[Content]:
        """Append images to the request body"""
        if not images:
            return iter(())
        for _image in images:
            match _image:
                case FileWithBase64():
                    image_url = f"data:image/{_image.mime};base64,{_image.base64}"
                    yield image(image_url=image_url, detail="high")
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
        self.logger.info(f"Model name: {self.model_name}")
        if "high-reasoning" in self.model_name:
            kwargs["reasoning_effort"] = "high"
        elif "low-reasoning" in self.model_name:
            self.logger.info("Using low-reasoning model")
            self.logger.info(self.model_name)
            kwargs["reasoning_effort"] = "low"
        # TODO: model names should not have this, just pass in reasoning as **kwargs
        new_name = self.model_name.replace("-high-reasoning", "")
        new_name = new_name.replace("-low-reasoning", "")
        self.model_name = new_name
        self.logger.info(f"New model name: {self.model_name}")

        if self.delegate:
            self.delegate.model_name = new_name
            return await self.delegate.query(
                prompt, images=images, files=files, **kwargs
            )

        system_prompt: str | None = None
        if "system_prompt" in kwargs:
            _system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if _system_prompt.strip():
                system_prompt = _system_prompt

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "model": self.model_name,
        }
        body.update(kwargs)

        chat: Chat = self.get_client().chat.create(
            **body,
        )  # type: ignore

        if system_prompt:
            _ = chat.append(system(system_prompt))

        _ = chat.append(user(prompt, *self._append_images(images)))

        output_text = ""
        reasoning_text: str = ""
        latest_response = None
        async for response, chunk in chat.stream():
            if chunk.content:
                output_text += chunk.content
            if chunk.reasoning_content:
                reasoning_text += chunk.reasoning_content

            latest_response = response
        if not latest_response:
            raise Exception("Model failed to produce a response")

        metadata = None
        metadata = QueryResultMetadata(
            in_tokens=latest_response.usage.prompt_tokens,
            out_tokens=latest_response.usage.total_tokens,
        )
        if reasoning_text:
            metadata["reasoning"] = reasoning_text

        return output_text, metadata
