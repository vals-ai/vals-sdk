from typing import Literal

from openai import AsyncOpenAI
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import LLM, FileInput, LLMConfig, QueryResult
from vals.sdk.model_library.openai.openai import OpenAIModel


class FireworksModel(LLM):
    @override
    def get_client(self) -> None:
        raise NotImplementedError("Not implemented")

    def __init__(
        self,
        model_name: str,
        provider: Literal["fireworks"] = "fireworks",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.model_name: str = "accounts/fireworks/models/" + self.model_name
        self.native: bool = False

        # https://docs.fireworks.ai/tools-sdks/openai-compatibility
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=self.model_name,
                provider=provider,
                config=config,
                custom_client=AsyncOpenAI(
                    api_key=sdk.model_library_settings.FIREWORKS_KEY,
                    base_url="https://api.fireworks.ai/inference/v1",
                ),
                use_completions=True,
            )
        )

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
            return await self.delegate.query(
                prompt, images=images, files=files, **kwargs
            )
        # TODO: fireworks has an sdk
        raise Exception("Not implemented")
