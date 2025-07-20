from typing import Literal

from openai.lib.azure import AsyncAzureOpenAI
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import LLMConfig

from .openai import OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    _client: AsyncAzureOpenAI | None = None

    @override
    def get_client(self) -> AsyncAzureOpenAI:
        if not AzureOpenAIModel._client:
            AzureOpenAIModel._client = AsyncAzureOpenAI(
                api_key=sdk.model_library_settings.AZURE_KEY,
                azure_endpoint="https://playgroundrl.openai.azure.com/",
                api_version="2025-03-01-preview",
            )
        return AzureOpenAIModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["azure"] = "azure",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(
            model_name=model_name,
            provider=provider,
            config=config,
        )
