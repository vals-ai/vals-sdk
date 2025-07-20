import re
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

from openai._client import AsyncOpenAI
from together import AsyncTogether
from together.types.chat_completions import ChatCompletionResponse
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

STOP_OPTIONS = {
    # Mistral Models
    "mistralai/Mistral-7B-v0.1": [
        "</s>",
    ],
    "mistralai/Mixtral-8x7B-v0.1": [
        "</s>",
    ],
    "mistralai/Mistral-7B-Instruct-v0.1": [
        "[/INST]",
        "</s>",
    ],  # chat
    "mistralai/Mistral-7B-Instruct-v0.2": ["[/INST]", "</s>"],  # chat
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ["[/INST]", "</s>"],  # chat
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ["[/INST]", "</s>"],  # chat
    "mistralai/Mistral-7B-Instruct-v0.3": ["[/INST]", "</s>"],  # chat
    #  Llama
    "togethercomputer/llama-2-7b": ["[/INST]", "</s>"],  # non-chat
    "togethercomputer/llama-2-13b": ["[/INST]", "</s>"],  # non-chat
    "meta-llama/Llama-2-70b-hf": ["[/INST]", "</s>"],  # non-chat
    "togethercomputer/llama-2-7b-chat": ["[/INST]", "</s>"],  # chat
    "togethercomputer/llama-2-13b-chat": ["[/INST]", "</s>"],  # chat
    "togethercomputer/llama-2-70b-chat": ["[/INST]", "</s>"],  # chat
    "togethercomputer/Llama-2-7B-32K-Instruct": ["[INST]", "\n\n"],  # chat
    "meta-llama/Llama-3-70b-chat-hf": ["<|eot_id|>"],  # chat
    "meta-llama/Llama-3-8b-chat-hf": ["<|eot_id|>"],  # chat
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ["<|eot_id|>"],  # chat
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ["<|eot_id|>"],  # chat
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ["<|eot_id|>"],  # chat
    "meta-llama/Llama-Vision-Free": ["<|eot_id|>"],  # chat
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ["<|eot_id|>"],  # chat
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ["<|eot_id|>"],  # chat
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ["<|eot_id|>"],  # chat
    # Alpaca
    "togethercomputer/alpaca-7b": ["</s>", "###"],
    # Falcon
    "togethercomputer/falcon-7b": ["<|endoftext|>"],  # non-chat
    "togethercomputer/falcon-40b": ["<|endoftext|>"],  # non-chat
    "togethercomputer/falcon-7b-instruct": ["User:", "</s>"],  # chat
    "togethercomputer/falcon-40b-instruct": ["User:", "</s>"],  # chat
    # Qwen
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ["<|im_end|>"],  # chat
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ["<|im_end|>"],  # chat
    # Gemma IT
    "google/gemma-2-9b-it": ["<end_of_turn>", "<eos>"],
    "google/gemma-2-27b-it": ["<end_of_turn>", "<eos>"],
}


class TogetherModel(LLM):
    _client: AsyncTogether | None = None

    @override
    def get_client(self) -> AsyncTogether:
        if not TogetherModel._client:
            TogetherModel._client = AsyncTogether(
                api_key=sdk.model_library_settings.TOGETHER_KEY,
            )
        return TogetherModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["together"] = "together",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.custom_retrier: Callable[..., Any] | None = None

        # https://docs.together.ai/docs/openai-api-compatibility
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=AsyncOpenAI(
                    api_key=sdk.model_library_settings.TOGETHER_KEY,
                    base_url="https://api.together.xyz/v1",
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
        # docs show that we can pass in s3 location somehow
        for image in images:
            match image:
                case FileWithBase64():
                    yield {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image.mime};base64,{image.base64}"
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
            return await self.delegate.query(
                prompt, images=images, files=files, **kwargs
            )

        content_user: list[dict[str, Any]] = []

        is_nemotron_super = "nemotron-super" in self.model_name
        is_reasoning = "thinking" in self.model_name or self.reasoning

        messages: list[dict[str, Any]] = []

        # special-cases nemotron
        if is_nemotron_super:
            mode = "on" if is_reasoning else "off"
            kwargs["system_prompt"] = f"detailed thinking {mode}"
            messages.append(
                {
                    "role": "system",
                    "content": f"detailed thinking {mode}",
                }
            )

        elif "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

        content_user.extend(self._append_images(images))
        content_user.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content_user})

        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "model": self.model_name.replace("-thinking", ""),
            "messages": messages,
        }
        body.update(kwargs)

        response = await self.get_client().chat.completions.create(**body, stream=False)  # type: ignore
        response = cast(ChatCompletionResponse, response)

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

        if "-thinking" in self.model_name and metadata:
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
