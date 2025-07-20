import re
from typing import Any, Iterator, Literal, Sequence, cast

from mistralai import Mistral
from mistralai.models.chatcompletionresponse import ChatCompletionResponse
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
from vals.sdk.model_library.file_utils import concat_images


class MistralModel(LLM):
    _client: Mistral | None = None

    REASONING_SYSTEM_PROMPT: str = """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

    Your thinking process must follow the template below:
    <think>
    Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
    </think>

    Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

    Problem:"""

    @override
    def get_client(self) -> Mistral:
        if not MistralModel._client:
            MistralModel._client = Mistral(
                api_key=sdk.model_library_settings.MISTRAL_KEY,
            )
        return MistralModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["mistral"] = "mistral",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

    def _append_images(
        self,
        images: Sequence[FileInput],
    ) -> Iterator[dict[str, Any]]:
        """Append images to the request body"""
        if not images:
            return iter(())
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
        # mistral supports max 8 images
        if not all(isinstance(img, FileWithBase64) for img in images):
            raise TypeError("All image inputs must be of type FileWithBase64")
        images: list[FileWithBase64] = cast(list[FileWithBase64], images)
        if len(images) > 8:
            joined_image = concat_images(
                images[7:], max_height=10000, max_width=10000, resize=True
            )
            images = images[:7] + [joined_image]

        content_user: list[dict[str, Any]] = []
        content_user.extend(self._append_images(images))
        content_user.append({"type": "text", "text": prompt})

        is_magistral_medium_2506 = "magistral-medium-2506" in self.model_name

        messages: list[dict[str, Any]] = []
        if "system_prompt" in kwargs:
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                if is_magistral_medium_2506:
                    system_prompt += f"\n\n{MistralModel.REASONING_SYSTEM_PROMPT}"
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_user})

        body: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        body.update(kwargs)

        response: ChatCompletionResponse = await self.get_client().chat.complete_async(
            **body, stream=False
        )  # type: ignore

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

        if self.reasoning and metadata:
            think_content_list: list[str] = re.findall(
                r"<think>(.*?)</think>", str(text), flags=re.DOTALL
            )
            if len(think_content_list) == 1:
                think_content = think_content_list[0]
                text = re.sub(r"<think>.*?</think>\s*", "", str(text), flags=re.DOTALL)
            else:
                think_content = "Error: multiple or no reasoning tokens found"

            metadata["reasoning"] = think_content

        return str(text), metadata
