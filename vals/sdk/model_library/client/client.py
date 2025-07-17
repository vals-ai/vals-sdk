from collections.abc import Iterable
from typing import (
    Literal,
    overload,
)

import anthropic
import openai
from anthropic.types import (
    MessageParam,
    MetadataParam,
    ModelParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUnionParam,
)
from openai._types import Headers
from openai.types import ChatModel, Metadata, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.completion_create_params import (
    Function,
    ResponseFormat,
    WebSearchOptions,
)

from vals.sdk.models.types import SDK


class Vals:
    VALS_ENDPOINT = "http://localhost:8080/completions/v1"

    def __init__(self, vals_api_key: str):
        self.api_key = vals_api_key

    def _sdk_to_request(self, sdk: SDK):
        from vals.sdk.models.types import (
            AnthropicCompletionRequest,
            OpenAICompletionRequest,
        )

        match sdk:
            case SDK.OPENAI:
                return OpenAICompletionRequest
            case SDK.ANTHROPIC:
                return AnthropicCompletionRequest
            case _:
                raise ValueError(f"Unsupported sdk: {sdk}")

    # --- Shared Implementation ---
    async def complete(
        self, sdk: Literal[SDK.OPENAI, SDK.ANTHROPIC] = SDK.OPENAI, **kwargs
    ):  # -> (
        #     ChatCompletion
        #     | openai.Stream[ChatCompletionChunk]
        #     | anthropic.types.Completion
        #     | anthropic.Stream[anthropic.types.Completion]
        # ):
        from vals.sdk.models.types import ValsCompletionRequest

        request_model: type[ValsCompletionRequest] = self._sdk_to_request(sdk)

        try:
            request = request_model(**kwargs)
        except ValidationError as e:
            raise ValueError(f"Invalid request: {e}")

        response = httpx.post(
            self.VALS_ENDPOINT,
            json=request.model_dump(),
        )

    def _request(self):
        pass

    # --- OpenAI-compatible overload ---
    # --- stream = False ---
    @overload
    async def complete(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: ChatModel | str,
        audio: ChatCompletionAudioParam | openai.NotGiven | None = openai.NOT_GIVEN,
        frequency_penalty: float | openai.NotGiven | None = openai.NOT_GIVEN,
        function_call: FunctionCall | openai.NotGiven = openai.NOT_GIVEN,
        functions: Iterable[Function] | openai.NotGiven = openai.NOT_GIVEN,
        logit_bias: dict[str, int] | openai.NotGiven | None = openai.NOT_GIVEN,
        logprobs: bool | openai.NotGiven | None = openai.NOT_GIVEN,
        max_completion_tokens: int | openai.NotGiven | None = openai.NOT_GIVEN,
        max_tokens: int | openai.NotGiven | None = openai.NOT_GIVEN,
        metadata: Metadata | openai.NotGiven | None = openai.NOT_GIVEN,
        modalities: list[Literal["text", "audio"]]
        | openai.NotGiven
        | None = openai.NOT_GIVEN,
        n: int | openai.NotGiven | None = openai.NOT_GIVEN,
        parallel_tool_calls: bool | openai.NotGiven = openai.NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam
        | openai.NotGiven
        | None = openai.NOT_GIVEN,
        presence_penalty: float | openai.NotGiven | None = openai.NOT_GIVEN,
        reasoning_effort: ReasoningEffort | openai.NotGiven = openai.NOT_GIVEN,
        response_format: ResponseFormat | openai.NotGiven = openai.NOT_GIVEN,
        seed: int | openai.NotGiven | None = openai.NOT_GIVEN,
        service_tier: openai.NotGiven
        | Literal["auto", "default", "flex", "scale", "priority"]
        | None = openai.NOT_GIVEN,
        stop: str | list[str] | openai.NotGiven | None = openai.NOT_GIVEN,
        store: bool | openai.NotGiven | None = openai.NOT_GIVEN,
        stream: openai.NotGiven | Literal[False] | None = openai.NOT_GIVEN,
        temperature: float | openai.NotGiven | None = openai.NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | openai.NotGiven = openai.NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | openai.NotGiven = openai.NOT_GIVEN,
        top_logprobs: int | openai.NotGiven | None = openai.NOT_GIVEN,
        top_p: float | openai.NotGiven | None = openai.NOT_GIVEN,
        user: str | openai.NotGiven = openai.NOT_GIVEN,
        web_search_options: WebSearchOptions | openai.NotGiven = openai.NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: openai._types.Query | None = None,
        extra_body: openai._types.Body | None = None,
        timeout: float | openai.Timeout | openai.NotGiven | None = openai.NOT_GIVEN,
    ) -> ChatCompletion: ...

    # --- stream = True ---
    @overload
    async def complete(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: ChatModel | str,
        audio: ChatCompletionAudioParam | openai.NotGiven | None = openai.NOT_GIVEN,
        frequency_penalty: float | openai.NotGiven | None = openai.NOT_GIVEN,
        function_call: FunctionCall | openai.NotGiven = openai.NOT_GIVEN,
        functions: Iterable[Function] | openai.NotGiven = openai.NOT_GIVEN,
        logit_bias: dict[str, int] | openai.NotGiven | None = openai.NOT_GIVEN,
        logprobs: bool | openai.NotGiven | None = openai.NOT_GIVEN,
        max_completion_tokens: int | openai.NotGiven | None = openai.NOT_GIVEN,
        max_tokens: int | openai.NotGiven | None = openai.NOT_GIVEN,
        metadata: Metadata | openai.NotGiven | None = openai.NOT_GIVEN,
        modalities: list[Literal["text", "audio"]]
        | openai.NotGiven
        | None = openai.NOT_GIVEN,
        n: int | openai.NotGiven | None = openai.NOT_GIVEN,
        parallel_tool_calls: bool | openai.NotGiven = openai.NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam
        | openai.NotGiven
        | None = openai.NOT_GIVEN,
        presence_penalty: float | openai.NotGiven | None = openai.NOT_GIVEN,
        reasoning_effort: ReasoningEffort | openai.NotGiven = openai.NOT_GIVEN,
        response_format: ResponseFormat | openai.NotGiven = openai.NOT_GIVEN,
        seed: int | openai.NotGiven | None = openai.NOT_GIVEN,
        service_tier: openai.NotGiven
        | Literal["auto", "default", "flex", "scale", "priority"]
        | None = openai.NOT_GIVEN,
        stop: str | list[str] | openai.NotGiven | None = openai.NOT_GIVEN,
        store: bool | openai.NotGiven | None = openai.NOT_GIVEN,
        stream: Literal[True],
        stream_options: ChatCompletionStreamOptionsParam
        | openai.NotGiven
        | None = openai.NOT_GIVEN,
        temperature: float | openai.NotGiven | None = openai.NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam
        | openai.NotGiven = openai.NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | openai.NotGiven = openai.NOT_GIVEN,
        top_logprobs: int | openai.NotGiven | None = openai.NOT_GIVEN,
        top_p: float | openai.NotGiven | None = openai.NOT_GIVEN,
        user: str | openai.NotGiven = openai.NOT_GIVEN,
        web_search_options: WebSearchOptions | openai.NotGiven = openai.NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: anthropic._types.Query | None = None,
        extra_body: anthropic._types.Body | None = None,
        timeout: float | openai.Timeout | openai.NotGiven | None = openai.NOT_GIVEN,
    ) -> openai.Stream[ChatCompletionChunk]: ...

    # --- Anthropic-compatible overload ---
    # --- stream = False ---
    @overload
    async def complete(
        self,
        *,
        sdk: Literal[SDK.ANTHROPIC] = SDK.ANTHROPIC,
        max_tokens: int,
        messages: Iterable[MessageParam],
        model: ModelParam,
        metadata: MetadataParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        service_tier: anthropic.NotGiven
        | Literal["auto", "standard_only"] = anthropic.NOT_GIVEN,
        stop_sequences: list[str] | anthropic.NotGiven = anthropic.NOT_GIVEN,
        stream: anthropic.NotGiven | Literal[False] = anthropic.NOT_GIVEN,
        system: str
        | Iterable[TextBlockParam]
        | anthropic.NotGiven = anthropic.NOT_GIVEN,
        temperature: float | anthropic.NotGiven = anthropic.NOT_GIVEN,
        thinking: ThinkingConfigParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        tool_choice: ToolChoiceParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        tools: Iterable[ToolUnionParam] | anthropic.NotGiven = anthropic.NOT_GIVEN,
        top_k: int | anthropic.NotGiven = anthropic.NOT_GIVEN,
        top_p: float | anthropic.NotGiven = anthropic.NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: anthropic._types.Query | None = None,
        extra_body: anthropic._types.Body | None = None,
        timeout: float
        | anthropic.Timeout
        | anthropic.NotGiven
        | None = anthropic.NOT_GIVEN,
    ) -> anthropic.types.Completion: ...

    # --- stream = True ---
    @overload
    async def complete(
        self,
        *,
        sdk: Literal[SDK.ANTHROPIC] = SDK.ANTHROPIC,
        max_tokens: int,
        messages: Iterable[MessageParam],
        model: ModelParam,
        metadata: MetadataParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        service_tier: anthropic.NotGiven
        | Literal["auto", "standard_only"] = anthropic.NOT_GIVEN,
        stop_sequences: list[str] | anthropic.NotGiven = anthropic.NOT_GIVEN,
        stream: Literal[True],
        system: str
        | Iterable[TextBlockParam]
        | anthropic.NotGiven = anthropic.NOT_GIVEN,
        temperature: float | anthropic.NotGiven = anthropic.NOT_GIVEN,
        thinking: ThinkingConfigParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        tool_choice: ToolChoiceParam | anthropic.NotGiven = anthropic.NOT_GIVEN,
        tools: Iterable[ToolUnionParam] | anthropic.NotGiven = anthropic.NOT_GIVEN,
        top_k: int | anthropic.NotGiven = anthropic.NOT_GIVEN,
        top_p: float | anthropic.NotGiven = anthropic.NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: anthropic._types.Query | None = None,
        extra_body: anthropic._types.Body | None = None,
        timeout: float
        | anthropic.Timeout
        | anthropic.NotGiven
        | None = anthropic.NOT_GIVEN,
    ) -> anthropic.Stream[anthropic.types.Completion]: ...
