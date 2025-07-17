import inspect
from enum import Enum
from typing import Annotated, Literal, get_type_hints

import anthropic
import openai
from pydantic.config import ConfigDict
from pydantic.fields import Field
from pydantic.main import BaseModel, create_model


class SDK(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


def function_to_model(fn, model_name: str) -> type[BaseModel]:
    signature = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields = {}

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, param.annotation)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)

    return create_model(
        model_name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )


# === Create models dynamically ===
OpenAICompletionModel = function_to_model(
    openai.OpenAI(api_key="dummy").completions.create, "OpenAICompletionModel"
)
AnthropicCompletionModel = function_to_model(
    anthropic.Anthropic(api_key="dummy").completions.create, "AnthropicCompletionModel"
)


class OpenAICompletionRequest(OpenAICompletionModel):
    sdk: Literal[SDK.OPENAI] = Field(default=SDK.OPENAI)


class AnthropicCompletionRequest(AnthropicCompletionModel):
    sdk: Literal[SDK.ANTHROPIC] = Field(default=SDK.ANTHROPIC)


ValsCompletionRequest = Annotated[
    OpenAICompletionRequest | AnthropicCompletionRequest,
    Field(discriminator="sdk"),
]
