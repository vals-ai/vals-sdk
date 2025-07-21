from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic.fields import Field
from pydantic.main import BaseModel

from vals.sdk.model_library.base import LLM
from vals.sdk.model_library.openai.azure import AzureOpenAIModel
from vals.sdk.model_library.openai.openai import OpenAIModel
from vals.sdk.model_library.providers.ai21labs import AI21LabsModel
from vals.sdk.model_library.providers.amazon import AmazonModel
from vals.sdk.model_library.providers.anthropic import AnthropicModel
from vals.sdk.model_library.providers.cohere import CohereModel
from vals.sdk.model_library.providers.fireworks import FireworksModel
from vals.sdk.model_library.providers.google import GoogleModel
from vals.sdk.model_library.providers.mistral import MistralModel
from vals.sdk.model_library.providers.saga import SagaModel
from vals.sdk.model_library.providers.together import TogetherModel
from vals.sdk.model_library.providers.vals import DummyAIModel
from vals.sdk.model_library.providers.xai import XAIModel
from vals.sdk.model_library.utils import get_logger

MAPPING_PROVIDERS: dict[str, type[LLM]] = {
    "openai": OpenAIModel,
    "azure": AzureOpenAIModel,
    "saga": SagaModel,
    "anthropic": AnthropicModel,
    "together": TogetherModel,
    "mistralai": MistralModel,
    "grok": XAIModel,
    "fireworks": FireworksModel,
    "ai21labs": AI21LabsModel,
    "amazon": AmazonModel,
    "bedrock": AmazonModel,
    "cohere": CohereModel,
    "google": GoogleModel,
    "vals": DummyAIModel,
}

"""
    "vals": DummyAIModel,
"""


logger = get_logger(__name__)
path_library = Path(__file__).parent / "config"


"""
Model Registry structure
Do not set model defaults here, they should be set in the LLMConfig class
You can set metadata configs that are not passed into the LLMConfig class here, ex:
    available_for_everyone, deprecated, available_as_evaluator, etc.
"""


class Properties(BaseModel):
    context_window: int | None = None
    max_token_output: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    training_cutoff: str | None = None
    reasoning_model: bool | None = None


class ClassProperties(BaseModel):
    supports_images: bool | None = None
    supports_metadata: bool | None = None
    supports_files: bool | None = None
    supports_batch_requests: bool | None = None
    available_for_everyone: bool = True
    available_as_evaluator: bool = False
    deprecated: bool = False


class RawModelConfig(BaseModel):
    company: str
    label: str
    description: str | None = None
    release_date: date | None = None
    properties: Properties = Field(default_factory=Properties)
    class_properties: ClassProperties = Field(default_factory=ClassProperties)
    costs_per_million_tokens: dict[str, float | None] = Field(default_factory=dict)
    alternative_keys: list[str] = Field(default_factory=list)


class ModelConfig(RawModelConfig):
    # post processing fields
    provider_name: str
    provider_endpoint: str
    full_key: str
    slug: str


ModelRegistry = dict[str, ModelConfig]


def deep_update(
    base: dict[str, Any], updates: dict[str, str | dict[str, Any]]
) -> dict[str, Any]:
    """Recursively update a dictionary, merging nested dictionaries instead of replacing them."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def register_models() -> ModelRegistry:
    logger.info(f"Loading model registry from {path_library}")

    registry: ModelRegistry = {}

    sections = Path(path_library).glob("*.yaml")
    for section in sections:
        with open(section, "r") as file:
            model_blocks = cast(
                dict[str, dict[str, dict[str, Any]]] | None, yaml.safe_load(file)
            )
            if not model_blocks:
                continue

            provider_base_config = model_blocks.get("base-config", {})
            for model_block, model_data in model_blocks.items():
                if model_block == "base-config":
                    continue

                block_config = deepcopy(provider_base_config)
                if "base-config" in model_data:
                    block_config = deep_update(block_config, model_data["base-config"])

                for model_name, model_config in model_data.items():
                    if model_name == "base-config":
                        continue

                    current_model_config = deepcopy(block_config)
                    current_model_config = deep_update(
                        current_model_config, model_config
                    )

                    # create model config object
                    raw_model_obj: RawModelConfig = RawModelConfig.model_validate(
                        current_model_config
                    )

                    provider_endpoint = (
                        current_model_config.get("provider_endpoint", None)
                        or model_name.split("/", 1)[1]
                    )
                    model_obj = ModelConfig.model_validate(
                        {
                            **raw_model_obj.model_dump(),
                            "provider_name": model_name.split("/")[0],
                            "provider_endpoint": provider_endpoint,
                            "full_key": model_name,
                            "slug": model_name.replace("/", "_"),
                        }
                    )

                    registry[model_name] = model_obj

                    # add alternative keys
                    alternative_keys = cast(
                        list[str], model_config.get("alternative_keys", [])
                    )
                    for key in alternative_keys:
                        registry[key] = deepcopy(registry[model_name])

    return registry


_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = register_models()
    return _model_registry
