# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any

from pydantic import Field

from .base_model import BaseModel


class RunParamInfo(BaseModel):
    run_parameter_info: Any = Field(alias="runParameterInfo")
