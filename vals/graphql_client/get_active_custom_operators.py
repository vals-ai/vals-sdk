# Generated by ariadne-codegen
# Source: vals/graphql/

from datetime import datetime
from typing import List

from pydantic import Field

from .base_model import BaseModel


class GetActiveCustomOperators(BaseModel):
    custom_operators: List["GetActiveCustomOperatorsCustomOperators"] = Field(
        alias="customOperators"
    )


class GetActiveCustomOperatorsCustomOperators(BaseModel):
    id: str
    name: str
    prompt: str
    is_unary: bool = Field(alias="isUnary")
    created_by: str = Field(alias="createdBy")
    created_at: datetime = Field(alias="createdAt")
    archived: bool


GetActiveCustomOperators.model_rebuild()
