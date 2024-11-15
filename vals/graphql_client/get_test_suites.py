# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any, List

from pydantic import Field

from .base_model import BaseModel


class GetTestSuites(BaseModel):
    test_suites: List["GetTestSuitesTestSuites"] = Field(alias="testSuites")


class GetTestSuitesTestSuites(BaseModel):
    description: str
    id: str
    org: str
    title: str
    created: Any
    creator: str


GetTestSuites.model_rebuild()
