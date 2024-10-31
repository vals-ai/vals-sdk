# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any, List

from pydantic import Field

from .base_model import BaseModel


class GetTestSuiteData(BaseModel):
    test_suites: List["GetTestSuiteDataTestSuites"] = Field(alias="testSuites")


class GetTestSuiteDataTestSuites(BaseModel):
    description: str
    id: str
    org: str
    title: str
    created: Any
    global_checks: Any = Field(alias="globalChecks")


GetTestSuiteData.model_rebuild()
