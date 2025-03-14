# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Optional

from pydantic import Field

from .base_model import BaseModel


class RemoveOldTests(BaseModel):
    remove_unused_tests: Optional["RemoveOldTestsRemoveUnusedTests"] = Field(
        alias="removeUnusedTests"
    )


class RemoveOldTestsRemoveUnusedTests(BaseModel):
    success: bool


RemoveOldTests.model_rebuild()
