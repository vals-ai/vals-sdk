# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Optional

from pydantic import Field

from .base_model import BaseModel


class RerunTests(BaseModel):
    rerun_failing_tests: Optional["RerunTestsRerunFailingTests"] = Field(
        alias="rerunFailingTests"
    )


class RerunTestsRerunFailingTests(BaseModel):
    success: Optional[bool]


RerunTests.model_rebuild()
