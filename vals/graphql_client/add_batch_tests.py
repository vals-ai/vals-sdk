# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import List, Optional

from pydantic import Field

from .base_model import BaseModel


class AddBatchTests(BaseModel):
    batch_update_test: Optional["AddBatchTestsBatchUpdateTest"] = Field(
        alias="batchUpdateTest"
    )


class AddBatchTestsBatchUpdateTest(BaseModel):
    tests: List["AddBatchTestsBatchUpdateTestTests"]


class AddBatchTestsBatchUpdateTestTests(BaseModel):
    test_id: str = Field(alias="testId")


AddBatchTests.model_rebuild()
AddBatchTestsBatchUpdateTest.model_rebuild()
