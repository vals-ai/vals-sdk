# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import List, Optional

from pydantic import Field

from .base_model import BaseModel
from .fragments import TestFragment


class AddBatchTests(BaseModel):
    batch_update_test: Optional["AddBatchTestsBatchUpdateTest"] = Field(
        alias="batchUpdateTest"
    )


class AddBatchTestsBatchUpdateTest(BaseModel):
    tests: List["AddBatchTestsBatchUpdateTestTests"]


class AddBatchTestsBatchUpdateTestTests(TestFragment):
    pass


AddBatchTests.model_rebuild()
AddBatchTestsBatchUpdateTest.model_rebuild()
