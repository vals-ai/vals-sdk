import json
from datetime import datetime

from pydantic import BaseModel
from vals.graphql_client.enums import (
    RunReviewStatusEnum,
    TemplateType,
    TestResultReviewStatusEnum,
)
from vals.sdk.types import TestResult
from vals.sdk.util import get_ariadne_client


class SingleRunReview(BaseModel):
    id: str
    created_by: str
    created_at: datetime
    status: RunReviewStatusEnum
    pass_rate: float | None
    flagged_rate: float | None
    agreement_rate: float | None
    completed_time: datetime | None
    number_of_reviews: int
    assigned_reviewers: list[str]
    rereview_auto_eval: bool
    single_test_result_reviews: list["SingleTestResultReview"]
    custom_review_templates: list["CustomReviewTemplate"]

    @classmethod
    async def from_id(cls, id: str) -> "SingleRunReview":
        # Human review functionality is not implemented in this version
        # This is a placeholder to maintain API compatibility
        raise NotImplementedError("Human review functionality is not available")


class CustomReviewTemplate(BaseModel):
    id: str
    name: str
    instructions: str
    categories: list[str] | None
    type: TemplateType
    min_value: int | None
    max_value: int | None
    optional: bool


class CustomReviewValue(BaseModel):
    template: CustomReviewTemplate
    value: str


class SingleTestResultReview(BaseModel):
    id: str
    agreement_rate: float
    pass_percentage: float
    feedback: str
    completed_by: str
    completed_at: datetime
    started_at: datetime
    created_by: str
    status: TestResultReviewStatusEnum
    test_result: TestResult
    custom_review_values: list[CustomReviewValue]