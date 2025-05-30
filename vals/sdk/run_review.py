import json
from datetime import datetime

from pydantic import BaseModel
from vals.graphql_client.enums import (
    RunReviewStatusEnum,
    TemplateType,
    TestResultReviewStatusEnum,
)
from vals.graphql_client.get_single_test_reviews_with_count import (
    GetSingleTestReviewsWithCountSingleTestReviewsWithCountSingleTestReviews,
)
from vals.graphql_client.input_types import TestReviewFilterOptionsInput
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
        client = get_ariadne_client()
        run_review_query = await client.get_single_run_review(run_review_id=id)
        LIMIT = 200
        offset = 0
        custom_review_templates = []
        run_review = run_review_query.single_run_review

        for template in run_review.custom_review_templates:
            custom_review_templates.append(
                CustomReviewTemplate(
                    id=template.id,
                    name=template.name,
                    instructions=template.instructions,
                    categories=template.categories,
                    type=template.type,
                    min_value=template.min_value,
                    max_value=template.max_value,
                    optional=template.optional,
                )
            )

        single_test_reviews_with_count = (
            await client.get_single_test_reviews_with_count(
                run_id=run_review.run.id,
                filter_options=TestReviewFilterOptionsInput(
                    status=TestResultReviewStatusEnum.COMPLETED,
                    limit=LIMIT,
                    offset=offset,
                ),
            )
        )

        total_reviews_in_run = (
            single_test_reviews_with_count.single_test_reviews_with_count.count
        )

        test_reviews = create_single_test_reviews(
            single_test_reviews_with_count.single_test_reviews_with_count.single_test_reviews
        )

        if len(test_reviews) < total_reviews_in_run:
            while len(test_reviews) < total_reviews_in_run:
                offset += LIMIT
                single_test_reviews_with_count = (
                    await client.get_single_test_reviews_with_count(
                        run_id=run_review.run.id,
                        filter_options=TestReviewFilterOptionsInput(
                            status=TestResultReviewStatusEnum.COMPLETED,
                            limit=LIMIT,
                            offset=offset,
                        ),
                    )
                )

                test_reviews += create_single_test_reviews(
                    single_test_reviews_with_count.single_test_reviews_with_count.single_test_reviews
                )

        return cls(
            id=run_review.id,
            created_by=run_review.created_by,
            created_at=run_review.created_at,
            status=run_review.status,
            pass_rate=run_review.pass_rate or None,
            flagged_rate=run_review.flagged_rate or None,
            agreement_rate=run_review.agreement_rate or None,
            completed_time=run_review.completed_time or None,
            number_of_reviews=run_review.number_of_reviews,
            assigned_reviewers=(
                run_review.assigned_reviewers if run_review.assigned_reviewers else []
            ),
            rereview_auto_eval=run_review.rereview_auto_eval,
            single_test_result_reviews=test_reviews,
            custom_review_templates=custom_review_templates,
        )


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


def create_single_test_reviews(
    test_reviews: list[
        GetSingleTestReviewsWithCountSingleTestReviewsWithCountSingleTestReviews
    ],
) -> list[SingleTestResultReview]:
    single_test_result_reviews = []
    for test_result_review in test_reviews:

        single_test_result_review = SingleTestResultReview(
            id=test_result_review.id,
            status=test_result_review.status,
            agreement_rate=test_result_review.agreement_rate,
            pass_percentage=test_result_review.pass_percentage,
            feedback=test_result_review.feedback,
            completed_by=test_result_review.completed_by,
            completed_at=test_result_review.completed_at,
            started_at=test_result_review.started_at,
            created_by=test_result_review.created_by,
            test_result=TestResult.from_graphql(test_result_review.test_result),
            custom_review_values=[],
        )

        custom_review_values = []
        for custom_review_value in test_result_review.custom_review_values:
            custom_review_values.append(
                CustomReviewValue(
                    template=CustomReviewTemplate(
                        **custom_review_value.template.model_dump()
                    ),
                    value=custom_review_value.value,
                )
            )

        single_test_result_review.custom_review_values = custom_review_values

        single_test_result_reviews.append(single_test_result_review)

    return single_test_result_reviews
