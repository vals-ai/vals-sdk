import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from vals.graphql_client.enums import (
    RunReviewStatusEnum,
    TemplateType,
    TestResultReviewStatusEnum,
)
from vals.graphql_client.input_types import TestReviewFilterOptionsInput
from vals.graphql_client.single_test_result_reviews_with_count import (
    SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResults,
    SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResultsSingleTestReviews,
)
from vals.sdk.types import Metadata, Test, TestResult
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

        test_result_reviews_with_count = (
            await client.single_test_result_reviews_with_count(
                run_id=run_review.run.id,
                filter_options=TestReviewFilterOptionsInput(
                    status=TestResultReviewStatusEnum.COMPLETED,
                    limit=LIMIT,
                    offset=offset,
                ),
            )
        )

        total_test_result_reviews_in_run = (
            test_result_reviews_with_count.test_result_reviews_with_count.count
        )

        test_result_reviews = create_single_test_result_reviews(
            test_result_reviews_with_count.test_result_reviews_with_count.single_test_results
        )

        if len(test_result_reviews) < total_test_result_reviews_in_run:
            while len(test_result_reviews) < total_test_result_reviews_in_run:
                offset += LIMIT
                test_result_reviews_with_count = (
                    await client.single_test_result_reviews_with_count(
                        run_id=run_review.run.id,
                        filter_options=TestReviewFilterOptionsInput(
                            status=TestResultReviewStatusEnum.COMPLETED,
                            limit=LIMIT,
                            offset=offset,
                        ),
                    )
                )

                test_result_reviews += create_single_test_result_reviews(
                    test_result_reviews_with_count.test_result_reviews_with_count.single_test_results
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
            single_test_result_reviews=test_result_reviews,
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


class AggregatedCustomMetric(BaseModel):
    name: str
    type: TemplateType
    displayed: bool
    instructions: str
    values: list[str]
    max: int


class AutoEvalReview(BaseModel):
    criteria: str
    operator: str
    check_value: str


class AutoEvalReviewValue(BaseModel):
    auto_eval: AutoEvalReview
    human_eval: int
    is_flagged: bool


class SingleTestResultReview(BaseModel):
    id: str
    feedback: str
    completed_by: str
    completed_at: datetime
    started_at: datetime
    created_by: str
    status: TestResultReviewStatusEnum
    auto_eval_review_values: list[AutoEvalReviewValue]
    custom_review_values: list[CustomReviewValue]


class TestResult(BaseModel):
    id: str
    reviewed_by: list[str]
    has_feedback: bool
    agreement_rate_auto_eval: float
    agreement_rate_human_eval: float
    pass_rate_human_eval: float
    pass_percentage: float
    amount_reviewed: int
    llm_output: str
    input_under_test: str
    context: dict[str, Any]
    output_context: dict[str, Any]
    metadata: Metadata | None
    latest_completed_review: datetime
    auto_eval_values: list[AutoEvalReview]
    aggregated_custom_metrics: list[AggregatedCustomMetric]
    single_test_result_reviews: list[SingleTestResultReview]


def create_single_test_result_reviews(
    test_result_reviews: list[
        SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResults
    ],
) -> list[TestResult]:
    single_test_result_reviews = []
    for test_result_review in test_result_reviews:

        test_result = create_test_result(test_result_review)

        single_test_result_reviews = []

        for single_test_review in test_result_review.single_test_reviews:
            single_test_result_reviews.append(
                create_single_test_review(single_test_review)
            )

            custom_review_values = []
            for custom_review_value in single_test_review.custom_review_values:
                custom_review_values.append(
                    CustomReviewValue(
                        template=CustomReviewTemplate(
                            **custom_review_value.template.model_dump()
                        ),
                        value=custom_review_value.value,
                    )
                )

            single_test_review.custom_review_values = custom_review_values
            single_test_result_reviews.append(single_test_review)

        test_result.single_test_result_reviews = single_test_result_reviews

    return test_result


def create_test_result(
    test_result_review: SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResults,
) -> TestResult:
    return TestResult(
        id=test_result_review.id,
        reviewed_by=test_result_review.reviewed_by,
        has_feedback=test_result_review.has_feedback,
        agreement_rate_auto_eval=test_result_review.agreement_rate_auto_eval,
        agreement_rate_human_eval=test_result_review.agreement_rate_human_eval,
        pass_rate_human_eval=test_result_review.pass_rate_human_eval,
        pass_percentage=test_result_review.pass_percentage,
        amount_reviewed=test_result_review.amount_reviewed,
        latest_completed_review=test_result_review.latest_completed_review,
        aggregated_custom_metrics=test_result_review.aggregated_custom_metrics,
        single_test_result_reviews=[],
        auto_eval_values=test_result_review.typed_result_json,
        input_under_test=test_result_review.test.input_under_test,
        context=test_result_review.test.context,
        output_context=test_result_review.qa_pair.output_context,
        metadata=test_result_review.metadata,
        llm_output=test_result_review.llm_output,
    )


def create_single_test_review(
    single_test_review: SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResultsSingleTestReviews,
) -> SingleTestResultReview:
    return SingleTestResultReview(
        id=single_test_review.id,
        status=single_test_review.status,
        feedback=single_test_review.feedback,
        completed_by=single_test_review.completed_by,
        completed_at=single_test_review.completed_at,
        started_at=single_test_review.started_at,
        created_by=single_test_review.created_by,
        custom_review_values=[],
    )
