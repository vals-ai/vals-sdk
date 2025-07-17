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
from vals.sdk.types import Metadata
from vals.sdk.util import get_ariadne_client

LIMIT = 200


class SingleRunReview(BaseModel):
    id: str
    created_by: str
    created_at: datetime
    status: RunReviewStatusEnum
    pass_rate_human_eval: float | None
    flagged_rate: float | None
    agreement_rate_auto_eval: float | None
    completed_time: datetime | None
    number_of_reviews: int
    assigned_reviewers: list[str]
    auto_eval_review: bool
    single_test_result_reviews: list["TestResult"]
    custom_review_templates: list["CustomReviewTemplate"] | None

    @classmethod
    async def from_id(cls, id: str) -> "SingleRunReview":
        client = get_ariadne_client()
        run_review_query = await client.get_single_run_review(run_review_id=id)
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
            test_result_reviews_with_count.test_result_reviews_with_count.single_test_results,
            run_review.rereview_auto_eval,
        )

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
                test_result_reviews_with_count.test_result_reviews_with_count.single_test_results,
                run_review.rereview_auto_eval,
            )

        if len(run_review.assigned_reviewers) == 0:  # -> default for all users selected
            user_options_query = await client.get_user_options()
            run_review.assigned_reviewers = user_options_query.user_emails

        return cls(
            id=run_review.id,
            created_by=run_review.created_by,
            created_at=run_review.created_at,
            status=run_review.status,
            pass_rate_human_eval=run_review.pass_rate_human_eval or None,
            flagged_rate=run_review.flagged_rate or None,
            agreement_rate_auto_eval=run_review.agreement_rate_auto_eval or None,
            completed_time=run_review.completed_time or None,
            number_of_reviews=run_review.number_of_reviews,
            assigned_reviewers=run_review.assigned_reviewers,
            auto_eval_review=run_review.rereview_auto_eval,
            single_test_result_reviews=test_result_reviews,
            custom_review_templates=(
                custom_review_templates if len(custom_review_templates) > 0 else None
            ),
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
    base: dict[str, Any] | None
    comparative: bool
    result_a: dict[str, Any] | None
    result_b: dict[str, Any] | None


class AutoEvalReview(BaseModel):
    criteria: str
    operator: str
    check_value: float


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
    auto_eval_review_values: list[AutoEvalReviewValue] | None
    custom_review_values: list[CustomReviewValue] | None


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
    metadata: Metadata
    latest_completed_review: datetime
    auto_eval_values: list[AutoEvalReview]
    aggregated_custom_metrics: list[AggregatedCustomMetric]
    single_test_result_reviews: list[SingleTestResultReview]


def create_single_test_result_reviews(
    test_result_reviews: list[
        SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResults
    ],
    auto_eval_review: bool,
) -> list[TestResult]:
    results: list[TestResult] = []
    for test_result_review in test_result_reviews:
        test_result = create_test_result(test_result_review)

        single_test_reviews: list[SingleTestResultReview] = []
        for single_test_review in test_result_review.single_test_reviews:
            if (
                single_test_review is None
                or single_test_review.status != TestResultReviewStatusEnum.COMPLETED
            ):
                continue

            review = create_single_test_review(single_test_review)

            custom_review_values: list[CustomReviewValue] = []

            if len(single_test_review.custom_review_values) > 0:
                for custom_review_value in single_test_review.custom_review_values:
                    custom_review_values.append(
                        CustomReviewValue(
                            template=CustomReviewTemplate(
                                **custom_review_value.template.model_dump()
                            ),
                            value=custom_review_value.value,
                        )
                    )

            review.custom_review_values = (
                custom_review_values if len(custom_review_values) > 0 else None
            )

            auto_eval_review_values: list[AutoEvalReviewValue] = []

            if auto_eval_review:
                for (
                    auto_eval_review_value,
                    auto_eval_value,
                ) in zip(
                    single_test_review.per_check_test_review_typed,
                    test_result.auto_eval_values,
                ):
                    auto_eval_review_values.append(
                        AutoEvalReviewValue(
                            auto_eval=auto_eval_value,
                            human_eval=auto_eval_review_value.binary_human_eval,
                            is_flagged=(
                                auto_eval_review_value.is_flagged
                                if auto_eval_review_value.is_flagged is not None
                                else False
                            ),
                        )
                    )

            review.auto_eval_review_values = (
                auto_eval_review_values if len(auto_eval_review_values) > 0 else None
            )

            single_test_reviews.append(review)

        test_result.single_test_result_reviews = single_test_reviews
        results.append(test_result)

    return results


def create_test_result(
    test_result_review: SingleTestResultReviewsWithCountTestResultReviewsWithCountSingleTestResults,
) -> TestResult:
    metadata = test_result_review.typed_metadata

    metadata_value = Metadata(
        in_tokens=metadata.in_tokens,
        out_tokens=metadata.out_tokens,
        duration_seconds=metadata.duration_seconds,
    )

    aggregated_metrics: list[AggregatedCustomMetric] = []

    metrics_to_process = test_result_review.aggregated_custom_metrics or []

    for metric in metrics_to_process:
        metric_dict = {
            "name": metric.name,
            "type": metric.type,
            "base": metric.base.model_dump() if metric.base else {},
            "comparative": metric.comparative,
            "result_a": metric.result_a.model_dump() if metric.result_a else {},
            "result_b": metric.result_b.model_dump() if metric.result_b else {},
        }

        aggregated_metrics.append(AggregatedCustomMetric(**metric_dict))

    auto_eval_results: list[AutoEvalReview] = []
    for auto_eval_result in test_result_review.typed_result_json:
        auto_eval_results.append(
            AutoEvalReview(
                criteria=auto_eval_result.criteria,
                operator=auto_eval_result.operator,
                check_value=auto_eval_result.auto_eval,
            )
        )

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
        aggregated_custom_metrics=aggregated_metrics,
        single_test_result_reviews=[],
        auto_eval_values=auto_eval_results,
        input_under_test=test_result_review.test.input_under_test,
        context=test_result_review.test.typed_context,
        output_context=test_result_review.qa_pair.output_context,
        metadata=metadata_value,
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
        auto_eval_review_values=[],
    )
