import json
from pydantic import BaseModel
from vals.graphql_client.enums import TemplateType
from vals.sdk.types import TestResult
from vals.sdk.util import (
    get_ariadne_client,
)


class SingleRunReview(BaseModel):
    id: str
    created_by: str
    created_at: str
    status: str
    pass_rate: float
    flagged_rate: float
    agreement_rate: float
    completed_time: str | None
    number_of_reviews: int
    assigned_reviewers: list[str]
    rereview_auto_eval: bool
    single_test_result_reviews: list["SingleTestResultReview"]
    custom_review_templates: list["CustomReviewTemplate"]

    @classmethod
    async def from_id(cls, id: str) -> "SingleRunReview":
        client = get_ariadne_client()
        run_review_query = await client.get_single_run_review(id)

        # Using a multi run review query to get the single run review
        run_review = run_review_query.single_run_reviews_with_count.single_run_reviews[
            0
        ]

        single_test_result_reviews = []
        for test_result_review in run_review.singletestresultreview_set:

            single_test_result_review = SingleTestResultReview(
                id=test_result_review.id,
                status=test_result_review.status.value.lower(),
                agreement_rate=test_result_review.agreement_rate,
                pass_percentage=test_result_review.pass_percentage,
                feedback=test_result_review.feedback,
                completed_by=test_result_review.completed_by or None,
                completed_at=str(test_result_review.completed_at) or None,
                started_at=str(test_result_review.started_at),
                created_by=test_result_review.created_by,
                locked_by=test_result_review.locked_by or None,
                last_heartbeat_at=str(test_result_review.last_heartbeat_at) or None,
                last_active_at=str(test_result_review.last_active_at) or None,
                test_result=TestResult.from_graphql(test_result_review.test_result),
                custom_review_values=[],
            )

            custom_review_values = []
            for custom_review_value in test_result_review.custom_review_values:
                custom_review_values.append(
                    CustomReviewValue(
                        template=custom_review_value.template,
                        value=custom_review_value.value,
                    )
                )

            single_test_result_review.custom_review_values = custom_review_values

            single_test_result_reviews.append(single_test_result_review)

        custom_review_templates = []
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
                )
            )

        return cls(
            id=run_review.id,
            created_by=run_review.created_by,
            created_at=str(run_review.created_at),
            status=run_review.status.value.lower(),
            pass_rate=run_review.pass_rate,
            flagged_rate=run_review.flagged_rate,
            agreement_rate=run_review.agreement_rate,
            completed_time=str(run_review.completed_time) or None,
            number_of_reviews=run_review.number_of_reviews,
            assigned_reviewers=json.loads(run_review.assigned_reviewers),
            rereview_auto_eval=run_review.rereview_auto_eval,
            single_test_result_reviews=single_test_result_reviews,
            custom_review_templates=custom_review_templates,
        )


class SingleTestResultReview(BaseModel):
    id: str

    agreement_rate: float

    pass_percentage: float

    feedback: str

    completed_by: str | None

    completed_at: str | None

    started_at: str

    created_by: str

    status: str

    locked_by: str | None

    last_heartbeat_at: str | None

    last_active_at: str | None

    test_result: TestResult

    custom_review_values: list["CustomReviewValue"]


class CustomReviewTemplate(BaseModel):
    id: str
    name: str
    instructions: str
    categories: list[str]
    type: TemplateType
    min_value: int
    max_value: int


class Template(BaseModel):
    id: str
    name: str
    instructions: str
    categories: list[str]
    type: str
    min_value: float
    max_value: float


class CustomReviewValue(BaseModel):
    template: CustomReviewTemplate
    value: str
