from pydantic import BaseModel
from vals.graphql_client.enums import TemplateType
from vals.sdk.types import RunReviewStatus, TestResult
from vals.sdk.util import (
    get_ariadne_client,
)


class SingleRunReview(BaseModel):
    id: str
    created_by: str
    created_at: str
    status: RunReviewStatus
    completed_time: str
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
            single_test_result_reviews.append(
                SingleTestResultReview(
                    id=test_result_review.id,
                    agreement_rate=test_result_review.agreement_rate,
                    pass_percentage=test_result_review.pass_percentage,
                    test_result=TestResult.from_graphql(test_result_review.test_result),
                )
            )

            custom_review_values = []
            for custom_review_value in test_result_review.custom_review_values:
                custom_review_values.append(
                    CustomReviewValue(
                        template=custom_review_value.template,
                        value=custom_review_value,
                    )
                )

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
            created_at=run_review.created_at,
            status=run_review.status,
            completed_time=run_review.completed_time,
            number_of_reviews=run_review.number_of_reviews,
            assigned_reviewers=run_review.assigned_reviewers,
        )


class SingleTestResultReview(BaseModel):
    id: str

    agreement_rate: float

    pass_percentage: float

    test_result: TestResult


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
