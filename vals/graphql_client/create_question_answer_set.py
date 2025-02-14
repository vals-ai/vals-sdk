# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any, Optional

from pydantic import Field

from .base_model import BaseModel


class CreateQuestionAnswerSet(BaseModel):
    create_question_answer_set: Optional[
        "CreateQuestionAnswerSetCreateQuestionAnswerSet"
    ] = Field(alias="createQuestionAnswerSet")


class CreateQuestionAnswerSetCreateQuestionAnswerSet(BaseModel):
    question_answer_set: (
        "CreateQuestionAnswerSetCreateQuestionAnswerSetQuestionAnswerSet"
    ) = Field(alias="questionAnswerSet")
    run_id: Optional[str] = Field(alias="runId")


class CreateQuestionAnswerSetCreateQuestionAnswerSetQuestionAnswerSet(BaseModel):
    id: Any


CreateQuestionAnswerSet.model_rebuild()
CreateQuestionAnswerSetCreateQuestionAnswerSet.model_rebuild()
