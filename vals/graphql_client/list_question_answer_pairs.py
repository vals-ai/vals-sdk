# Generated by ariadne-codegen
# Source: vals/graphql/

from datetime import datetime
from typing import Any, List

from pydantic import Field

from .base_model import BaseModel


class ListQuestionAnswerPairs(BaseModel):
    question_answer_pairs_with_count: (
        "ListQuestionAnswerPairsQuestionAnswerPairsWithCount"
    ) = Field(alias="questionAnswerPairsWithCount")


class ListQuestionAnswerPairsQuestionAnswerPairsWithCount(BaseModel):
    question_answer_pairs: List[
        "ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairs"
    ] = Field(alias="questionAnswerPairs")
    count: int


class ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairs(BaseModel):
    id: Any
    input_under_test: str = Field(alias="inputUnderTest")
    llm_output: str = Field(alias="llmOutput")
    context: Any
    output_context: Any = Field(alias="outputContext")
    metadata: (
        "ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsMetadata"
    )
    file_ids: List[str] = Field(alias="fileIds")
    local_evals: List[
        "ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsLocalEvals"
    ] = Field(alias="localEvals")
    test: "ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsTest"


class ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsMetadata(
    BaseModel
):
    in_tokens: int = Field(alias="inTokens")
    out_tokens: int = Field(alias="outTokens")
    duration_seconds: float = Field(alias="durationSeconds")


class ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsLocalEvals(
    BaseModel
):
    id: str
    score: float
    feedback: str
    created_at: datetime = Field(alias="createdAt")


class ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairsTest(
    BaseModel
):
    id: str


ListQuestionAnswerPairs.model_rebuild()
ListQuestionAnswerPairsQuestionAnswerPairsWithCount.model_rebuild()
ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairs.model_rebuild()
