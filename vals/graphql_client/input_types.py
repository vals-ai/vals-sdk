# Generated by ariadne-codegen
# Source: http://localhost:8000/graphql/

from typing import Any, List, Optional

from pydantic import Field

from .base_model import BaseModel
from .enums import (
    ReviewTableSortField,
    RunHumanReviewStatusEnum,
    RunResultSortField,
    RunStatus,
    SortOrder,
    TestSuiteSortField,
)


class ReviewTableFilterOptionsInput(BaseModel):
    limit: Optional[int] = None
    offset: Optional[int] = None
    search: Optional[str] = None
    sort_by: Optional[ReviewTableSortField] = Field(alias="sortBy", default=None)
    sort_order: Optional[SortOrder] = Field(alias="sortOrder", default=None)
    status: Optional[RunHumanReviewStatusEnum] = None


class FilterOptionsInput(BaseModel):
    limit: Optional[int] = None
    offset: Optional[int] = None
    search: Optional[str] = None
    folder_id: Optional[str] = Field(alias="folderId", default=None)
    test_suite_id: Optional[str] = Field(alias="testSuiteId", default=None)
    sort_by: Optional[TestSuiteSortField] = Field(alias="sortBy", default=None)
    sort_order: Optional[SortOrder] = Field(alias="sortOrder", default=None)


class RunResultFilterOptionsInput(BaseModel):
    limit: Optional[int] = None
    offset: Optional[int] = None
    search: Optional[str] = None
    status: Optional[List[Optional[RunStatus]]] = None
    archived: Optional[bool] = None
    run_by: Optional[List[str]] = Field(alias="runBy", default=None)
    models_under_test: Optional[List[str]] = Field(
        alias="modelsUnderTest", default=None
    )
    suite_id: Optional[str] = Field(alias="suiteId", default=None)
    suite_ids: Optional[List[str]] = Field(alias="suiteIds", default=None)
    sort_by: Optional[RunResultSortField] = Field(alias="sortBy", default=None)
    sort_order: Optional[SortOrder] = Field(alias="sortOrder", default=None)


class CheckInputType(BaseModel):
    operator: str
    criteria: Optional[str] = None
    modifiers: Optional["CheckModifiersInputType"] = None


class CheckModifiersInputType(BaseModel):
    optional: bool
    severity: Optional[float] = None
    examples: List["ExampleInputType"]
    extractor: Optional[str] = None
    conditional: Optional["ConditionalCheckInputType"] = None
    category: Optional[str] = None


class ExampleInputType(BaseModel):
    type: str
    text: str


class ConditionalCheckInputType(BaseModel):
    operator: str
    criteria: str


class TestMutationInfo(BaseModel):
    test_suite_id: str = Field(alias="testSuiteId")
    test_id: str = Field(alias="testId")
    input_under_test: str = Field(alias="inputUnderTest")
    checks: List["CheckInputType"]
    golden_output: Optional[str] = Field(alias="goldenOutput", default=None)
    file_ids: Optional[List[Optional[str]]] = Field(alias="fileIds", default=None)
    context: Optional[str] = None
    typed_context: Optional[Any] = Field(alias="typedContext", default=None)
    tags: Optional[List[Optional[str]]] = None


class ParameterInputType(BaseModel):
    eval_model: str = Field(alias="evalModel")
    maximum_threads: int = Field(alias="maximumThreads")
    run_golden_eval: bool = Field(alias="runGoldenEval")
    run_confidence_evaluation: bool = Field(alias="runConfidenceEvaluation")
    heavyweight_factor: int = Field(alias="heavyweightFactor")
    create_text_summary: bool = Field(alias="createTextSummary")
    model_under_test: str = Field(alias="modelUnderTest")
    temperature: float
    max_output_tokens: int = Field(alias="maxOutputTokens")
    system_prompt: str = Field(alias="systemPrompt")
    new_line_stop_option: bool = Field(alias="newLineStopOption")


class PerCheckTestReviewInputType(BaseModel):
    id: int
    binary_human_eval: Optional[int] = Field(alias="binaryHumanEval", default=None)
    is_flagged: Optional[bool] = Field(alias="isFlagged", default=None)


class FixedOutputInputType(BaseModel):
    label: str
    text: str


class QuestionAnswerPairInputType(BaseModel):
    input_under_test: str = Field(alias="inputUnderTest")
    file_ids: Optional[List[Optional[str]]] = Field(alias="fileIds", default=None)
    context: Optional[Any] = None
    output_context: Optional[Any] = Field(alias="outputContext", default=None)
    llm_output: str = Field(alias="llmOutput")
    metadata: Optional["MetadataType"] = None
    test_id: Optional[str] = Field(alias="testId", default=None)


class MetadataType(BaseModel):
    in_tokens: int = Field(alias="inTokens")
    out_tokens: int = Field(alias="outTokens")
    duration_seconds: float = Field(alias="durationSeconds")


CheckInputType.model_rebuild()
CheckModifiersInputType.model_rebuild()
TestMutationInfo.model_rebuild()
QuestionAnswerPairInputType.model_rebuild()
