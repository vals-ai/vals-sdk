# Generated by ariadne-codegen
# Source: vals/graphql/

from datetime import datetime
from typing import List, Optional

from pydantic import Field

from .base_model import BaseModel
from .enums import RunStatus


class ListRuns(BaseModel):
    runs_with_count: "ListRunsRunsWithCount" = Field(alias="runsWithCount")


class ListRunsRunsWithCount(BaseModel):
    run_results: List["ListRunsRunsWithCountRunResults"] = Field(alias="runResults")


class ListRunsRunsWithCountRunResults(BaseModel):
    run_id: str = Field(alias="runId")
    pass_percentage: Optional[float] = Field(alias="passPercentage")
    pass_rate: Optional["ListRunsRunsWithCountRunResultsPassRate"] = Field(
        alias="passRate"
    )
    success_rate: Optional["ListRunsRunsWithCountRunResultsSuccessRate"] = Field(
        alias="successRate"
    )
    name: str
    status: RunStatus
    text_summary: str = Field(alias="textSummary")
    timestamp: datetime
    completed_at: Optional[datetime] = Field(alias="completedAt")
    archived: bool
    parameters: "ListRunsRunsWithCountRunResultsParameters"
    test_suite: "ListRunsRunsWithCountRunResultsTestSuite" = Field(alias="testSuite")


class ListRunsRunsWithCountRunResultsPassRate(BaseModel):
    value: float
    error: float


class ListRunsRunsWithCountRunResultsSuccessRate(BaseModel):
    value: float
    error: float


class ListRunsRunsWithCountRunResultsParameters(BaseModel):
    eval_model: str = Field(alias="evalModel")
    maximum_threads: int = Field(alias="maximumThreads")
    run_confidence_evaluation: bool = Field(alias="runConfidenceEvaluation")
    heavyweight_factor: int = Field(alias="heavyweightFactor")
    create_text_summary: bool = Field(alias="createTextSummary")
    model_under_test: str = Field(alias="modelUnderTest")
    temperature: float
    max_output_tokens: int = Field(alias="maxOutputTokens")
    system_prompt: str = Field(alias="systemPrompt")


class ListRunsRunsWithCountRunResultsTestSuite(BaseModel):
    title: str


ListRuns.model_rebuild()
ListRunsRunsWithCount.model_rebuild()
ListRunsRunsWithCountRunResults.model_rebuild()
