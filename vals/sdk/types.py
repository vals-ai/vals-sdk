"""
Contains types we define explicitly, as opposed to those that
are auto-generated based on the GraphQL schema.

These are meant to be user-facing.
"""

import datetime
import json
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Literal

from pydantic import BaseModel
from vals.graphql_client.get_test_data import GetTestDataTestsWithCountTests
from vals.graphql_client.get_test_suites_with_count import (
    GetTestSuitesWithCountTestSuitesWithCountTestSuites,
)
from vals.graphql_client.input_types import (
    CheckInputType,
    CheckModifiersInputType,
    LocalEvalUploadInputType,
    MetadataType,
    QuestionAnswerPairInputType,
    TestMutationInfo,
)
from vals.graphql_client.list_question_answer_pairs import (
    ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairs,
)
from vals.graphql_client.list_runs import ListRunsRunsWithCountRunResults
from vals.graphql_client.pull_test_results_with_count import (
    PullTestResultsWithCountTestResultsWithCountTestResults,
)
from vals.sdk.operator_type import OperatorType


class TestSuiteMetadata(BaseModel):
    """
    Data returned about a test suite when we are calling the
    list_test_suites() function - does not include tests, global
    checks, etc.
    """

    id: str
    title: str
    description: str
    created: datetime.datetime
    creator: str
    last_modified_by: str
    last_modified_at: datetime.datetime
    folder_id: str | None
    folder_name: str | None

    # TODO: Often, in this file, we're mapping back and forth between our custom types
    # and the auto-generated types. There is probably a better solution for this.
    @classmethod
    def from_graphql(
        cls, graphql_suite: GetTestSuitesWithCountTestSuitesWithCountTestSuites
    ) -> "TestSuiteMetadata":

        return cls(
            id=graphql_suite.id,
            title=graphql_suite.title,
            description=graphql_suite.description,
            created=graphql_suite.created,
            creator=graphql_suite.creator,
            last_modified_by=graphql_suite.last_modified_by,
            last_modified_at=graphql_suite.last_modified_at,
            folder_id=graphql_suite.folder.id if graphql_suite.folder else None,
            folder_name=graphql_suite.folder.name if graphql_suite.folder else None,
        )


class Example(BaseModel):
    """
    In-context example modifier
    """

    type: Literal["positive", "negative"]
    text: str


class ConditionalCheck(BaseModel):
    """
    Predicate / conditional check modifier.
    """

    operator: OperatorType | str
    criteria: str = ""


class CheckModifiers(BaseModel):
    optional: bool = False
    """Do not include this check towards the final pass percentage"""

    severity: float | None = None
    """Relative weight of this check - see documentation."""

    examples: list[Example] = []
    """In-context examples for the check"""

    extractor: str | None = None
    """Extract certain aspects of the output before the check is evaluated"""

    conditional: ConditionalCheck | None = None
    """Only run this check if another check evaluates to true."""

    category: str | None = None
    """Override the default category of the check (correctness, formatting, etc.)"""

    @classmethod
    def from_graphql(cls, modifiers_dict: dict) -> "CheckModifiers":
        """Internal method to translate from what we receive from GraphQL to the CheckModifiers class."""
        if not modifiers_dict:
            return cls()

        examples = []
        if modifiers_dict.get("examples"):
            examples = [Example(**example) for example in modifiers_dict["examples"]]

        conditional = None
        if modifiers_dict.get("conditional"):
            conditional = ConditionalCheck(**modifiers_dict["conditional"])

        return cls(
            optional=modifiers_dict.get("optional", False),
            severity=modifiers_dict.get("severity"),
            examples=examples,
            extractor=modifiers_dict.get("extractor"),
            conditional=conditional,
            category=modifiers_dict.get("category"),
        )


class Check(BaseModel):
    operator: OperatorType | str
    """Operator - see documentation for the full list."""

    criteria: str = ""
    """Criteria for the check - only required if it is not a unary operator."""

    modifiers: CheckModifiers = CheckModifiers()
    """Optional additional modifiers for the check."""

    @classmethod
    def from_graphql(cls, check_dict: dict) -> "Check":
        """Internal method to translate from what we receive from GraphQL to the Check class displayed to the user."""
        modifiers = CheckModifiers.from_graphql(check_dict.get("modifiers", {}))
        return cls(
            operator=check_dict["operator"],
            criteria=check_dict.get("criteria", ""),
            modifiers=modifiers,
        )

    def to_graphql_input(self) -> dict:
        return CheckInputType(
            operator=self.operator,
            criteria=self.criteria,
            modifiers=CheckModifiersInputType(
                **self.modifiers.model_dump(exclude_none=True)
            ),
        )


class File(BaseModel):
    file_name: str

    file_id: str | None = None

    path: str | None = None

    hash: str | None = None


class Test(BaseModel):
    _id: str = "0"
    """Internal id of the test. 0 signifies it hasn't been created yet."""

    _cross_version_id: str = ""
    """Internal id that stays constant across versions."""

    _test_suite_id: str = ""
    """Maintain internal representation of which Test Suite the test originally belonged to"""

    input_under_test: str
    """Input to the LLM"""

    checks: list[Check]
    """List of checks to apply to the LLM's output"""

    golden_output: str = ""
    """ Expected output of the LLM - can be used instead of or in-conjuction with checks"""

    tags: list[str] = []
    """Tags for the test"""

    context: dict[str, Any] = {}
    """Arbitrary additional context to be used as input for the test."""

    files_under_test: list[File] | list[dict[str, Any]] | list[str] = []
    """Local file paths to upload as part of the test input - i.e. documents, etc."""

    _file_ids: list[str] = []
    """This is the *internal* representation of a file, as stored on the server. You generally should not edit or use these. """

    @classmethod
    def from_graphql_test(cls, graphql_test: GetTestDataTestsWithCountTests) -> "Test":
        """Internal method to translate from what we receive from GraphQL to the Test class displayed to the user."""
        test = cls(
            input_under_test=graphql_test.input_under_test,
            tags=json.loads(graphql_test.tags),
            context=json.loads(graphql_test.context),
            golden_output=graphql_test.golden_output,
            checks=[
                Check.from_graphql(check) for check in json.loads(graphql_test.checks)
            ],
            files_under_test=[
                File(
                    file_name=file.split("-", 1)[-1],
                    file_id=file,
                    hash=file.split("-", 1)[0].split("/", 1)[-1],
                    path=None,
                )
                for file in json.loads(graphql_test.file_ids)
            ],
        )
        test._file_ids = json.loads(graphql_test.file_ids)
        test._id = graphql_test.test_id
        test._cross_version_id = graphql_test.cross_version_id
        test._test_suite_id = graphql_test.test_suite.id
        return test

    def to_test_mutation_info(self, test_suite_id: str) -> TestMutationInfo:
        """Internal method to translate from the Test class to the TestMutationInfo class."""

        file_ids = [file.file_id for file in self.files_under_test]

        return TestMutationInfo(
            test_suite_id=test_suite_id,
            # If we're moving the test to a new test suite, we always need to create it
            test_id=self._id if test_suite_id == self._test_suite_id else "0",
            input_under_test=self.input_under_test,
            checks=[check.to_graphql_input() for check in self.checks],
            tags=self.tags,
            context=json.dumps(self.context),
            golden_output=self.golden_output,
            file_ids=file_ids,
        )


class RunParameters(BaseModel):
    """Parameters for a run."""

    eval_model: str = "gpt-4o"
    """Model to use for the LLM as judge - this is *not* the model being tested."""

    parallelism: int = 10
    """How many tests to run in parallel"""

    run_golden_eval: bool = False
    """Compares the output to the golden ansewr, if provided"""

    run_confidence_evaluation: bool = True
    """ If false, don't produce confidence scores for checks """

    heavyweight_factor: int = 1
    """Run the auto eval multiple times and take the mode of the results"""

    create_text_summary: bool = True
    """If false, will not generate a text summary of the run"""

    temperature: float = 0
    """ Temperature for model being tested"""

    max_output_tokens: int = 512
    """Maximum number of tokens in the output for the model under test"""

    system_prompt: str = ""
    """System prompt for the model under test"""

    new_line_stop_option: bool = False
    """If true, will stop generation at a new line"""

    retry_failed_calls_indefinitely: bool = False
    """ If true, when receiving an error from the model, will retry indefinitely until it receives a success."""


class RunStatus(str, Enum):
    """Status of a run: 'in_progress', 'error', or 'success'."""

    IN_PROGRESS = "in_progress"
    ERROR = "error"
    SUCCESS = "success"
    RERUNNING = "rerunning"


class RunMetadata(BaseModel):
    id: str
    name: str
    pass_percentage: float | None
    pass_rate: float | None
    pass_rate_error: float | None
    success_rate: float | None
    success_rate_error: float | None
    status: RunStatus
    text_summary: str
    timestamp: datetime.datetime
    completed_at: datetime.datetime | None
    archived: bool
    test_suite_title: str
    model: str
    parameters: RunParameters

    @classmethod
    def from_graphql(
        cls, graphql_run: ListRunsRunsWithCountRunResults
    ) -> "RunMetadata":
        return cls(
            id=graphql_run.run_id,
            name=graphql_run.name,
            pass_percentage=(
                graphql_run.pass_percentage if graphql_run.pass_percentage else None
            ),
            pass_rate=graphql_run.pass_rate.value if graphql_run.pass_rate else None,
            pass_rate_error=(
                graphql_run.pass_rate.error if graphql_run.pass_rate else None
            ),
            success_rate=(
                graphql_run.success_rate.value if graphql_run.success_rate else None
            ),
            success_rate_error=(
                graphql_run.success_rate.error if graphql_run.success_rate else None
            ),
            status=RunStatus(graphql_run.status),
            text_summary=graphql_run.text_summary,
            timestamp=graphql_run.timestamp,
            completed_at=graphql_run.completed_at,
            archived=graphql_run.archived,
            test_suite_title=graphql_run.test_suite.title,
            model=graphql_run.typed_parameters.model_under_test,
            parameters=RunParameters(**graphql_run.typed_parameters.model_dump()),
        )


class Metadata(BaseModel):
    in_tokens: int | None = None
    out_tokens: int | None = None
    duration_seconds: float | None = None


class Confidence(Enum):
    """Confidence of a check."""

    LOW = 0
    """Low confidence in the result of the check"""

    CONFIDENCE_NOT_RUN = 0.5
    """Did not run confidence checking on this check"""

    HIGH = 1
    """High confidence in the result of the check"""


class CheckResult(BaseModel):
    """Result of evaluation for a single check."""

    # Same as the input fields.
    operator: OperatorType | str
    criteria: str
    modifiers: CheckModifiers
    is_global: bool

    auto_eval: float
    """Binary pass / fail of the check, 0 for fail, 1 for pass"""

    feedback: str
    """Autogenerated free-text feedback for the check"""

    confidence: Confidence


class TestResult(BaseModel):
    """Result of evaluation for a single test."""

    _id: str
    test_id: str
    input_under_test: str

    context: dict[str, Any]
    """Context pulled from the test"""

    output_context: dict[str, Any]
    """Context produced while producing the output."""

    llm_output: str
    """Output produced by the LLM"""

    metadata: Metadata | None

    pass_percentage: float
    """Percent of passing checks for the test"""

    pass_percentage_with_optional: float
    """Percent of passing checks including optional checks"""

    check_results: list[CheckResult]
    """Results for every check"""

    error_message: str = ""

    @classmethod
    def from_graphql(
        cls,
        graphql_test_result: PullTestResultsWithCountTestResultsWithCountTestResults,
    ) -> "TestResult":
        output_context = {}
        context = {}
        if graphql_test_result.qa_pair:
            output_context = graphql_test_result.qa_pair.output_context
            context = graphql_test_result.qa_pair.context
            if len(context) == 0 and graphql_test_result.test.context is not None:
                context = json.loads(graphql_test_result.test.context)

        return cls(
            _id=graphql_test_result.id,
            test_id=graphql_test_result.test.test_id,
            input_under_test=graphql_test_result.test.input_under_test,
            context=context,
            output_context=output_context,
            llm_output=graphql_test_result.llm_output,
            pass_percentage=graphql_test_result.pass_percentage,
            pass_percentage_with_optional=graphql_test_result.pass_percentage_with_optional,
            error_message=(
                graphql_test_result.qa_pair.error_message
                if graphql_test_result.qa_pair
                else ""
            ),
            check_results=[
                CheckResult(
                    operator=check_result["operator"],
                    criteria=check_result.get("criteria", ""),
                    modifiers=CheckModifiers.from_graphql(
                        check_result.get("modifiers", {})
                    ),
                    is_global=check_result.get("is_global", False),
                    auto_eval=check_result.get("auto_eval", 0),
                    feedback=check_result.get("feedback", ""),
                    confidence=Confidence(
                        check_result.get("eval_cont", 0.5)
                        if check_result.get("eval_cont", 0.5) in [0, 0.5, 1]
                        else 0.5
                    ),
                )
                for check_result in json.loads(graphql_test_result.result_json)
            ],
            metadata=(
                Metadata(**json.loads(graphql_test_result.metadata))
                if graphql_test_result.metadata
                else None
            ),
        )


class QuestionAnswerPair(BaseModel):
    id: str | None = None
    input_under_test: str
    llm_output: str
    file_ids: list[str] | None = None
    context: dict[str, Any] = {}
    output_context: dict[str, Any] = {}
    metadata: Metadata | None = None
    test_id: str | None = None
    local_evals: list[LocalEvalUploadInputType] | None = None

    @classmethod
    def from_graphql(
        cls,
        graphql_qa_pair: ListQuestionAnswerPairsQuestionAnswerPairsWithCountQuestionAnswerPairs,
    ) -> "QuestionAnswerPair":
        metadata = None
        if graphql_qa_pair.typed_metadata:
            metadata = Metadata(
                in_tokens=graphql_qa_pair.typed_metadata.in_tokens,
                out_tokens=graphql_qa_pair.typed_metadata.out_tokens,
                duration_seconds=graphql_qa_pair.typed_metadata.duration_seconds,
            )

        return cls(
            id=graphql_qa_pair.id,
            input_under_test=graphql_qa_pair.input_under_test,
            llm_output=graphql_qa_pair.llm_output,
            file_ids=graphql_qa_pair.typed_file_ids,
            context=graphql_qa_pair.context or {},
            output_context=graphql_qa_pair.output_context or {},
            metadata=metadata,
            test_id=graphql_qa_pair.test.test_id if graphql_qa_pair.test else None,
            local_evals=(
                [
                    LocalEvalUploadInputType(
                        question_answer_pair_id=graphql_qa_pair.id,
                        score=eval.score,
                        feedback=eval.feedback,
                        name="local_eval",
                    )
                    for eval in graphql_qa_pair.local_evals
                ]
                if graphql_qa_pair.local_evals
                else []
            ),
        )

    def to_graphql(self) -> QuestionAnswerPairInputType:
        return QuestionAnswerPairInputType(
            input_under_test=self.input_under_test,
            file_ids=self.file_ids,
            context=self.context,
            output_context=self.output_context,
            llm_output=self.llm_output,
            metadata=(
                MetadataType(**self.metadata.model_dump()) if self.metadata else None
            ),
            test_id=self.test_id,
        )


class OperatorInput(BaseModel):
    input: str
    model_output: str
    context: dict[str, Any] | None = None
    output_context: dict[str, Any] | None = None
    files: dict[str, BytesIO] | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }


class OperatorOutput(BaseModel):
    name: str
    score: float
    explanation: str


ModelCustomOperatorFunctionType = Callable[[OperatorInput], OperatorOutput]


class CustomModelInput(BaseModel):
    input_under_test: str
    context: dict[str, Any]
    files: dict[str, BytesIO]

    model_config = {"arbitrary_types_allowed": True}


class CustomModelOutput(BaseModel):
    model_output: str | dict[str, Any] | QuestionAnswerPair

    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }


SimpleModelFunctionType = Callable[[str], str]

ModelFunctionWithFilesAndContextType = Callable[[CustomModelInput], CustomModelOutput]

ModelFunctionType = SimpleModelFunctionType | ModelFunctionWithFilesAndContextType


class RunReviewStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
