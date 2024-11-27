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
from vals.graphql_client.get_test_data import GetTestDataTests
from vals.graphql_client.get_test_suites_with_count import (
    GetTestSuitesWithCountTestSuitesWithCountTestSuites,
)
from vals.graphql_client.input_types import (
    QuestionAnswerPairInputType,
    TestMutationInfo,
)
from vals.graphql_client.pull_run import PullRunTestResults
from vals.sdk.v2.operator_type import OperatorType

SimpleModelFunctionType = Callable[[str], str]

ModelFunctionWithFilesAndContextType = Callable[
    [str, dict[str, BytesIO], dict[str, Any]], str
]

ModelFunctionType = SimpleModelFunctionType | ModelFunctionWithFilesAndContextType


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

    operator: OperatorType
    criteria: str


class CheckModifiers(BaseModel):
    optional: bool | None = None
    """Do not include this check towards the final pass percentage"""

    severity: float | None = None
    """Relative weight of this check - see documentation."""

    examples: list[Example] | None = None
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

        examples = None
        if modifiers_dict.get("examples", None):
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
    operator: OperatorType
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


class Test(BaseModel):
    _id: str = "0"
    """Internal id of the test. 0 signifies it hasn't been created yet."""

    _cross_version_id: str = ""
    """Internal id that stays constant across versions."""

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

    files_under_test: list[str] = []
    """Local file paths to upload as part of the test input - i.e. documents, etc."""

    _file_ids: list[str] = []
    """This is the *internal* representation of a file, as stored on the server. You generally should not edit or use these. """

    @classmethod
    def from_graphql_test(cls, graphql_test: GetTestDataTests) -> "Test":
        """Internal method to translate from what we receive from GraphQL to the Test class displayed to the user."""
        test = cls(
            input_under_test=graphql_test.input_under_test,
            tags=json.loads(graphql_test.tags),
            context=json.loads(graphql_test.context),
            golden_output=graphql_test.golden_output,
            checks=[
                Check.from_graphql(check) for check in json.loads(graphql_test.checks)
            ],
        )
        test._file_ids = json.loads(graphql_test.file_ids)
        test._id = graphql_test.test_id
        test._cross_version_id = graphql_test.cross_version_id
        return test

    def to_test_mutation_info(self, test_suite_id: str) -> TestMutationInfo:
        """Internal method to translate from the Test class to the TestMutationInfo class."""
        return TestMutationInfo(
            test_suite_id=test_suite_id,
            test_id=self._id,
            input_under_test=self.input_under_test,
            checks=json.dumps(
                [check.model_dump(exclude_none=True) for check in self.checks]
            ),
            tags=self.tags,
            context=json.dumps(self.context),
            golden_output=self.golden_output,
            file_ids=self._file_ids,
        )


class Metadata(BaseModel):
    in_tokens: int
    out_tokens: int
    duration_seconds: float


class CheckResult(BaseModel):
    """Result of evaluation for a single check."""

    # Same as the input fields.
    operator: OperatorType
    criteria: str
    severity: float
    modifiers: CheckModifiers
    is_global: bool

    auto_eval: int
    """Binary pass / fail of the check, 0 for fail, 1 for pass"""

    feedback: str
    """Autogenerated free-text feedback for the check"""


class TestResult(BaseModel):
    """Result of evaluation for a single test."""

    _id: str
    input_under_test: str

    llm_output: str
    """Output produced by the LLM"""

    metadata: Metadata | None

    pass_percentage: float
    """Percent of passing checks for the test"""

    check_results: list[CheckResult]
    """Results for every check"""

    @classmethod
    def from_graphql(cls, graphql_test_result: PullRunTestResults) -> "TestResult":
        obj = cls(
            _id=graphql_test_result.id,
            input_under_test=graphql_test_result.test.input_under_test,
            llm_output=graphql_test_result.llm_output,
            pass_percentage=graphql_test_result.pass_percentage,
            check_results=[
                CheckResult(**check_result)
                for check_result in json.loads(graphql_test_result.result_json)
            ],
            metadata=(
                Metadata(**json.loads(graphql_test_result.metadata))
                if graphql_test_result.metadata
                else None
            ),
        )
        obj._id = graphql_test_result.id
        return obj


class RunStatus(str, Enum):
    """Status of a run: 'in_progress', 'completed', or 'success'."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SUCCESS = "success"


class QuestionAnswerPair(BaseModel):
    input_under_test: str
    llm_output: str
    file_ids: list[str] | None = None
    context: dict[str, Any] | None = None
    metadata: Metadata | None = None

    def to_graphql(self) -> QuestionAnswerPairInputType:
        return QuestionAnswerPairInputType(
            input_under_test=self.input_under_test,
            file_ids=self.file_ids,
            context=self.context,
            llm_output=self.llm_output,
            metadata=self.metadata,
            test_id=None,
        )
