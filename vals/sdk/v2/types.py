"""
Contains types we define explicitly, as opposed to those that
are auto-generated based on the GraphQL schema. 

These are meant to be user-facing. 
"""

import json
from typing import Any, Literal

from pydantic import BaseModel
from vals.graphql_client.get_test_data import GetTestDataTests
from vals.graphql_client.input_types import TestMutationInfo
from vals.graphql_client.pull_run import PullRunTestResults


class Example(BaseModel):
    """
    In context example for operator
    """

    type: Literal["positive", "negative"]
    text: str


class ConditionalCheck(BaseModel):
    operator: str
    criteria: str


class CheckModifiers(BaseModel):
    optional: bool = None
    severity: float | None = None
    examples: list[Example] | None = None
    extractor: str | None = None
    conditional: ConditionalCheck | None = None

    @classmethod
    def from_graphql(cls, modifiers_dict: dict) -> "CheckModifiers":
        """Internal method to translate from what we receive from GraphQL to the CheckModifiers class."""
        if not modifiers_dict:
            return cls()

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
        )


class Check(BaseModel):
    # TODO: Reuse the typing code from the graphql server (?)
    operator: str
    criteria: str = ""
    modifiers: CheckModifiers = CheckModifiers()

    @classmethod
    def from_graphql(cls, check_dict: dict) -> "Check":
        """Internal method to translate from what we receive from GraphQL to the Check class displayed to the user."""
        modifiers = CheckModifiers.from_graphql(check_dict.get("modifiers", {}))

        return cls(
            operator=check_dict["operator"],
            criteria=check_dict["criteria"],
            modifiers=modifiers,
        )


class Test(BaseModel):
    # We probably want to keep this very in sync with the server somehow, especially
    # Cross verison id
    id: str = "0"
    cross_version_id: str = ""

    input_under_test: str
    checks: list[Check]
    golden_output: str = ""
    tags: list[str] = []
    context: dict[str, Any] = {}
    file_ids: list[str] = []
    """This is the *internal* representation of a file, as stored on the server."""
    files_under_test: list[str] = []
    """This is what the user should specify on their local machine"""

    @classmethod
    def from_graphql_test(cls, graphql_test: GetTestDataTests) -> "Test":
        """Internal method to translate from what we receive from GraphQL to the Test class displayed to the user."""
        return cls(
            id=graphql_test.test_id,
            input_under_test=graphql_test.input_under_test,
            cross_version_id=graphql_test.cross_version_id,
            tags=json.loads(graphql_test.tags),
            context=json.loads(graphql_test.context),
            golden_output=graphql_test.golden_output,
            checks=[
                Check.from_graphql(check) for check in json.loads(graphql_test.checks)
            ],
            file_ids=json.loads(graphql_test.file_ids),
        )

    def to_test_mutation_info(self, test_suite_id: str) -> TestMutationInfo:
        """Internal method to translate from the Test class to the TestMutationInfo class."""

        return TestMutationInfo(
            test_suite_id=test_suite_id,
            test_id=self.id,
            input_under_test=self.input_under_test,
            checks=json.dumps(
                [check.model_dump(exclude_none=True) for check in self.checks]
            ),
            tags=self.tags,
            context=json.dumps(self.context),
            golden_output=self.golden_output,
            file_ids=self.file_ids,
        )


class CheckResult(BaseModel):
    operator: str
    criteria: str
    severity: float
    modifiers: CheckModifiers
    auto_eval: int
    feedback: str
    is_global: bool


class Metadata(BaseModel):
    in_tokens: int
    out_tokens: int
    duration_seconds: float


class TestResult(BaseModel):
    id: str
    input_under_test: str
    llm_output: str
    pass_percentage: float
    pass_percentage_with_optional: float
    check_results: list[CheckResult]
    metadata: Metadata

    @classmethod
    def from_graphql(cls, graphql_test_result: PullRunTestResults) -> "TestResult":
        # TODO: Map this better
        return cls(
            id=graphql_test_result.id,
            input_under_test=graphql_test_result.test.input_under_test,
            llm_output=graphql_test_result.llm_output,
            pass_percentage=graphql_test_result.pass_percentage,
            pass_percentage_with_optional=graphql_test_result.pass_percentage_with_optional,
            check_results=[
                CheckResult(**check_result)
                for check_result in json.loads(graphql_test_result.result_json)
            ],
            metadata=Metadata(**json.loads(graphql_test_result.metadata)),
        )
