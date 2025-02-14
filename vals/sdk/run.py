import asyncio
import json
import time
from datetime import datetime
from typing import Any

import requests
from pydantic import BaseModel, PrivateAttr
from vals.graphql_client import Client
from vals.graphql_client.pull_run import PullRun
from vals.sdk.exceptions import ValsException
from vals.sdk.inspect_wrapper import InspectWrapper
from vals.sdk.types import (
    Metadata,
    ModelCustomOperatorFunctionType,
    ModelFunctionType,
    QuestionAnswerPair,
    RunMetadata,
    RunParameters,
    RunStatus,
    TestResult,
)
from vals.sdk.util import _get_auth_token, be_host, fe_host, get_ariadne_client


class Run(BaseModel):
    id: str
    """Unique identifier for the run."""

    name: str
    """Name of the run."""

    qa_set_id: str | None
    """Unique identifier for the QA set run was created with."""

    test_suite_id: str
    """Unique identifier for the test suite run was created with."""

    test_suite_title: str
    """Title of the test suite run was created with."""

    model: str
    """Model used to perform the run."""

    pass_percentage: float | None
    """Average pass percentage of all tests."""

    pass_rate: float | None
    """Percentage of checks that passed"""

    pass_rate_error: float | None
    """Error margin for pass rate"""

    success_rate: float | None
    """Number of tests where all checks passed"""

    success_rate_error: float | None
    """Error margin for success rate"""

    status: RunStatus
    """Status of the run"""

    archived: bool
    """Whether the run has been archived"""

    text_summary: str
    """Automatically generated summary of common error modes for the run."""

    timestamp: datetime
    """Timestamp of when the run was created."""

    completed_at: datetime | None
    """Timestamp of when the run was completed."""

    parameters: RunParameters
    """Parameters used to create the run."""

    test_results: list[TestResult]
    """List of test results for the run."""

    _client: Client = PrivateAttr(default_factory=get_ariadne_client)

    @staticmethod
    def _create_from_pull_result(run_id: str, result: PullRun) -> "Run":
        """Helper method to create a Run instance from a pull_run query result"""

        # Map maximum_threads to parallelism for backwards compatibility
        parameters_dict = result.run.typed_parameters.model_dump()
        model = parameters_dict.pop("model_under_test", "")
        if "maximum_threads" in parameters_dict:
            parameters_dict["parallelism"] = parameters_dict.pop("maximum_threads")
        parameters = RunParameters(**parameters_dict)

        return Run(
            id=run_id,
            name=result.run.name,
            qa_set_id=result.run.qa_set.id if result.run.qa_set else None,
            model=model,
            pass_percentage=(
                result.run.pass_percentage * 100
                if result.run.pass_percentage is not None
                else None
            ),
            pass_rate=result.run.pass_rate.value if result.run.pass_rate else None,
            pass_rate_error=(
                result.run.pass_rate.error if result.run.pass_rate else None
            ),
            success_rate=(
                result.run.success_rate.value if result.run.success_rate else None
            ),
            success_rate_error=(
                result.run.success_rate.error if result.run.success_rate else None
            ),
            status=RunStatus(result.run.status),
            archived=result.run.archived,
            text_summary=result.run.text_summary,
            timestamp=result.run.timestamp,
            completed_at=result.run.completed_at,
            parameters=parameters,
            test_results=[
                TestResult.from_graphql(test_result)
                for test_result in result.test_results
            ],
            test_suite_title=result.run.test_suite.title,
            test_suite_id=result.run.test_suite.id,
        )

    @classmethod
    async def list_runs(
        cls,
        limit: int = 25,
        offset: int = 0,
        suite_id: str | None = None,
        show_archived: bool = False,
        search: str = "",
    ) -> list["RunMetadata"]:
        """List runs associated with this organization"""
        client = get_ariadne_client()
        result = await client.list_runs(
            limit=limit,
            offset=offset,
            suite_id=suite_id,
            archived=show_archived,
            search=search,
        )
        return [
            RunMetadata.from_graphql(run) for run in result.runs_with_count.run_results
        ]

    @classmethod
    async def from_id(cls, run_id: str) -> "Run":
        """Pull most recent metadata and test results from the vals servers."""
        client = get_ariadne_client()
        result = await client.pull_run(run_id=run_id)
        return cls._create_from_pull_result(run_id, result)

    @property
    def url(self) -> str:
        return f"{fe_host()}/results/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        """Converts the run to a dictionary."""
        return self.model_dump(exclude_none=True, exclude_defaults=True, mode="json")

    def to_json_file(self, file_path: str) -> None:
        """Converts the run to a JSON file."""
        with open(file_path, "w") as f:
            f.write(self.to_json_string())

    async def pull(self) -> None:
        """Update this Run instance with latest data from vals servers."""
        result = await self._client.pull_run(run_id=self.id)
        updated = self._create_from_pull_result(self.id, result)
        # TODO: There's probably a better way to update the object.
        for field in updated.__fields__:
            setattr(self, field, getattr(updated, field))

    async def get_qa_pairs(
        self, offset: int = 0, limit: int = 1000
    ) -> list[QuestionAnswerPair]:
        """Get all QA pairs for a run."""
        result = await self._client.list_question_answer_pairs(
            qa_set_id=self.qa_set_id, offset=offset, limit=limit
        )
        return [
            QuestionAnswerPair.from_graphql(graphql_qa_pair)
            for graphql_qa_pair in result.question_answer_pairs_with_count.question_answer_pairs
        ]

    async def run_status(self) -> RunStatus:
        """Get the status of a run"""
        result = await self._client.run_status(run_id=self.id)
        status = result.run.status
        self.status = RunStatus(status.lower())
        return self.status

    async def wait_for_run_completion(
        self,
    ) -> RunStatus:
        """
        Block a process until a given run has finished running.

        Returns the status of the run after completion.
        """
        await asyncio.sleep(1)
        status = RunStatus.IN_PROGRESS
        start_time = time.time()
        while status == RunStatus.IN_PROGRESS:
            status = await self.run_status()

            # Sleep longer between polls, the longer the run goes.
            if time.time() - start_time < 60:
                sleep_time = 1
            elif time.time() - start_time < 60 * 10:
                sleep_time = 5
            else:
                sleep_time = 10

            await asyncio.sleep(sleep_time)

        return status

    async def to_csv_string(self) -> str:
        """Same as to_csv, but returns a string instead of writing to a file."""
        response = requests.post(
            url=f"{be_host()}/export_results_to_file/?run_id={self.id}",
            headers={"Authorization": _get_auth_token()},
        )

        if response.status_code != 200:
            raise ValsException("Received Error from Vals Servers: " + response.text)

        return response.text

    async def to_json_string(self) -> str:

        response = requests.post(
            url=f"{be_host()}/export_run_to_json/?run_id={self.id}",
            headers={"Authorization": _get_auth_token()},
        )

        if response.status_code != 200:
            raise ValsException("Received Error from Vals Servers: " + response.text)

        return response.text

    async def to_csv(self, file_path: str) -> None:
        """Get the CSV results of a run, as bytes."""
        with open(file_path, "w") as f:
            f.write(await self.to_csv_string())

    async def retry_failing_tests(self) -> None:
        """Retry all failing tests in a run."""

        await self._client.rerun_tests(run_id=self.id)

    async def resume_run(
        self,
        model: str | ModelFunctionType | list[QuestionAnswerPair] | InspectWrapper,
        wait_for_completion: bool = False,
        upload_concurrency: int | None = None,
        custom_operators: list[ModelCustomOperatorFunctionType] | None = None,
        parallelism: int | None = None,
    ) -> None:
        """Resume a run that was paused.

        This method will:
        1. Check for existing QA pairs that haven't been auto-evaluated
        2. Check for existing completed test results
        3. Run the remaining tests that haven't been processed yet
        """
        if custom_operators is None:
            custom_operators = []

        if parallelism is not None:
            self.parameters.parallelism = parallelism
        # Import Suite here to avoid circular import
        from vals.sdk.suite import Suite

        # Get the test suite
        suite = await Suite.from_id(self.test_suite_id)

        # Create sets of existing test IDs for both QA pairs and completed results
        completed_qa_pairs = {}
        completed_test_results = {}

        qa_pairs = await self.get_qa_pairs(offset=0, limit=len(suite.tests))

        for qa_pair in qa_pairs:
            if len(qa_pair.local_evals) > 0:
                completed_test_results[qa_pair.test_id] = qa_pair
            else:
                completed_qa_pairs[qa_pair.test_id] = qa_pair

        # Filter suite tests to only include those that haven't been processed
        remaining_tests = {
            test._id: test
            for test in suite.tests
            if test._id not in completed_qa_pairs
            and test._id not in completed_test_results
        }

        uploaded_qa_pairs = [
            qa_pair for qa_pair in qa_pairs if qa_pair.test_id in completed_qa_pairs
        ]

        print(f"Remaining model outputs to generate: {len(remaining_tests)}")

        print(
            "Remaining model outputs to evaluate: ",
            len(suite.tests) - len(completed_test_results),
        )

        # Run the remaining tests, including existing QA pairs
        await self._client.update_run_status(
            run_id=self.id, status=RunStatus.IN_PROGRESS
        )
        await suite.run(
            model=model,
            model_name=self.model,
            run_name=self.name,
            wait_for_completion=wait_for_completion,
            parameters=self.parameters,
            upload_concurrency=upload_concurrency,
            custom_operators=custom_operators,
            eval_model_name=self.parameters.eval_model,
            run_id=self.id,
            qa_set_id=self.qa_set_id,
            remaining_tests=list(remaining_tests.values()),
            uploaded_qa_pairs=uploaded_qa_pairs,
        )
