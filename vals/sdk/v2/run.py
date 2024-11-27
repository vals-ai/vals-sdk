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
from vals.sdk.util import be_host, fe_host
from vals.sdk.v2.types import RunStatus, TestResult
from vals.sdk.v2.util import _get_auth_token, get_ariadne_client


class Run(BaseModel):
    id: str
    test_suite_id: str
    test_suite_title: str
    pass_percentage: float | None
    status: str
    archived: bool
    text_summary: str
    timestamp: datetime
    completed_at: datetime | None
    parameters: dict[str, Any]
    test_results: list[TestResult]

    _client: Client = PrivateAttr(default_factory=get_ariadne_client)

    @staticmethod
    def _create_from_pull_result(run_id: str, result: PullRun) -> "Run":
        """Helper method to create a Run instance from a pull_run query result"""
        return Run(
            id=run_id,
            test_results=[
                TestResult.from_graphql(test_result)
                for test_result in result.test_results
            ],
            pass_percentage=result.run.pass_percentage,
            text_summary=result.run.text_summary,
            archived=result.run.archived,
            parameters=json.loads(result.run.parameters),
            test_suite_title=result.run.test_suite.title,
            test_suite_id=result.run.test_suite.title,
            status=result.run.status,
            timestamp=result.run.timestamp,
            completed_at=result.run.completed_at,
        )

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
            json.dump(self.to_dict(), f, indent=2)

    async def pull(self) -> None:
        """Update this Run instance with latest data from vals servers."""
        result = await self._client.pull_run(run_id=self.id)
        updated = self._create_from_pull_result(self.id, result)
        # TODO: There's probably a better way to update the object.
        for field in updated.__fields__:
            setattr(self, field, getattr(updated, field))

    async def run_status(self) -> RunStatus:
        """Get the status of a run"""

        result = await self._client.run_status(run_id=self.id)
        self.status = result.run.status
        return RunStatus(result.run.status)

    async def wait_for_run_completion(
        self,
    ) -> RunStatus:
        """
        Block a process until a given run has finished running.

        Returns the status of the run after completion.
        """
        await asyncio.sleep(1)
        status = "in_progress"
        start_time = time.time()
        while status == "in_progress":
            status = await self.run_status()

            # Sleep longer between polls, the longer the run goes.
            if time.time() - start_time < 60:
                sleep_time = 1
            elif time.time() - start_time < 60 * 10:
                sleep_time = 5
            else:
                sleep_time = 10

            await asyncio.sleep(sleep_time)

        return RunStatus(status)

    async def to_csv(self, file_path: str) -> None:
        """Get the CSV results of a run, as bytes."""

        with open(file_path, "wb") as f:
            response = requests.post(
                url=f"{be_host()}/export_results_to_file/?run_id={self.id}",
                headers={"Authorization": _get_auth_token()},
            )

            if response.status_code != 200:
                raise ValsException(
                    "Received Error from Vals Servers: " + response.text
                )

            f.write(response.content)
