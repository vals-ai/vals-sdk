import asyncio
import json
from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, PrivateAttr
from vals.graphql_client.client import Client
from vals.graphql_client.get_test_data import GetTestDataTests
from vals.graphql_client.get_test_suites import GetTestSuitesTestSuites
from vals.sdk.run import _get_default_parameters
from vals.sdk.v2.run import wait_for_run_completion
from vals.sdk.v2.types import Check, Test
from vals.sdk.v2.util import get_ariadne_client


class Suite(BaseModel):
    id: str | None = None

    title: str
    description: str = ""
    global_checks: list[Check] = []
    tests: list[Test] = []

    """
    Helper to see if we've instantiated the suite, but haven't 
    fetched it from the server yet. 
    """
    _client: Client = PrivateAttr(default_factory=get_ariadne_client)

    @classmethod
    async def list_suites(cls) -> list[GetTestSuitesTestSuites]:
        """
        Generate a list of all suites on the server.
        """
        # TODO: What type should this be?
        client = get_ariadne_client()
        return (await client.get_test_suites()).test_suites

    async def pull(self) -> None:
        """
        Updates the local fields to match the suite on the server.
        """
        if self.id is None:
            raise Exception("Suite ID is not set for this test suite, can't pull.")

        # TODO: We could do this all lazily and only pull the
        # actual title, description, etc. when they query the fields.

        suites_list = await self._client.get_test_suite_data(self.id)
        if len(suites_list.test_suites) == 0:
            raise Exception("Couldn't find suite with id: " + self.id)

        suite_data = suites_list.test_suites[0]
        title = suite_data.title
        description = suite_data.description

        if suite_data.global_checks is not None:
            global_checks = [
                Check.from_graphql(check)
                for check in json.loads(suite_data.global_checks)
            ]

        test_data = await self._client.get_test_data(self.suite_id)

        tests = []
        for graphql_test in test_data.tests:
            test = Test.from_graphql_test(graphql_test)
            tests.append(test)

        return cls(
            id=suite_id,
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )

    async def create(self) -> None:
        """
        Creates the test suite on the server.
        """

        # TODO: Some variation of if id is already set,
        # or if the suite exists already, we should error.

        # TODO: Client side validation of test suite

        # TODO: Upload files

        # TODO: This should really be transactional somehow,
        # if uploading tests fails, we shouldn't create the suite.
        # 0 signifies new test suite.
        suite = await self._client.create_or_update_test_suite(
            "0", self.title, self.description
        )
        self.id = suite.update_test_suite.test_suite.id

        await self._client.update_global_checks(
            self.id,
            json.dumps([gc.model_dump(exclude_none=True) for gc in self.global_checks]),
        )

        # TODO: Batch in batches of 100?
        created_tests = await self._client.add_batch_tests(
            tests=[test.to_test_mutation_info(self.id) for test in self.tests],
            create_only=True,
        )

        # for local_test, server_test in zip(
        #     self.tests, created_tests.batch_update_test.tests
        # ):
        #     local_test.cross = server_test.test_id

    async def delete(self) -> None:
        """
        Deletes the test suite from the server.
        """
        # TODO: We don't want to have them pull

        await self._client.delete_test_suite(self.id)

    async def update(self) -> None:

        suite = await self._client.create_or_update_test_suite(
            self.id, self.title, self.description
        )

        # TODO: May want to abstract these two?
        await self._client.update_global_checks(
            self.id,
            json.dumps([gc.model_dump(exclude_none=True) for gc in self.global_checks]),
        )
        updated_tests = (
            await self._client.add_batch_tests(
                tests=[test.to_test_mutation_info(self.id) for test in self.tests],
                create_only=False,
            )
        ).batch_update_test.tests

        # Remove any tests that are no longer used after the update.
        await self._client.remove_old_tests(
            self.id, [test.test_id for test in updated_tests]
        )

    async def run(
        self,
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str = "gpt-4o",
        description: str = "Ran with PRL SDK.",
        wait_for_completion: bool = False,
    ) -> str:
        # TODO: Return a run object instead of just the ID.
        # TODO: Include a qa set id optionally.
        _default_parameters = _get_default_parameters()
        parameters = {**_default_parameters, **parameters}
        parameters["description"] = description
        parameters["model_under_test"] = model_under_test

        response = await self._client.start_run(self.id, parameters)

        if wait_for_completion:
            await wait_for_run_completion(response.start_run.run_id)

        return response.start_run.run_id
