import inspect
import json
import os
from io import BytesIO
from time import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm
from vals.graphql_client.client import Client
from vals.graphql_client.get_test_suites import GetTestSuitesTestSuites
from vals.graphql_client.input_types import MetadataType, QuestionAnswerPairInputType
from vals.sdk.run import _get_default_parameters
from vals.sdk.suite import _upload_file, _validate_suite
from vals.sdk.v2.run import Run
from vals.sdk.v2.types import Check, Test
from vals.sdk.v2.util import get_ariadne_client, md5_hash, parse_file_id

SimpleModelFunctionType = Callable[[str], str]
ModelFunctionWithFilesAndContextType = Callable[
    [str, list[BytesIO], dict[str, Any]], str
]
ModelFunctionType = SimpleModelFunctionType | ModelFunctionWithFilesAndContextType

from vals.sdk.v2 import patch


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

    @classmethod
    async def from_id(cls, suite_id: str) -> "Suite":
        """
        Updates the local fields to match the suite on the server.
        """

        client = get_ariadne_client()
        suites_list = await client.get_test_suite_data(suite_id)
        if len(suites_list.test_suites) == 0:
            raise Exception("Couldn't find suite with id: " + suite_id)

        suite_data = suites_list.test_suites[0]
        title = suite_data.title
        description = suite_data.description

        if suite_data.global_checks is not None:
            global_checks = [
                Check.from_graphql(check)
                for check in json.loads(suite_data.global_checks)
            ]

        test_data = await client.get_test_data(suite_id)

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
        if self.id is not None:
            raise Exception("This suite has already been created.")

        # TODO: Client side validation of test suite
        # Main / only thing we need to check is that the operators are from
        # the valid set of operators - and this will make more sense
        # after Antoine finishes his changes.

        # TODO: Upload files

        suite = await self._client.create_or_update_test_suite(
            "0", self.title, self.description
        )
        self.id = suite.update_test_suite.test_suite.id

        await self._client.update_global_checks(
            self.id,
            json.dumps([gc.model_dump(exclude_none=True) for gc in self.global_checks]),
        )

        await self._upload_files()

        # TODO: Batch in batches of 100?
        created_tests = await self._client.add_batch_tests(
            tests=[test.to_test_mutation_info(self.id) for test in self.tests],
            create_only=True,
        )

        # TODO: If uploading tests fails, we should remove the suite that has
        # already been created.

        # TODO: Set test id locally - or cross version id.

    async def delete(self) -> None:
        """
        Deletes the test suite from the server.
        """
        # TODO: We don't want to have them pull
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to delete"
            )

        await self._client.delete_test_suite(self.id)

    async def update(self) -> None:
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to update"
            )

        suite = await self._client.create_or_update_test_suite(
            self.id, self.title, self.description
        )

        await self._upload_files()
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

    async def _create_qa_set(
        self,
        model_function: ModelFunctionType,
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str | None = None,
    ) -> str | None:
        """
        Creates a QA set for this test suite using the provided model function.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )
        # Pull latest suite data to ensure we're up to date
        updated_suite = await Suite.from_id(self.id)
        # Update all fields from the updated suite
        for field in self.__fields__:
            setattr(self, field, getattr(updated_suite, field))

        # Inspect the model function to determine if it takes 1 or 3 parameters
        sig = inspect.signature(model_function)
        num_params = len(sig.parameters)
        if num_params not in [1, 3]:
            raise ValueError("Model function must accept either 1 or 3 parameters")
        is_simple_model_function = num_params == 1

        qa_pairs: list[QuestionAnswerPairInputType] = []
        for test in tqdm(self.tests):
            time_start = time()

            in_tokens_start = patch.in_tokens
            out_tokens_start = patch.out_tokens
            if is_simple_model_function:
                llm_output = model_function(test.input_under_test)  # type: ignore
            else:
                llm_output = model_function(test.input, test.files_under_test, test.context)  # type: ignore
            time_end = time()
            in_tokens_end = patch.in_tokens
            out_tokens_end = patch.out_tokens

            qa_pairs.append(
                QuestionAnswerPairInputType(
                    input_under_test=test.input_under_test,
                    file_ids=test.file_ids,
                    context=test.context,
                    llm_output=llm_output,
                    metadata=MetadataType(
                        in_tokens=in_tokens_end - in_tokens_start,
                        out_tokens=out_tokens_end - out_tokens_start,
                        duration_seconds=time_end - time_start,
                    ),
                    test_id=test.id,
                )
            )

        response = await self._client.create_question_answer_set(
            self.id,
            qa_pairs,
            parameters,
            model_under_test or "sdk",
        )

        return response.create_question_answer_set.question_answer_set.id

    async def run(
        self,
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str = "gpt-4o",
        model_function: ModelFunctionType | None = None,
        run_name: str | None = None,
        wait_for_completion: bool = False,
    ) -> Run:
        if self.id is None:
            raise Exception(
                "This suite has not been created yet. Do suite.create() before calling suite.run()"
            )
        # TODO: Include a qa set id optionally.
        _default_parameters = _get_default_parameters()
        parameters = {**_default_parameters, **parameters}
        parameters["model_under_test"] = model_under_test

        qa_set_id = None
        if model_function is not None:
            qa_set_id = await self._create_qa_set(
                model_function, parameters, model_under_test
            )

        response = await self._client.start_run(
            self.id, parameters, qa_set_id, run_name
        )

        run_id = response.start_run.run_id

        run = await Run.from_id(run_id)

        if wait_for_completion:
            await run.wait_for_run_completion()

        await run.pull()

        return run

    async def _upload_files(self):
        # Map from file path to file id
        # Dictionary so we don't upload the same file multiple times.

        file_name_and_hash_to_file_id = {}

        # First, we populate the dictionary of files that have already been uploaded
        # for this test suite.
        for test in self.tests:
            if len(test.file_ids) != 0:
                for file_id in test.file_ids:
                    _, name, _hash = parse_file_id(file_id)
                    file_name_and_hash_to_file_id[(name, _hash)] = file_id

        # Next, for files we haven't uploaded yet, we upload them
        # and add them to the dictionary
        for test in self.tests:
            # If they haven't specified any local files, we don't need to do anything
            if len(test.files_under_test) != 0:
                file_ids = []
                for file_path in test.files_under_test:
                    if not os.path.exists(file_path):
                        raise Exception(f"File does not exist: {file_path}")

                    # First, we compute the filename and the hash of its contents
                    with open(file_path, "rb") as f:
                        file_hash = md5_hash(f)
                    filename = os.path.basename(file_path)
                    name_hash_tuple = (filename, file_hash)

                    # If the file hasn't been uploaded yet, we upload it
                    if name_hash_tuple not in file_name_and_hash_to_file_id:
                        file_id = await _upload_file(self.id, file_path)
                        file_name_and_hash_to_file_id[name_hash_tuple] = file_id
                        print("Uploaded file: ", file_id)

                    # Either way, we add the file id to the test.
                    file_ids.append(file_name_and_hash_to_file_id[name_hash_tuple])

                # Main downside of this approach is that we can only overwrite files
                # we can't add or remove just one file.
                test.file_ids = file_ids

    @classmethod
    async def from_json(cls, json_data: dict[str, Any]) -> "Suite":
        _validate_suite(json_data)

        title = json_data["title"]
        description = json_data["description"]
        global_checks = [
            Check.from_graphql(check_dict)
            for check_dict in json_data.get("global_checks", [])
        ]
        tests = [
            Test(
                input_under_test=test["input_under_test"],
                checks=[
                    Check.from_graphql(check_dict) for check_dict in test["checks"]
                ],
                golden_output=test.get("golden_output", ""),
                tags=test.get("tags", []),
                files_under_test=test.get("files_under_test", []),
            )
            for test in json_data["tests"]
        ]
        return cls(
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )
