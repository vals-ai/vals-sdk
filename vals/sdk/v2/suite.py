import inspect
import json
import os
from time import time
from typing import Any, Callable, cast, overload

import requests
from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm
from vals.graphql_client.client import Client
from vals.graphql_client.get_operators import GetOperatorsOperators
from vals.graphql_client.input_types import MetadataType, QuestionAnswerPairInputType
from vals.sdk.run import _get_default_parameters
from vals.sdk.util import _get_auth_token, be_host
from vals.sdk.v2 import patch
from vals.sdk.v2.run import Run
from vals.sdk.v2.types import (
    Check,
    ModelFunctionType,
    ModelFunctionWithFilesAndContextType,
    QuestionAnswerPair,
    SimpleModelFunctionType,
    Test,
    TestSuiteMetadata,
)
from vals.sdk.v2.util import get_ariadne_client, md5_hash, parse_file_id, read_file


class Suite(BaseModel):
    _id: str | None = None

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
    async def list_suites(cls) -> list[TestSuiteMetadata]:
        """
        Generate a list of all the test suites on the server.
        """
        client = get_ariadne_client()
        gql_response = await client.get_test_suites_with_count()
        gql_suites = gql_response.test_suites_with_count.test_suites
        return [TestSuiteMetadata.from_graphql(gql_suite) for gql_suite in gql_suites]

    @classmethod
    async def from_id(cls, suite_id: str) -> "Suite":
        """
        Create a new local test suite based on the data from the server.
        """

        client = get_ariadne_client()
        suites_list = await client.get_test_suite_data(suite_id)
        if len(suites_list.test_suites) == 0:
            raise Exception("Couldn't find suite with id: " + suite_id)

        suite_data = suites_list.test_suites[0]
        title = suite_data.title
        description = suite_data.description

        global_checks = []
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

        suite = cls(
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )
        suite._id = suite_id
        return suite

    @classmethod
    async def from_dict(cls, data: dict[str, Any]) -> "Suite":
        """
        Imports the test suite from a dictionary - useful if
        importing from the old format.
        """

        title = data["title"]
        description = data["description"]
        global_checks = [
            Check.from_graphql(check_dict)
            for check_dict in data.get("global_checks", [])
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
                context=test.get("context", {}),
            )
            for test in data["tests"]
        ]
        suite = cls(
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )
        await suite._validate_suite()
        return suite

    @classmethod
    async def from_json_file(cls, file_path: str) -> "Suite":
        """
        Imports the test suite from a local JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return await cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the test suite to a dictionary.
        """
        return self.model_dump(exclude_none=True, exclude_defaults=True)

    def to_json_file(self, file_path: str) -> None:
        """
        Converts the test suite to a JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    async def create(self) -> None:
        """
        Creates the test suite on the server.
        """
        if self._id is not None:
            raise Exception("This suite has already been created.")

        await self._validate_suite()

        # Create suite object on the server
        suite = await self._client.create_or_update_test_suite(
            "0", self.title, self.description
        )
        if suite.update_test_suite is None:
            raise Exception("Unable to update the test suite.")
        self._id = suite.update_test_suite.test_suite.id

        await self._upload_global_checks()
        await self._upload_files()
        await self._upload_tests(create_only=True)

    async def delete(self) -> None:
        """
        Deletes the test suite from the server.
        """
        if self._id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to delete"
            )

        await self._client.delete_test_suite(self._id)

    async def update(self) -> None:
        """
        Pushes any local changes to the the test suite object to the server.
        """
        if self._id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to update"
            )

        await self._client.create_or_update_test_suite(
            self._id, self.title, self.description
        )

        await self._upload_files()
        await self._upload_global_checks()
        await self._upload_tests(create_only=False)

        # Remove any tests that are no longer used after the update.
        await self._client.remove_old_tests(self._id, [test._id for test in self.tests])

    @overload
    async def run(
        self,
        model: str,
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: dict[str, int | float | str | bool] = {},
    ):
        """
        Runs based on a model string (e.g. "gpt-4o").
        """
        pass

    @overload
    async def run(
        self,
        model: ModelFunctionType,
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: dict[str, int | float | str | bool] = {},
    ):
        """
        Runs based on a model function. This can either be a simple model function, which just takes
        an input as a string and produces an output string, or a model function that takes in the input string, a dictionary of filename to file contents, and a context dictionary.
        """
        pass

    @overload
    async def run(
        self,
        model: list[QuestionAnswerPair],
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: dict[str, int | float | str | bool] = {},
    ) -> Run:
        """
        Runs based on on a list of question-answer pairs, which contain inputs and outputs.

        IMPORTANT NOTE: The combined "inputs" of the question-answer pairs (the input under test, the context, and the files) need
        to match exactly the inputs of the tests in the suite, otherwise, Vals will not know how to match to the auto-eval correctly.
        """
        pass

    async def run(
        self,
        model: str | ModelFunctionType | list[QuestionAnswerPair],
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: dict[str, int | float | str | bool] = {},
    ) -> Run:
        """
        Base method for running the test suite. See overloads for documentation.
        """
        if self._id is None:
            raise Exception(
                "This suite has not been created yet. Call suite.create() before calling suite.run()"
            )

        # Use the default parameters, and then override with any user-provided parameters.
        _default_parameters = _get_default_parameters()
        parameters = {**_default_parameters, **parameters}

        # Collate the local output pairs if we're using a model function.
        qa_set_id = None
        if isinstance(model, Callable):
            # Generate the QA pairs from the model function
            qa_pairs = await self._generate_qa_pairs_from_function(model)
            qa_set_id = await self._create_qa_set(qa_pairs, parameters, model_name)
            parameters["model_under_test"] = model_name
        elif isinstance(model, list):
            # Use the QA pairs we are already provided
            qa_set_id = await self._create_qa_set(
                [qa_pair.to_graphql() for qa_pair in model], parameters, model_name
            )
            parameters["model_under_test"] = model_name
        elif isinstance(model, str):
            # Just use a model string (e.g. "gpt-4o")
            parameters["model_under_test"] = model
            qa_set_id = None

        # Start and pull the run.
        response = await self._client.start_run(
            self._id, parameters, qa_set_id, run_name
        )
        if response.start_run is None:
            raise Exception("Unable to start the run.")

        run_id = response.start_run.run_id
        run = await Run.from_id(run_id)

        if wait_for_completion:
            await run.wait_for_run_completion()

        await run.pull()

        return run

    async def _upload_files(self):
        """
        Helper method to upload the files to the server.
        Uploads any tests.files_under_test that haven't been uploaded yet.
        """
        if self._id is None:
            raise Exception("This suite has not been created yet.")

        file_name_and_hash_to_file_id = {}

        # First, we populate the dictionary of files that have already been uploaded
        # for this test suite.
        for test in self.tests:
            for file_id in test._file_ids:
                _, name, _hash = parse_file_id(file_id)
                file_name_and_hash_to_file_id[(name, _hash)] = file_id

        # Next, for files we haven't uploaded yet, we upload them
        # and add them to the dictionary
        for test in self.tests:
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
                        file_id = self._upload_file(self._id, file_path)
                        file_name_and_hash_to_file_id[name_hash_tuple] = file_id

                    # Either way, we add the file id to the test.
                    file_ids.append(file_name_and_hash_to_file_id[name_hash_tuple])

                # Main downside of this approach is that we can only overwrite files
                # we can't add or remove just one file.
                test._file_ids = file_ids

    async def _upload_tests(self, create_only: bool = True) -> None:
        """
        Helper method to upload the tests to the server in batches of 100.
        """
        if self._id is None:
            raise Exception("This suite has not been created yet.")

        # Upload tests in batches of 100
        created_tests = []
        test_mutations = [test.to_test_mutation_info(self._id) for test in self.tests]
        for i in range(0, len(test_mutations), 100):
            batch = test_mutations[i : i + 100]
            batch_result = await self._client.add_batch_tests(
                tests=batch,
                create_only=create_only,
            )
            if batch_result.batch_update_test is None:
                raise Exception("Unable to add tests to the test suite.")
            created_tests.extend(batch_result.batch_update_test.tests)

        # Update the local suite with the new tests to ensure everything is in sync.
        self.tests = [Test.from_graphql_test(test) for test in created_tests]

    async def _validate_checks(
        self, check: Check, operators_dict: dict[str, GetOperatorsOperators]
    ) -> None:
        """Helper method to ensure that operator is correct"""
        if check.operator not in operators_dict:
            raise ValueError(f"Invalid operator: {check.operator}")
        operator = operators_dict[check.operator]
        if not operator.is_unary and (check.criteria is None or check.criteria == ""):
            raise ValueError(
                f"Operator {operator.name_in_doc} is unary, but no criteria was provided."
            )

    async def _validate_suite(self) -> None:
        """Helper method to ensure that all operator strings are correct. Most of the validation is done by Pydantic
        but this handles things Pydantic can't check for, like whether a file exists."""
        operators = (await self._client.get_operators()).operators
        operators_dict = {op.name_in_doc: op for op in operators}

        for test in self.tests:
            for check in test.checks:
                await self._validate_checks(check, operators_dict)

            for file_path in test.files_under_test:
                if not os.path.exists(file_path):
                    raise ValueError(f"File does not exist: {file_path}")
                if not os.path.isfile(file_path):
                    raise ValueError(f"Path is a directory: {file_path}")

        for check in self.global_checks:
            await self._validate_checks(check, operators_dict)

    async def _upload_global_checks(self) -> None:
        """
        Helper method to upload the global checks to the server.
        """
        if self._id is None:
            raise Exception("This suite has not been created yet.")

        await self._client.update_global_checks(
            self._id,
            json.dumps([gc.model_dump(exclude_none=True) for gc in self.global_checks]),
        )

    async def _generate_qa_pairs_from_function(
        self,
        model_function: ModelFunctionType,
    ) -> list[QuestionAnswerPairInputType]:
        if self._id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )
        # Pull latest suite data to ensure we're up to date
        updated_suite = await Suite.from_id(self._id)
        for field in self.__fields__:
            setattr(self, field, getattr(updated_suite, field))
        # Inspect the model function to determine if it takes 1 or 3 parameters
        # We dynamically change our call signature based on what the user passes in.
        sig = inspect.signature(model_function)
        num_params = len(sig.parameters)
        if num_params not in [1, 3]:
            raise ValueError("Model function must accept either 1 or 3 parameters")
        is_simple_model_function = num_params == 1

        qa_pairs: list[QuestionAnswerPairInputType] = []
        for test in tqdm(self.tests):
            files = {}
            for file_id in test._file_ids:
                _, file_name, _ = parse_file_id(file_id)
                files[file_name] = read_file(file_id)

            time_start = time()
            in_tokens_start = patch.in_tokens
            out_tokens_start = patch.out_tokens

            if is_simple_model_function:
                casted_model_function = cast(SimpleModelFunctionType, model_function)
                llm_output = casted_model_function(test.input_under_test)
            else:
                casted_model_function = cast(
                    ModelFunctionWithFilesAndContextType, model_function
                )
                llm_output = casted_model_function(
                    test.input_under_test, files, test.context
                )

            time_end = time()
            in_tokens_end = patch.in_tokens
            out_tokens_end = patch.out_tokens

            qa_pairs.append(
                QuestionAnswerPairInputType(
                    input_under_test=test.input_under_test,
                    file_ids=test._file_ids,
                    context=test.context,
                    llm_output=llm_output,
                    metadata=MetadataType(
                        in_tokens=in_tokens_end - in_tokens_start,
                        out_tokens=out_tokens_end - out_tokens_start,
                        duration_seconds=time_end - time_start,
                    ),
                    test_id=test._id,
                )
            )
        return qa_pairs

    async def _create_qa_set(
        self,
        qa_pairs: list[QuestionAnswerPairInputType],
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str | None = None,
        batch_size: int = 50,
    ) -> str | None:
        """
        Helper function to create a question-answer set from a model function.
        A question-answer set is just a set of inputs to the LLM, and the LLM's responses.
        """
        if self._id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )

        response = await self._client.create_question_answer_set(
            self._id,
            [],
            parameters,
            model_under_test or "sdk",
        )
        set_id = response.create_question_answer_set.question_answer_set.id

        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i : i + batch_size]
            await self._client.batch_add_question_answer_pairs(set_id, batch)
        if response.create_question_answer_set is None:
            raise Exception("Unable to create the question-answer set.")

        return set_id

    def _upload_file(self, suite_id: str, file_path: str) -> str:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{be_host()}/upload_file/?test_suite_id={suite_id}",
                files={"file": f},
                headers={"Authorization": _get_auth_token()},
            )
            if response.status_code != 200:
                raise Exception(f"Failed to upload file {file_path}")
            return response.json()["file_id"]
