import inspect
import json
import os
from time import time
from typing import Any, cast

from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm
from vals.graphql_client.client import Client
from vals.graphql_client.input_types import MetadataType, QuestionAnswerPairInputType
from vals.sdk.run import _get_default_parameters
from vals.sdk.suite import _upload_file, _validate_suite
from vals.sdk.v2 import patch
from vals.sdk.v2.run import Run
from vals.sdk.v2.types import (
    Check,
    ModelFunctionType,
    ModelFunctionWithFilesAndContextType,
    SimpleModelFunctionType,
    Test,
    TestSuiteMetadata,
)
from vals.sdk.v2.util import get_ariadne_client, md5_hash, parse_file_id, read_file


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

        return cls(
            id=suite_id,
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )

    @classmethod
    async def from_dict(cls, data: dict[str, Any]) -> "Suite":
        """
        Imports the test suite from a dictionary - useful if
        importing from the old format.
        """
        _validate_suite(data)

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
            )
            for test in data["tests"]
        ]
        return cls(
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )

    @classmethod
    async def from_json_file(cls, file_path: str) -> "Suite":
        """
        Imports the test suite from a local JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return await cls.from_dict(data)

    async def create(self) -> None:
        """
        Creates the test suite on the server.
        """
        if self.id is not None:
            raise Exception("This suite has already been created.")

        await self._validate_tests()

        # Create suite object on the server
        suite = await self._client.create_or_update_test_suite(
            "0", self.title, self.description
        )
        if suite.update_test_suite is None:
            raise Exception("Unable to update the test suite.")
        self.id = suite.update_test_suite.test_suite.id

        await self._upload_global_checks()
        await self._upload_files()
        await self._upload_tests(create_only=True)

    async def delete(self) -> None:
        """
        Deletes the test suite from the server.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to delete"
            )

        await self._client.delete_test_suite(self.id)

    async def update(self) -> None:
        """
        Pushes any local changes to the the test suite object to the server.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so there's nothing to update"
            )

        await self._client.create_or_update_test_suite(
            self.id, self.title, self.description
        )

        await self._upload_files()
        await self._upload_global_checks()
        await self._upload_tests(create_only=False)

        # Remove any tests that are no longer used after the update.
        await self._client.remove_old_tests(self.id, [test.id for test in self.tests])

    async def run(
        self,
        model_under_test: str | None = None,
        model_function: ModelFunctionType | None = None,
        run_name: str | None = None,
        wait_for_completion: bool = False,
        # TODO: Type parameters properly.
        parameters: dict[str, int | float | str | bool] = {},
    ) -> Run:
        """
        Runs the test suite. This can be used in one of two ways.

        First, if "model_under_test" is provided, and "model_function" is not, then we will run that stock model. The supported models
        are the same ones in the "Model Under Test" dropdown in the Web App - e.g., "gpt-4o", "claude-3-5-sonnet", etc.

        Second, if "model_function" is provided, then we will use that function to generate responses from the LLM locally.
        For example, your model function may look like this:

        def my_model_function(input: str) -> str:
            # Replace with your custom logic / prompt chains, etc.
            output = OpenAi.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": input}],
            )
            return output.choices[0].message.content

        When using the model_function mode, the model_under_test parameter is just used for bookkeeping purposes -
        it's what shows up as the model in the frontend.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet. Call suite.create() before calling suite.run()"
            )

        if model_under_test is None and model_function is None:
            raise Exception(
                "One of 'model_under_test' or 'model_function' must be provided."
            )

        # Use the default parameters, and then override with any user-provided parameters.
        _default_parameters = _get_default_parameters()
        parameters = {**_default_parameters, **parameters}
        if model_under_test is not None:
            parameters["model_under_test"] = model_under_test

        # Collate the local output pairs if we're using a model function.
        qa_set_id = None
        if model_function is not None:
            qa_set_id = await self._create_qa_set(
                model_function, parameters, model_under_test
            )

        # Start and pull the run.
        response = await self._client.start_run(
            self.id, parameters, qa_set_id, run_name
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
        if self.id is None:
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
                        file_id = _upload_file(self.id, file_path)
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
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        # Upload tests in batches of 100
        created_tests = []
        test_mutations = [test.to_test_mutation_info(self.id) for test in self.tests]
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

    async def _validate_tests(self) -> None:
        """Helper method to ensure that all operator strings are correct."""
        operators = (await self._client.get_operators()).operators
        operators_dict = {
            op.name_in_doc: op for op in operators
        }  # Easy lookup by name.

        for test in self.tests:
            for check in test.checks:
                if check.operator not in operators_dict:
                    raise ValueError(f"Invalid operator: {check.operator}")
                operator = operators_dict[check.operator]
                if operator.is_unary and (
                    check.criteria is None or check.criteria == ""
                ):
                    raise ValueError(
                        "Operator {operator} is unary, but no criteria was provided."
                    )

    async def _upload_global_checks(self) -> None:
        """
        Helper method to upload the global checks to the server.
        """
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        await self._client.update_global_checks(
            self.id,
            json.dumps([gc.model_dump(exclude_none=True) for gc in self.global_checks]),
        )

    async def _create_qa_set(
        self,
        model_function: ModelFunctionType,
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str | None = None,
    ) -> str | None:
        """
        Helper function to create a question-answer set from a model function.
        A question-answer set is just a set of inputs to the LLM, and the LLM's responses.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )

        # Pull latest suite data to ensure we're up to date
        updated_suite = await Suite.from_id(self.id)
        for field in self.__fields__:
            setattr(self, field, getattr(updated_suite, field))

        # Inspect the model function to determine if it takes 1 or 3 parameters
        # We dynamically change our call signature based on what the user passes in.
        sig = inspect.signature(model_function)
        num_params = len(sig.parameters)
        if num_params not in [1, 3]:
            raise ValueError("Model function must accept either 1 or 3 parameters")
        is_simple_model_function = num_params == 1

        # Query the LLM for each test.
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
                    test_id=test.id,
                )
            )

        response = await self._client.create_question_answer_set(
            self.id,
            qa_pairs,
            parameters,
            model_under_test or "sdk",
        )
        if response.create_question_answer_set is None:
            raise Exception("Unable to create the question-answer set.")

        return response.create_question_answer_set.question_answer_set.id
