import asyncio
import concurrent.futures._base
import inspect
import json
import os
from time import time
from typing import Any, Callable, cast, overload

import requests
import vals.sdk.patch as patch
from pydantic import BaseModel, PrivateAttr
from tqdm import tqdm
from vals.graphql_client.client import UNSET, Client
from vals.graphql_client.get_operators import GetOperatorsOperators
from vals.graphql_client.input_types import (
    LocalEvalUploadInputType,
    MetadataType,
    ParameterInputType,
    QuestionAnswerPairInputType,
)
from vals.sdk.inspect_wrapper import InspectWrapper
from vals.sdk.run import Run
from vals.sdk.types import (
    Check,
    File,
    ModelCustomOperatorFunctionType,
    ModelFunctionType,
    ModelFunctionWithFilesAndContextType,
    OperatorInput,
    QuestionAnswerPair,
    RunParameters,
    RunStatus,
    SimpleModelFunctionType,
    Test,
    TestSuiteMetadata,
)
from vals.sdk.util import (
    _get_auth_token,
    be_host,
    download_files_bulk,
    fe_host,
    get_ariadne_client,
    md5_hash,
    parse_file_id,
    read_file,
)


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
    async def list_suites(
        cls, limit=50, offset=0, search=""
    ) -> list[TestSuiteMetadata]:
        """
        Generate a list of all the test suites on the server.

        limit: Total number to return
        offset: Start list at this index
        """
        client = get_ariadne_client()
        gql_response = await client.get_test_suites_with_count(
            limit=limit,
            offset=offset,
            search=search,
        )
        gql_suites = gql_response.test_suites_with_count.test_suites
        return [TestSuiteMetadata.from_graphql(gql_suite) for gql_suite in gql_suites]

    @classmethod
    async def from_id(
        cls,
        suite_id: str,
        download_files: bool = False,
        download_path: str | None = None,
        max_concurrent_downloads: int = 50,
    ) -> "Suite":
        """
        Create a new local test suite based on the data from the server.
        """
        client = get_ariadne_client()
        suite_query = await client.get_test_suite_data(suite_id)
        suite_data = suite_query.test_suite

        title = suite_data.title
        description = suite_data.description

        global_checks = []
        if suite_data.global_checks is not None:
            global_checks = [
                Check.from_graphql(check)
                for check in json.loads(suite_data.global_checks)
            ]

        tests = []
        offset = 0
        page_size = 200
        have_pulled_all_tests = False

        while not have_pulled_all_tests:
            test_data = await client.get_test_data(suite_id, offset, page_size)

            for graphql_test in test_data.tests_with_count.tests:
                test = Test.from_graphql_test(graphql_test)
                tests.append(test)

            if len(tests) >= test_data.tests_with_count.count:
                have_pulled_all_tests = True
            else:
                offset += page_size

        suite = cls(
            id=suite_id,
            title=title,
            description=description,
            global_checks=global_checks,
            tests=tests,
        )

        if download_files:
            file_ids = []
            path = suite.title if download_path is None else download_path

            for test in tests:
                file_ids.extend([file.file_id for file in test.files_under_test])

            if len(file_ids) > 0:
                name_to_path_map = await download_files_bulk(
                    file_ids, path, max_concurrent_downloads=max_concurrent_downloads
                )
                test.files_under_test = [
                    File(
                        file_name=file.file_name,
                        file_id=file.file_id,
                        path=name_to_path_map[file.file_name],
                    )
                    for file in test.files_under_test
                ]

        return suite

    @classmethod
    async def from_dict(cls, data: dict[str, Any]) -> "Suite":
        """
        Imports the test suite from a dictionary - useful if
        importing from the old format.

        Does not create the suite - you must call suite.create() after construction.
        """

        title = data["title"]
        description = data.get("description", "")
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

    @classmethod
    async def from_inspect_json_file(
        cls, file_path: str, suite_title: str, suite_description: str
    ) -> "Suite":
        """
        Imports the test suite from a local JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        suite = {"title": suite_title, "description": suite_description, "tests": []}

        for test in data:
            inspect_context = {
                "target": test.get("target", ""),
                "task_state": {
                    k: v for k, v in test.items() if k != "target" and k != "input"
                },
            }

            files_under_test = test.get("metadata", {}).get("documents_to_upload", [])
            context = {"inspect_context": inspect_context}

            suite["tests"].append(
                {
                    "input_under_test": test["input"],
                    "checks": [],
                    "files_under_test": files_under_test,
                    "context": context,
                    "tags": test.get("metadata", {}).get("tags", []),
                }
            )

        return await cls.from_dict(suite)

    @property
    def url(self):
        return f"{fe_host()}/suites/{self.id}"

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
            f.write(self.to_json_string())

    def to_csv_string(self) -> str:
        """
        Like to_csv_file, but returns the file as a string instead of writing it to a file.
        """
        if not self.id:
            raise Exception("Suite has not been created yet.")

        url = f"{be_host()}/export_tests_to_file/?suite_id={self.id}"
        response = requests.post(
            url,
            headers={"Authorization": _get_auth_token()},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to export tests: {response.text}")

        return response.text

    def to_json_string(self) -> str:
        """
        Converts the test suite to a JSON string.
        """
        if not self.id:
            raise Exception("Suite has not been created yet.")

        url = f"{be_host()}/export_tests_to_json/?suite_id={self.id}"
        response = requests.post(
            url,
            headers={"Authorization": _get_auth_token()},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to export tests: {response.text}")

        return response.text

    def to_csv_file(self, file_path: str) -> None:
        """
        Converts the test suite to a CSV file.
        """
        with open(file_path, "w") as f:
            f.write(self.to_csv_string())

    async def create(self, force_creation: bool = False) -> None:
        """
        Creates the test suite on the server.

        force_creation: If True, will create a new test suite, even if it
        already exists.
        """
        if force_creation:
            self.id = None

        if self.id is not None:
            raise Exception("This suite has already been created.")

        await self._validate_suite()

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

    async def update(self, upload_files_path: str | None = None) -> None:
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

        path = self.title if upload_files_path is None else upload_files_path

        await self._upload_files(path)
        await self._upload_global_checks()
        await self._upload_tests(create_only=False)

        # Remove any tests that are no longer used after the update.
        await self._client.remove_old_tests(self.id, [test._id for test in self.tests])

    @overload
    async def run(
        self,
        model: str,
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: RunParameters | None = None,
    ) -> Run:
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
        parameters: RunParameters | None = None,
        upload_concurrency: int | None = None,
    ) -> Run:
        """
        Runs based on a model function. This can either be a simple model function, which just takes
        an input as a string and produces an output string, or a model function that takes in the input string,
        a dictionary of filename to file contents, and a context dictionary.

        Args:
            upload_concurrency: How frequently to upload QA pairs to the server. Defaults to parameters.parallelism.
        """
        pass

    @overload
    async def run(
        self,
        model: list[QuestionAnswerPair],
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: RunParameters | None = None,
    ) -> Run:
        """
        Runs based on on a list of question-answer pairs, which contain inputs and outputs.

        IMPORTANT NOTE: The combined "inputs" of the question-answer pairs (the input under test, the context,
        and the files) need to match exactly the inputs of the tests in the suite, otherwise, Vals will not
        know how to match to the auto-eval correctly.
        """
        pass

    @overload
    async def run(
        self,
        model: InspectWrapper,
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: RunParameters | None = None,
    ) -> Run:
        """
        Runs based on on an InspectWrapper object.
        """
        pass

    async def _safe_local_execution(self, coro, run_id: str | None = None):
        """Wrapper to handle interruptions during local execution"""
        try:
            return await coro
        except (
            KeyboardInterrupt,
            asyncio.CancelledError,
            concurrent.futures._base.CancelledError,
        ):
            if run_id:
                await self._client.update_run_status(
                    run_id=run_id, status=RunStatus.ERROR
                )
            raise
        except Exception as e:
            if run_id:
                await self._client.update_run_status(
                    run_id=run_id, status=RunStatus.ERROR
                )
            raise

    async def run(
        self,
        model: str | ModelFunctionType | list[QuestionAnswerPair] | InspectWrapper,
        model_name: str = "sdk",
        run_name: str | None = None,
        wait_for_completion: bool = False,
        parameters: RunParameters | None = None,
        upload_concurrency: int | None = None,
        custom_operators: list[ModelCustomOperatorFunctionType] | None = None,
        eval_model_name: str | None = None,
        run_id: str | None = None,
        qa_set_id: str | None = None,
        remaining_tests: list[Test] | None = None,
        uploaded_qa_pairs: list[QuestionAnswerPairInputType] | None = None,
    ) -> Run:
        """
        Base method for running the test suite. See overloads for documentation.

        Args:
            upload_concurrency: How frequently to upload QA pairs to the server.
                         Defaults to parameters.parallelism.
            custom_operator: A custom operator function that takes in an OperatorInput and returns an OperatorOutput.
        """
        if self.id is None:
            raise Exception(
                "This suite has not been created yet. Call suite.create() before calling suite.run()"
            )
        if parameters is None:
            parameters = RunParameters()

        if custom_operators is None:
            custom_operators = []

        if eval_model_name is not None:
            parameters.eval_model = eval_model_name

        if upload_concurrency is None:
            upload_concurrency = parameters.parallelism

        if uploaded_qa_pairs is None:
            uploaded_qa_pairs = []

        if isinstance(model, str) and len(custom_operators) > 0:
            raise Exception(
                "Custom operator functions are not supported when using a model string (e.g. a nonlocal model)."
            )

        if isinstance(model, InspectWrapper):
            model_name = model.model_name
            eval_model_name = model.eval_model_name
            inspect_wrapper = model
            custom_operators = model.get_custom_operators()
            model = model.get_custom_model()

        parameter_json = parameters.model_dump()
        del parameter_json[
            "parallelism"
        ]  # Need to explicitly map parallelism to maximum_threads
        parameter_input = ParameterInputType(
            **parameter_json,
            model_under_test="",
            maximum_threads=parameters.parallelism,
        )

        # Collate the local output pairs if we're using a model function.
        if isinstance(model, Callable) or isinstance(model, InspectWrapper):
            # Generate the QA pairs from the model function
            parameter_input.model_under_test = model_name
            if run_id is None and qa_set_id is None:
                qa_set_id, run_id = await self._create_empty_qa_set(
                    parameter_input.model_dump(), model_name, run_name
                )
                print(f"Created run with run id: {run_id}")
            elif (
                run_id is None
                and qa_set_id is not None
                or run_id is not None
                and qa_set_id is None
            ):
                raise Exception("Must provide both run_id and qa_set_id or neither.")

            # Wrap the QA pair generation with safe execution
            uploaded_qa_pairs += await self._safe_local_execution(
                self._generate_qa_pairs_from_function(
                    model,
                    parameter_input,
                    qa_set_id,
                    upload_concurrency,
                    remaining_tests,
                ),
                run_id,
            )
        elif isinstance(model, list):
            # Use the QA pairs we are already provided

            parameter_input.model_under_test = model_name
            qa_pairs = model
            uploadable_qa_pairs = [qa_pair.to_graphql() for qa_pair in qa_pairs]
            qa_set_id, run_id = await self._create_empty_qa_set(
                parameter_input.model_dump(), model_name, run_name
            )
            uploaded_qa_pairs = []
            for i in range(0, len(uploadable_qa_pairs), upload_concurrency):
                batch_to_upload = uploadable_qa_pairs[i : i + upload_concurrency]
                response = await self._client.batch_add_question_answer_pairs(
                    qa_set_id, batch_to_upload
                )
                uploaded_qa_pairs.extend(
                    [
                        await self._deserialize_qa_pair_file_ids(qa_pair)
                        for qa_pair in response.batch_add_question_answer_pairs.question_answer_pairs
                    ]
                )

        elif isinstance(model, str):
            # Just use a model string (e.g. "gpt-4o")
            parameter_input.model_under_test = model
            qa_set_id = None
        else:
            raise Exception(f"Got unexpected type for model: {type(model)}")

        # Start and pull the run.
        if len(custom_operators) > 0:
            # Wrap the local eval with safe execution
            await self._safe_local_execution(
                self._upload_local_eval(
                    custom_operators,
                    parameter_input,
                    qa_set_id,
                    uploaded_qa_pairs,
                    upload_concurrency,
                ),
                run_id,
            )

        response = await self._client.start_run(
            self.id,
            parameter_input,
            qa_set_id or UNSET,
            run_name or UNSET,
            run_id or UNSET,
        )

        if response.start_run is None:
            raise Exception("Unable to start the run.")

        run_id = response.start_run.run_id

        run = await Run.from_id(run_id)

        if wait_for_completion:
            await run.wait_for_run_completion()

        await run.pull()

        return run

    async def _upload_files(self, upload_files_path: str | None = None):
        """
        Helper method to upload the files to the server.
        Uploads any tests.files_under_test that haven't been uploaded yet.
        """
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        for test in self.tests:
            # Convert files_under_test to a list of File objects if they're not already
            normalized_files = []
            for file_item in test.files_under_test:
                if isinstance(file_item, str):
                    # Handle string path
                    normalized_files.append(
                        File(file_name=os.path.basename(file_item), path=file_item)
                    )
                elif isinstance(file_item, dict):
                    # Handle dictionary representation
                    file_name = file_item.get("file_name") or os.path.basename(
                        file_item.get("path", "")
                    )
                    normalized_files.append(
                        File(
                            file_name=file_name,
                            file_id=file_item.get("file_id"),
                            path=file_item.get("path"),
                            hash=file_item.get("hash"),
                        )
                    )
                elif isinstance(file_item, File):
                    # Already a File object
                    normalized_files.append(file_item)
                else:
                    raise ValueError(f"Unexpected file type: {type(file_item)}")

            # Replace the original files_under_test with normalized File objects
            test.files_under_test = normalized_files

            # Process each file
            for file in test.files_under_test:
                # Determine the file path
                file_path = None

                # Case 1: File has a path
                if file.path is not None and os.path.exists(file.path):
                    file_path = file.path
                # Case 2: File has no path but we have upload_files_path
                elif upload_files_path is not None:
                    # Try standard path
                    potential_path = os.path.join(upload_files_path, file.file_name)
                    if os.path.exists(potential_path):
                        file_path = potential_path
                    # Try hash subdirectory path if hash exists
                    elif file.hash is not None:
                        hash_path = os.path.join(
                            upload_files_path, file.hash, file.file_name
                        )
                        if os.path.exists(hash_path):
                            file_path = hash_path

                # Skip if we couldn't find the file
                if file_path is None:
                    continue

                # Calculate file hash
                with open(file_path, "rb") as f:
                    file_hash = md5_hash(f)

                # Case: File is new or hash has changed
                if file.file_id is None or file.hash != file_hash:
                    # Check if this hash already exists in another file in the test
                    if file_hash in [
                        f.hash
                        for f in test.files_under_test
                        if f != file and f.hash is not None
                    ]:
                        print(
                            f"File {file.file_name} with same hash already exists in the test suite."
                        )
                        continue

                    file.file_id = self._upload_file(self.id, file_path)
                    file.hash = file_hash

            # Update file IDs for the test
            test._file_ids = [
                file.file_id
                for file in test.files_under_test
                if file.file_id is not None
            ]

    async def _upload_tests(self, create_only: bool = True) -> None:
        """
        Helper method to upload the tests to the server in batches of 100.
        """
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        # Upload tests in batches of 100
        created_tests = []
        test_mutations = [test.to_test_mutation_info(self.id) for test in self.tests]

        with tqdm(total=len(test_mutations), desc="Uploading tests") as pbar:
            for i in range(0, len(test_mutations), 25):
                batch = test_mutations[i : i + 25]
                batch_result = await self._client.add_batch_tests(
                    tests=batch,
                    create_only=create_only,
                )
                if batch_result.batch_update_test is None:
                    raise Exception("Unable to add tests to the test suite.")
                created_tests.extend(batch_result.batch_update_test.tests)
                pbar.update(len(batch))

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
        base_operators_dict = {op.name_in_doc: op for op in operators}
        custom_operators = (
            await self._client.get_active_custom_operators(0, 200)
        ).custom_operators.operators
        custom_operators_dict = {op.name: op for op in custom_operators}
        operators_dict = {**base_operators_dict, **custom_operators_dict}

        for test in self.tests:
            for check in test.checks:
                await self._validate_checks(check, operators_dict)

            for file in test.files_under_test:
                if isinstance(file, str):
                    file_path = file
                elif isinstance(file, dict):
                    file_path = file["path"]
                elif isinstance(file, File):
                    file_path = file.path
                else:
                    raise ValueError(f"Unexpected file type: {type(file)}")
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
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        await self._client.update_global_checks(
            self.id,
            [gc.to_graphql_input() for gc in self.global_checks],
        )

    async def _create_empty_qa_set(
        self,
        parameters: dict[str, int | float | str | bool] = {},
        model_under_test: str | None = None,
        run_name: str | None = None,
    ) -> str:
        """Creates an empty QA set and returns its ID."""
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )

        response = await self._client.create_question_answer_set(
            self.id,
            [],
            parameters,
            model_under_test or "sdk",
            run_name,
        )

        if response.create_question_answer_set is None:
            raise Exception("Unable to create the question-answer set.")

        return (
            response.create_question_answer_set.question_answer_set.id,
            response.create_question_answer_set.run_id,
        )

    async def _upload_local_eval(
        self,
        custom_operators: list[ModelCustomOperatorFunctionType],
        parameter_input: ParameterInputType,
        qa_set_id: str,
        qa_pairs: list[QuestionAnswerPairInputType],
        upload_concurrency: int,
    ) -> str:
        """Process QA pairs through custom operator and upload results in batches."""
        if self.id is None:
            raise Exception("This suite has not been created yet.")

        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(parameter_input.maximum_threads)
        operator_results = []
        last_uploaded_idx = 0
        upload_tasks = []

        async def process_and_upload_eval(qa_pair):
            async with semaphore:
                # Process through all custom operators
                results = await asyncio.gather(
                    *(
                        self._process_single_local_eval(custom_operator, qa_pair)
                        for custom_operator in custom_operators
                    )
                )
                operator_results.extend(results)
                pbar.update(1)

                # Upload batch when we reach batch_size
                nonlocal last_uploaded_idx
                while len(operator_results) >= last_uploaded_idx + upload_concurrency:
                    pbar.set_description(
                        f"Processing operator evaluations (uploading evaluations {last_uploaded_idx} to {last_uploaded_idx + upload_concurrency})"
                    )
                    batch_to_upload = operator_results[
                        last_uploaded_idx : last_uploaded_idx + upload_concurrency
                    ]
                    task = asyncio.create_task(
                        self._client.upload_local_evaluation(qa_set_id, batch_to_upload)
                    )
                    upload_tasks.append(task)
                    last_uploaded_idx += upload_concurrency

        # Process all QA pairs concurrently with limited concurrency
        with tqdm(total=len(qa_pairs), desc="Processing operator evaluations") as pbar:
            await asyncio.gather(*(process_and_upload_eval(qa) for qa in qa_pairs))

        # Upload any remaining results
        if last_uploaded_idx < len(operator_results):
            batch_to_upload = operator_results[last_uploaded_idx:]
            task = asyncio.create_task(
                self._client.upload_local_evaluation(qa_set_id, batch_to_upload)
            )
            upload_tasks.append(task)

        # Wait for all uploads to complete
        if upload_tasks:
            await asyncio.gather(*upload_tasks)

    async def _process_single_test(
        self,
        test: Test,
        model_function: ModelFunctionType,
        is_simple_model_function: bool,
    ) -> QuestionAnswerPairInputType:
        """Inner implementation of process_single_test"""
        files = {}
        for file_id in test._file_ids:
            _, file_name, _ = parse_file_id(file_id)
            files[file_name] = read_file(file_id)

        time_start = time()
        in_tokens_start = patch.in_tokens
        out_tokens_start = patch.out_tokens

        # Get the current event loop
        loop = asyncio.get_running_loop()

        if is_simple_model_function:
            casted_model_function = cast(SimpleModelFunctionType, model_function)
            if inspect.iscoroutinefunction(casted_model_function):
                # Already async, just await it
                output = await casted_model_function(test.input_under_test)
            else:
                # Run synchronous function in a thread pool to avoid blocking
                output = await loop.run_in_executor(
                    None, lambda: casted_model_function(test.input_under_test)
                )
        else:
            casted_model_function = cast(
                ModelFunctionWithFilesAndContextType, model_function
            )
            if inspect.iscoroutinefunction(casted_model_function):
                # Already async, just await it
                output = await casted_model_function(
                    test.input_under_test, files, test.context
                )
            else:
                # Run synchronous function in a thread pool to avoid blocking
                output = await loop.run_in_executor(
                    None,
                    lambda: casted_model_function(
                        test.input_under_test, files, test.context
                    ),
                )

        time_end = time()
        in_tokens_end = patch.in_tokens
        out_tokens_end = patch.out_tokens

        return await self._process_model_output(
            output,
            test,
            test._file_ids,
            time_start,
            time_end,
            in_tokens_start,
            out_tokens_start,
            in_tokens_end,
            out_tokens_end,
        )

    async def _process_single_local_eval(
        self,
        custom_operator: ModelCustomOperatorFunctionType,
        qa_pair: QuestionAnswerPairInputType,
    ) -> LocalEvalUploadInputType:
        """Inner implementation of process_single_local_eval"""
        files = {}
        for file_id in qa_pair.file_ids:
            _, file_name, _ = parse_file_id(file_id)
            files[file_name] = read_file(file_id)

        operator_input = OperatorInput(
            input=qa_pair.input_under_test,
            model_output=qa_pair.llm_output,
            files=files,
            context=qa_pair.context,
            output_context=qa_pair.output_context,
        )

        # Get the current event loop
        loop = asyncio.get_running_loop()

        if inspect.iscoroutinefunction(custom_operator):
            # Already async, just await it
            result = await custom_operator(operator_input)
        else:
            # Run synchronous function in a thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, lambda: custom_operator(operator_input)
            )

        return LocalEvalUploadInputType(
            questionAnswerPairId=qa_pair.id,
            score=result.score,
            feedback=result.explanation,
            name=result.name,
        )

    async def _deserialize_qa_pair_file_ids(self, qa_pair):
        """Helper method to deserialize file_ids in a QA pair if they're stored as a string."""
        if hasattr(qa_pair, "file_ids") and isinstance(qa_pair.file_ids, str):
            qa_pair.file_ids = json.loads(qa_pair.file_ids)
        return qa_pair

    async def _generate_qa_pairs_from_function(
        self,
        model_function: ModelFunctionType,
        parameter_input: ParameterInputType,
        qa_set_id: str,
        upload_concurrency: int,
        remaining_tests: list[Test] | None = None,
    ) -> list[QuestionAnswerPairInputType]:
        if self.id is None:
            raise Exception(
                "This suite has not been created yet, so we can't create a QA set from it."
            )
        # Pull latest suite data to ensure we're up to date
        updated_suite = await Suite.from_id(self.id)
        for field in self.__fields__:
            setattr(self, field, getattr(updated_suite, field))

        # Inspect the model function
        sig = inspect.signature(model_function)
        num_params = len(sig.parameters)
        if num_params not in [1, 3]:
            raise ValueError("Model function must accept either 1 or 3 parameters")
        is_simple_model_function = num_params == 1

        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(parameter_input.maximum_threads)
        last_uploaded_idx = 0
        upload_tasks = []
        qa_pairs: list[QuestionAnswerPairInputType] = []

        async def process_and_upload_test(test):
            async with semaphore:
                result = await self._process_single_test(
                    test, model_function, is_simple_model_function
                )
                qa_pairs.append(result)
                pbar.update(1)

                # Upload batch when we reach batch_size
                nonlocal last_uploaded_idx
                while len(qa_pairs) >= last_uploaded_idx + upload_concurrency:
                    pbar.set_description(
                        f"Processing tests (uploading outputs {last_uploaded_idx} to {last_uploaded_idx + upload_concurrency})"
                    )
                    batch_to_upload = qa_pairs[
                        last_uploaded_idx : last_uploaded_idx + upload_concurrency
                    ]
                    task = asyncio.create_task(
                        self._client.batch_add_question_answer_pairs(
                            qa_set_id, batch_to_upload
                        )
                    )
                    upload_tasks.append(task)
                    last_uploaded_idx += upload_concurrency

        # Process all tests concurrently with limited concurrency

        with tqdm(
            total=(
                len(self.tests) if remaining_tests is None else len(remaining_tests)
            ),
            desc="Processing tests",
        ) as pbar:
            await asyncio.gather(
                *(
                    process_and_upload_test(test)
                    for test in (
                        self.tests if remaining_tests is None else remaining_tests
                    )
                )
            )

        # Upload any remaining pairs
        if last_uploaded_idx < len(qa_pairs):
            remaining_pairs = qa_pairs[last_uploaded_idx:]
            task = asyncio.create_task(
                self._client.batch_add_question_answer_pairs(qa_set_id, remaining_pairs)
            )
            upload_tasks.append(task)

        # Wait for all uploads to complete
        uploaded_qa_pairs = []
        if upload_tasks:
            uploaded_qa_pairs.extend(
                [
                    await self._deserialize_qa_pair_file_ids(qa_pair)
                    for result in await asyncio.gather(*upload_tasks)
                    for qa_pair in result.batch_add_question_answer_pairs.question_answer_pairs
                ]
            )

        await self._client.mark_question_answer_set_as_complete(qa_set_id)

        return uploaded_qa_pairs

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

    async def _process_model_output(
        self,
        output: str | dict | QuestionAnswerPairInputType,
        test: Test,
        file_ids: list[str],
        time_start: float,
        time_end: float,
        in_tokens_start: int,
        out_tokens_start: int,
        in_tokens_end: int,
        out_tokens_end: int,
    ) -> QuestionAnswerPairInputType:
        """Helper function to process model output into a QuestionAnswerPairInputType.
        The output can now be either a string, a dict or a QuestionAnswerPairInputType, so we need to handle all cases.
        """
        if isinstance(output, QuestionAnswerPairInputType):
            return output

        # If output is just a string, treat it as llm_output
        if isinstance(output, str):
            return QuestionAnswerPairInputType(
                input_under_test=test.input_under_test,
                file_ids=file_ids,
                context=test.context,
                llm_output=output,
                metadata=MetadataType(
                    in_tokens=in_tokens_end - in_tokens_start,
                    out_tokens=out_tokens_end - out_tokens_start,
                    duration_seconds=time_end - time_start,
                ),
                test_id=test._id,
            )

        # If output is a dict, use provided values or defaults
        llm_output = output.get("llm_output", "")
        metadata = output.get("metadata", {})
        in_tokens = metadata.get("in_tokens", in_tokens_end - in_tokens_start)
        out_tokens = metadata.get("out_tokens", out_tokens_end - out_tokens_start)
        output_context = output.get("output_context", None)

        return QuestionAnswerPairInputType(
            input_under_test=test.input_under_test,
            file_ids=file_ids,
            context=test.context,
            llm_output=llm_output,
            output_context=output_context,
            metadata=MetadataType(
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                duration_seconds=time_end - time_start,
            ),
            test_id=test._id,
        )
