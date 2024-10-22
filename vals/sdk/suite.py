import json
import os
from typing import Any, Dict, List

import requests
from gql import gql
from jsonschema import ValidationError, validate
from vals.sdk.auth import _get_auth_token
from vals.sdk.exceptions import ValsException
from vals.sdk.util import SUITE_SCHEMA_PATH, be_host, get_client

OPERATORS = [
    "includes",
    "logical_includes",
    "excludes",
    "equals",
    "mostly_equals",
    "includes_exactly",
    "includes_nearly_exactly",
    "excludes_exactly",
    "equals_exactly",
    "less_than_length",
    "answers",
    "not_answers",
    "is_concise",
    "is_coherent",
    "is_safe",
    "is_not_hallucinating",
    "is_not_generated",
    "no_legal_advice",
    "is_not_templated",
    "satisfies_statement",
    "consistent_phrasing",
    "valid_json",
    "valid_yaml",
    "regex",
    "matches_json_schema",
    "includes_any",
    "includes_any_v2",
    "list_format",
    "paragraph_format",
    "affirmative_answer",
    "negative_answer",
    "matches_tone",
    "check_salesiness",
    "matches_format",
    "grammar",
    "equal_intent",
    "capitalized_correctly",
    "is_polite",
    "progresses_conversation",
    "legacy_equals",
    "consistent_with_json",
    "consistent_with_context",
    "consistent_with_docs",
    "includes_only",
]

UNARY_OPERATORS = [
    "answers",
    "not_answers",
    "is_concise",
    "is_coherent",
    "is_safe",
    "is_not_hallucinating",
    "is_not_generated",
    "no_legal_advice",
    "is_not_templated",
    "valid_json",
    "valid_yaml",
    "list_format",
    "paragraph_format",
    "affirmative_answer",
    "negative_answer",
    "grammar",
    "capitalized_correctly",
    "is_polite",
    "progresses_conversation",
    "consistent_with_json",
    "consistent_with_context",
    "consistent_with_docs",
    "safety_consistency",
]


def create_suite(suite_data: Dict[str, Any]) -> str:
    """
    Method to create a test suite. Suite_data
    should be a JSON-like python dict that contains the information outlined in our docs.

    Returns the ID of the newly created suite.
    """
    _validate_suite(suite_data)
    suite_id = _create_test_suite(suite_data)

    files = _upload_files(suite_id, suite_data)

    _add_global_checks(suite_data, suite_id)
    _add_tests(suite_data, files, suite_id, create_only=True)

    return suite_id


def update_suite(suite_id: str, suite_data: Dict[str, Any]) -> None:
    """
    Method to update a test suite. data is in the same format as create_suite.
    """
    _validate_suite(suite_data)

    files = _upload_files(suite_id, suite_data)

    _add_tests(suite_data, files, suite_id)
    _add_global_checks(suite_data, suite_id)


def list_test_suites() -> List[Dict[str, Any]]:
    """

    Method to produce a list of test suites for a given org.

    """
    query = gql(
        f"""
        query getTestSuites {{
            testSuites {{
            description
            id
            org
            title
            created
            creator
            }}
            }}
        """
    )
    response = get_client().execute(query)

    # TODO: Error check
    return response["testSuites"]


def pull_suite(suite_id: str, include_id=False):
    """
    Get the JSON representation of a given test suite.
    """
    output = {}
    query = gql(
        f"""
        query getTestSuiteData {{
            testSuites(testSuiteId: "{suite_id}") {{
                description
                id
                org
                title
                created
                globalChecks
            }}
        }}
    """
    )

    response = get_client().execute(query)

    if len(response["testSuites"]) == 0:
        raise Exception(f"Unable to find test suite with id: {suite_id}")

    suite = response["testSuites"][0]
    output["title"] = suite["title"]
    output["description"] = suite["description"]

    query = gql(
        f"""
        query getTestData {{
            tests(testSuiteId: "{suite_id}") {{
                checks
                testId
                crossVersionId
                fileIds
                inputUnderTest
                sampleOutput
                goldenOutput
                sampleOutputType
                fileUids
                tags
                context
            }}
        }}    
        """
    )
    response = get_client().execute(query)
    raw_tests = response["tests"]

    tests = []
    for raw_test in raw_tests:
        test = {}
        if include_id:
            test["id"] = raw_test["testId"]
            test["cross_version_id"] = raw_test["crossVersionId"]

        test["input_under_test"] = raw_test["inputUnderTest"]

        if raw_test["fileIds"] is not None:
            file_ids = json.loads(raw_test["fileIds"])
            if len(file_ids) != 0:
                test["file_ids"] = file_ids

        if raw_test["fileUids"] is not None:
            file_uids = json.loads(raw_test["fileUids"])
            if len(file_uids) != 0:
                test["file_uids"] = file_uids

        if raw_test["sampleOutput"] != "":
            if raw_test["sampleOutputType"] == "file":
                test["file_fixed_output"] = raw_test["sampleOutput"]
            else:
                test["fixed_output"] = raw_test["sampleOutput"]

        if raw_test["goldenOutput"] != "":
            test["golden_output"] = raw_test["goldenOutput"]

        test["checks"] = json.loads(raw_test["checks"])
        context = json.loads(raw_test["context"])
        if len(context) != 0:
            test["context"] = context

        tags = json.loads(raw_test["tags"])

        if len(tags) != 0:
            test["tags"] = tags

        tests.append(test)

    output["tests"] = tests
    return output


def _validate_checks(checks):
    for check in checks:
        operator = check["operator"]
        if operator not in OPERATORS:
            raise ValsException(f"Unrecognized operator: {operator}")
        if operator not in UNARY_OPERATORS and "criteria" not in check:
            raise ValsException(
                f"'criteria' field must be specified for check with operator: '{operator}'"
            )


def _validate_suite(parsed_json):
    if not os.path.exists(SUITE_SCHEMA_PATH):
        raise ValsException(
            "Could not find schema file. The CLI tool is likely misconfigured."
        )

    # Use jsonschema to do most of our validation
    try:
        with open(SUITE_SCHEMA_PATH, "r") as schema_file:
            schema = json.load(schema_file)
            validate(instance=parsed_json, schema=schema)
    except ValidationError as e:
        raise ValsException(
            f"The file provided did not conform to the correct format. Validation Error: {e.message}. Look at the examples or the jsonschema to see the correct format."
        )

    if "global_checks" in parsed_json:
        _validate_checks(parsed_json["global_checks"])

    # We need to do some custom validation that JSON schema doesn't support
    tests = parsed_json["tests"]
    for test in tests:
        if "input_under_test" not in test and "file_under_test" not in test:
            raise ValsException(
                "For all tests, either 'input_under_test' or 'file_under_test' must be provided"
            )

        if "file_under_test" in test:
            fp = test["file_under_test"]
            if not os.path.exists(fp):
                raise ValsException(f"File does not exist: {fp}")
            if not os.path.isfile(fp):
                raise ValsException(f"Path is a directory: {fp}")

        if "file_uids" in test:
            if not isinstance(test["file_uids"], list):
                raise ValsException("file_uids must be a list")

        _validate_checks(test["checks"])


def _upload_file(suite_id: str, file_path: str) -> str:
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{be_host()}/upload_file/?test_suite_id={suite_id}",
            files={"file": f},
            headers={"Authorization": _get_auth_token()},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to upload file {file_path}")
        return response.json()["file_id"]


def _upload_files(suite_id: str, data: Dict[str, Any]):
    # Map from file path to file id
    files = {}
    for test in data["tests"]:
        if "file_under_test" in test:
            file_path = test["file_under_test"]
            if file_path not in files:
                files[file_path] = _upload_file(suite_id, file_path)

        if "files_under_test" in test:
            # We can either have a single file path, or a list of file paths
            file_path = test["files_under_test"]
            if type(file_path) == str:
                file_paths = [file_path]
            elif type(file_path) == list:
                file_paths = file_path
            for file_path in file_paths:
                files[file_path] = _upload_file(suite_id, file_path)

        if "file_fixed_output" in test:
            file_path = test["file_fixed_output"]
            if file_path not in files:
                files[file_path] = _upload_file(suite_id, file_path)

    return files


def _create_test_suite(data: Dict[str, Any]) -> str:
    query = gql(
        f"""
    mutation createTestSuite {{
        updateTestSuite(
            description: {json.dumps(data['description'])},
            testSuiteId: "0",
            title: "{data['title']}"
        ) {{
            testSuite {{
            description
            id
            org
            title
            }}
        }}
    }}
    """
    )
    result = get_client().execute(query)
    suite_id = result["updateTestSuite"]["testSuite"]["id"]
    return suite_id


def _add_global_checks(data, suite_id):
    if "global_checks" not in data:
        return

    query = gql(
        f"""
    mutation mutateGlobalChecks {{
        updateGlobalChecks (
            testSuiteId: "{suite_id}",
            checks: {json.dumps(json.dumps(data["global_checks"]))}
        ) {{
            success
        }}
    }}
    """
    )
    result = get_client().execute(query)


def _add_batch_to_suite(batch: List[str], create_only: bool):
    """
    Helper method to add a list of tests to a given suite
    """
    #
    query_str = f"""
        mutation addBatchTests {{
            batchUpdateTest(
                tests: [
                    {",".join(batch)}
                ],
                createOnly: {str(create_only).lower()}
            ) {{
                tests {{
                    testId
                }}
            }}
        }}
        """
    query = gql(query_str)
    response = get_client().execute(query)
    tests = response["batchUpdateTest"]["tests"]
    return [t["testId"] for t in tests]


def _add_tests(data, files, suite_id, create_only=False):
    test_ids = []
    batch = []
    i = 0

    for test in data["tests"]:
        # TODO: Escape chars better

        input_under_test = test["input_under_test"]

        # TODO: avoid double json
        checks = json.dumps(json.dumps(test["checks"]))
        # TODO: Do this server side

        if "fixed_output" in test:
            fixed_output = test["fixed_output"]
            fixed_output_type = "raw"
        elif "file_fixed_output" in test:
            fixed_output = files[test["file_fixed_output"]]
            fixed_output_type = "file"
        else:
            fixed_output = ""
            fixed_output_type = "raw"

        file_ids = []

        if "files_under_test" in test:
            file_path = test["files_under_test"]
            if type(file_path) == str:
                file_paths = [file_path]
            elif type(file_path) == list:
                file_paths = file_path

            file_ids = [files[file_path] for file_path in file_paths]
        # Backwards compatability for old suites
        elif "file_under_test" in test:
            file_paths = [test["file_under_test"]]

            file_ids = [files[file_path] for file_path in file_paths]
        elif "file_ids" in test:
            file_ids = test["file_ids"]

        if "file_uids" in test:
            file_uids = test["file_uids"]

        else:
            file_uids = []

        if "context" in test:
            context = json.dumps(json.dumps(test["context"]))
        else:
            context = json.dumps(json.dumps({}))

        if "golden_output" in test:
            golden_output = test["golden_output"]
        else:
            golden_output = ""

        tags = []
        if "tags" in test:
            tags = test["tags"]

        test_id = 0 if "id" not in test else test["id"]

        # We collate the tests we want to add in batches
        # When we have 100 tests in a batch, we add it to the suite.
        batch.append(
            f"""
            {{
                  sampleOutput: {json.dumps(fixed_output)},
                  sampleOutputType: "{fixed_output_type}",
                  checks: {checks}, 
                  inputUnderTest: {json.dumps(input_under_test)}, 
                  testSuiteId: "{suite_id}",
                  fileIds: {json.dumps(file_ids)}
                  fileUids: {json.dumps(file_uids)}
                  context: {context}
                  goldenOutput: {json.dumps(golden_output)}
                  tags: {json.dumps(tags)}
                  testId: "{test_id}"
            }}
            """
        )
        i += 1
        if i % 100 == 0:
            new_ids = _add_batch_to_suite(batch, create_only=create_only)
            test_ids.extend(new_ids)
            batch = []

    if len(batch) != 0:
        new_ids = _add_batch_to_suite(batch, create_only=create_only)
        test_ids.extend(new_ids)

    test_id_list = ", ".join([f'"{test_id}"' for test_id in test_ids])
    query = gql(
        f"""
            mutation removeOldTests {{
              removeUnusedTests(
                  testSuiteId: "{suite_id}",
                  inUseTests: [{test_id_list}]
                ) {{
                    success
                }}
            }}
            """
    )
    response = get_client().execute(query)

    # TODO: Check for errors
