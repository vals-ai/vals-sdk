import json
import time
from typing import Any, Dict, List

import requests
from gql import gql
from jsonschema import ValidationError, validate
from vals.sdk.auth import _get_auth_token
from vals.sdk.exceptions import ValsException
from vals.sdk.util import RUN_SCHEMA_PATH, be_host, fe_host, get_client

# Rudimentary cache for pulling params
_default_params = None


def _get_default_parameters() -> Dict[str, Any]:
    global _default_params
    if _default_params is not None:
        return _default_params

    response = requests.get(
        url=f"{be_host()}/run_parameter_info/",
        headers={"Authorization": _get_auth_token()},
    )
    if response.status_code != 200:
        raise ValsException(response.text)

    param_info = response.json()

    _default_params = {}
    for k, v in param_info.items():
        default_val = v["default"]
        if v["type"] == "select":
            default_val = default_val["value"]
        _default_params[k] = default_val

    return _default_params


def start_run(suite_id: str, parameters: Dict[Any, Any] = {}, metadata_map=None) -> str:
    """
    Method to start a run.

    suite_id: The ID of the suite to run.
    parameters: Any run parameters. Examples include 'use_golden_output', 'use_fixed_output',
    'threads', etc.

    Returns the run id.
    """
    try:
        with open(RUN_SCHEMA_PATH, "r") as f:
            schema = json.load(f)
            default_params = _get_default_parameters()
            parameters = {**default_params, **parameters}

            # Description isn't included by default
            if "description" not in parameters:
                parameters["description"] = "Ran with PRL SDK."

            # Validation
            validate(instance=parameters, schema=schema)

    except ValidationError as e:
        raise ValsException(
            f"Config file provided did not conform to JSON schema. Message: {e.message}"
        )

    body = {"test_suite_id": suite_id, "parameters": parameters}
    if metadata_map is not None:
        body["metadata"] = metadata_map

    response = requests.post(
        url=f"{be_host()}/start_run/",
        headers={"Authorization": _get_auth_token()},
        json=body,
    )

    if response.status_code == 200:
        run_id = response.json()["run_id"]
        return run_id
    else:
        raise ValsException(
            f"Could not start run. Received error from server: {response.text}"
        )


def get_csv(run_id: str) -> bytes:
    """
    A method to pull a CSV from a run result.

    Returns: The CSV, as bytes.
    """
    response = requests.post(
        url=f"{be_host()}/export_results_to_file/?run_id={run_id}",
        headers={"Authorization": _get_auth_token()},
    )

    if response.status_code != 200:
        raise ValsException("Received Error from Vals Servers: " + response.text)

    return response.content


def get_run_results(
    include_archived: bool = False, suite_id: str = None
) -> List[Dict[str, Any]]:
    query = gql(
        f"""
        query MyQuery {{
          runs(archived: {"null" if include_archived else "false"}, suiteId: {"null" if suite_id is None else '"' + suite_id + '"'}) {{
            runId
            passPercentage
            status
            runId
            textSummary
            timestamp
            archived
            parameters
            testSuite {{
              title
            }}
          }}
        }}
    """
    )
    results = get_client().execute(query)["runs"]

    for result in results:
        result["parameters"] = json.loads(result["parameters"])
    # TODO: Error Handling

    return results


def run_status(run_id: str) -> str:
    """
    For a given run, returns the run status, either 'in_progress', 'success', or 'error'.
    """
    query = gql(
        f"""
        query MyQuery {{
            run(runId: "{run_id.strip()}") {{
                status
            }}
        }}
        """
    )
    results = get_client().execute(query)
    status = results["run"]["status"]
    if status == "":
        status = "in_progress"

    return status


def run_summary(run_id: str) -> Dict[str, Any]:
    """
    Produces the top-level analytics and summary data for a given run (text summary, pass percentage, etc.)
    """
    query = gql(
        f"""          
        query MyQuery {{
            run(runId: "{run_id.strip()}") {{
                status
                archived
                parameters
                textSummary
                timestamp
                completedAt
                archived
                passPercentage
                passPercentageWithOptional
                humanEvalAccuracy
                humanEvalF1
                humanEvalPhi
                humanEvalMean
                humanEvalCoverage
                checkResultSummary
            }}
        }}
        """
    )
    results = get_client().execute(query)

    # TODO: Make a non-json, pretty version if useful
    dict_result = results["run"]
    dict_result["parameters"] = json.loads(dict_result["parameters"])
    dict_result["checkResultSummary"] = json.loads(dict_result["checkResultSummary"])

    return dict_result


def _pull_test_results(run_id: str) -> Dict[str, Any]:
    query = gql(
        f"""
        query MyQuery2 {{
                testResults(runId: "{run_id}") {{
                  id
                  llmOutput
                  passPercentage
                  passPercentageWithOptional
                  resultJson
                  humanEval
                  humanFeedback
                  test {{
                    testId
                    inputUnderTest
                  }}
                  metadata
                }}
              }}
    """
    )
    results = get_client().execute(query)["testResults"]
    for result in results:
        result["resultJson"] = json.loads(result["resultJson"])

    return results


def pull_run_results_json(run_id: str) -> Dict[str, Any]:
    """
    Pull all data about a run as JSON, including both the summary
    and the test results
    """
    summary = run_summary(run_id)
    test_results = _pull_test_results(run_id)
    return {**summary, "results": test_results}


def wait_for_run_completion(run_id: str) -> str:
    """
    Block a process until a given run has finished running.

    """
    time.sleep(5)
    status = "in_progress"
    while status == "in_progress":
        status = run_status(run_id)
        time.sleep(1)

    return status


def get_run_url(run_id: str) -> str:
    """
    Utility function to transform a run id to a viewable Vals AI URL.
    """
    return f"{fe_host()}/results?run_id={run_id}"
