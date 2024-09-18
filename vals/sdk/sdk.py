from functools import wraps
from io import BytesIO
from time import time
from typing import Callable

import click
import pypandoc
import requests
from openai import OpenAI
from pypdf import PdfReader
from tqdm import tqdm
from vals.cli.suite import pull_suite, update_suite
from vals.sdk.auth import _get_auth_token
from vals.sdk.run import get_run_url, start_run
from vals.sdk.util import be_host

in_tokens = 0
out_tokens = 0


def _parse_test_suite_id_from_url(test_suite_url: str) -> str:
    start_index = test_suite_url.find("test_suite_id=") + len("test_suite_id=")
    return test_suite_url[start_index:]


def _read_file(file_id: str):
    response = requests.post(
        url=f"{be_host()}/download_file/?file_id={file_id}",
        headers={"Authorization": _get_auth_token()},
    )
    return BytesIO(response.content)


def _parse_file_id(file_id: str):
    org, rest = file_id.split("/")
    filename, test_suite_id = rest.split(";")

    return org, filename, test_suite_id


def read_pdf(file: BytesIO):
    """
    Convenience method to parse PDFs to strings
    """
    text = ""
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text


def read_docx(file: BytesIO):
    """
    Convenience method to parse docx files to strings
    """
    output = pypandoc.convert_text(file.read(), to="plain", format="docx")
    return output


def read_text(file: BytesIO):
    return file.read().decode("utf-8")


def run_evaluations(
    test_suite_url: str,
    generate_fn: Callable[[str], str],
    description="Ran automatically using the PRL SDK",
    maximum_threads=4,
    verbosity=1,
    model_under_test="sdk",
    **kwargs,
):
    """
    This function is meant to be used to test arbitrary models and pipelines with the Vals platform.

    It takes in a `generate_fn` and the url of a test suite.

    For every test input in the test suite, it queries generate_fn(test_input) to produce the
    output from the LLM model or application. It then updates the test suite with the results,
    and kicks off a new run.

    It returns the run_id of the run.

    See examples/sdk_example.py for example usage.
    """
    test_suite_id = _parse_test_suite_id_from_url(test_suite_url)

    suite_data = pull_suite(test_suite_id, include_id=True)

    if verbosity == 0:
        iterator = suite_data["tests"]
    else:
        iterator = tqdm(suite_data["tests"])

    global in_tokens, out_tokens
    metadata = {}

    uses_files = False
    uses_file_uids = False
    uses_context = False

    for test in suite_data["tests"]:
        if (
            "file_ids" in test
            and test["file_ids"] is not None
            and len(test["file_ids"])
        ):
            uses_files = True

        if (
            "file_uids" in test
            and test["file_uids"] is not None
            and len(test["file_uids"]) > 0
        ):
            uses_file_uids = True

        if (
            "context" in test
            and test["context"] is not None
            and len(test["context"]) > 0
        ):
            uses_context = True

    # Kwargs that we shouldn't register as extra parameters
    non_param_kwargs = {}

    for test in iterator:
        start = time()

        if uses_files:
            if "file_ids" in test and test["file_ids"] is not None:
                file_ids = test["file_ids"]

                files = {}
                for file_id in file_ids:
                    _, file_name, _ = _parse_file_id(file_id)
                    files[file_name] = _read_file(file_id)
            else:
                files = {}
            non_param_kwargs["files"] = files

        if uses_file_uids:
            if "file_uids" in test and test["file_uids"] is not None:
                file_uids = test["file_uids"]
                non_param_kwargs["file_uids"] = file_uids
            else:
                non_param_kwargs["file_uids"] = []

        if uses_context:
            if "context" in test and test["context"] is not None:
                non_param_kwargs["context"] = test["context"]
            else:
                non_param_kwargs["context"] = {}

        non_param_kwargs.update(kwargs)
        fixed_output = generate_fn(test["input_under_test"], **non_param_kwargs)
        test["fixed_output"] = fixed_output

        end = time()
        metadata[test["id"]] = {
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "duration_seconds": end - start,
        }
        in_tokens = 0
        out_tokens = 0

    update_suite(test_suite_id, suite_data)
    run_id = start_run(
        test_suite_id,
        {
            "use_fixed_output": True,
            "description": description,
            "maximum_threads": maximum_threads,
            "model_under_test": model_under_test,
            **kwargs,
        },
        metadata_map=metadata,
    )
    run_url = get_run_url(run_id)

    if verbosity >= 1:
        click.secho(
            "Successfully updated test suite with new fixed outputs and started a new run.",
            fg="green",
        )
        click.secho(run_url, bold=True)

    return run_id


def _wrap_chatcompletion(func: Callable):
    @wraps(func)
    def wrapper(**kwargs):
        response = func(**kwargs)
        global in_tokens, out_tokens

        in_tokens += response.usage.prompt_tokens
        out_tokens += response.usage.completion_tokens

        return response

    return wrapper


# External Facing
def patch(client: OpenAI):
    """
    Calling this function allows the Vals SDK to collect token metadata from any calls to OpenAI
    or models that use the OpenAI API.
    """
    client.chat.completions.create = _wrap_chatcompletion(
        client.chat.completions.create
    )
    return client
