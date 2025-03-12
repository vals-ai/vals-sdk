import asyncio
import json
from io import TextIOWrapper
from typing import Any

import click
from tabulate import tabulate
from vals.cli.util import display_error_and_exit
from vals.sdk.exceptions import ValsException
from vals.sdk.suite import Suite
from vals.sdk.types import RunParameters, RunStatus
from vals.sdk.util import download_files_bulk


@click.group(name="suite")
def suite_group():
    """
    Start, create, or view tests and test suites
    """
    pass


async def create_commmand_async(file: TextIOWrapper):

    suite = await Suite.from_dict(json.loads(file.read()))
    await suite.create()

    click.secho("Successfully created test suite.", fg="green")
    click.secho(f"ID: {suite.id}")
    click.secho(suite.url, bold=True)


@click.command(name="create")
@click.argument("file", type=click.File("r"))
def create_command(file: TextIOWrapper):
    """
    Creates a new test suite based on the json file provided.

    See the documentation for information on the format.
    """
    asyncio.run(create_commmand_async(file))


async def update_command_async(file: TextIOWrapper, suite_id: str):
    try:
        suite = await Suite.from_dict(json.loads(file.read()))
        suite.id = suite_id
        await suite.update()

        click.secho("Successfully updated test suite.", fg="green")
        click.secho(suite.url, bold=True)

    except ValsException as e:
        click.secho(e.message, fg="red")
    except Exception as e:
        click.secho("Suite Update Failed. Error:" + str(e), fg="red")


@click.command(name="update")
@click.argument("file", type=click.File("r"))
@click.argument("suite_id", type=str)
def update_command(file: TextIOWrapper, suite_id: str):
    """
    Update the test and checks of an already existing suite
    """
    asyncio.run(update_command_async(file, suite_id))


async def list_command_async(limit: int, offset: int, search: str):
    suites = await Suite.list_suites(limit=limit, offset=offset - 1, search=search)
    headers = ["#", "Title", "Suite ID", "Last Modified"]
    rows = []
    for i, suite in enumerate(suites, start=offset):
        truncated_title = suite.title
        date_str = suite.last_modified_at.strftime("%Y/%m/%d %H:%M")
        rows.append([i, truncated_title, suite.id, date_str])

    table = tabulate(rows, headers=headers, tablefmt="tsv")
    click.echo(table)


@click.command(name="list")
@click.option("-l", "--limit", type=int, default=25, help="Number of rows to return")
@click.option(
    "-o", "--offset", type=int, default=1, help="Start table at this row (1-indexed)"
)
@click.option("--search", type=str, default="", help="Search for a suite by title")
def list_command(
    limit: int,
    offset: int,
    search: str,
):
    """
    List test suites associated with this organization
    """
    asyncio.run(list_command_async(limit, offset, search))


async def pull_command_async(
    file: TextIOWrapper,
    suite_id: str,
    to_csv: bool,
    to_json: bool,
    download_files: bool,
    download_path: str | None,
    max_concurrent_downloads: int = 50,
):
    if to_csv and to_json:
        display_error_and_exit(
            "Cannot specify both --csv and --json - they are mutually exclusive."
        )

    suite = await Suite.from_id(suite_id)

    if download_files:
        file_ids = []
        for test in suite.tests:
            file_ids.extend(
                [file_id for file_id in test._file_ids if file_id is not None]
            )

        path = suite.title if download_path is None else download_path

        if len(file_ids) != 0:
            click.secho(
                f"This suite has files. Downloading them to directory '{path}'...",
                fg="green",
            )
        filename_to_filepath_map = await download_files_bulk(
            file_ids, path, max_concurrent_downloads=max_concurrent_downloads
        )

    if to_csv:
        file.write(suite.to_csv_string())
    else:
        json_string = suite.to_json_string()
        if download_files:
            parsed_json = json.loads(json_string)
            for i, test in enumerate(parsed_json["tests"]):
                file_paths = []
                for filename in test["files_under_test"]:
                    file_paths.append(filename_to_filepath_map[filename])
                parsed_json["tests"][i]["files_under_test"] = file_paths
            json_string = json.dumps(parsed_json, indent=2)
        file.write(json_string)

    click.secho("Successfully pulled test suite.", fg="green")


@click.command(name="pull")
@click.argument(
    "file",
    type=click.File("w"),
    required=True,
)
@click.argument("suite_id", type=str, required=True)
@click.option("--csv", is_flag=True, help="Output in CSV format")
@click.option("--json", is_flag=True, help="Output in JSON format")
@click.option("--download-files", is_flag=True, help="Download files from the suite")
@click.option(
    "--download-path", type=str, default=None, help="Path to download the files to"
)
@click.option(
    "--max-concurrent-downloads",
    type=int,
    default=50,
    help="Maximum number of concurrent downloads",
)
def pull_command(
    file: TextIOWrapper,
    suite_id: str,
    csv: bool,
    json: bool,
    download_files: bool,
    download_path: str,
    max_concurrent_downloads: int = 50,
):
    """
    Read a suite from the PRL server to a local JSON file.
    """
    asyncio.run(
        pull_command_async(
            file,
            suite_id,
            csv,
            json,
            download_files,
            download_path,
            max_concurrent_downloads,
        )
    )


async def run_command_async(
    suite_id: str,
    model: str,
    run_name: str | None,
    wait_for_completion: bool,
    params: dict[str, Any],
):
    # Remove None values from parameters dictionary
    params = {k: v for k, v in params.items() if v is not None}
    parameters = RunParameters(**params)
    suite = await Suite.from_id(suite_id)

    if wait_for_completion:
        click.echo("Starting run and waiting for it to finish...")

    run = await suite.run(
        model=model,
        run_name=run_name,
        wait_for_completion=wait_for_completion,
        parameters=parameters,
    )

    if wait_for_completion:
        if run.status == RunStatus.SUCCESS:
            click.secho(f"Run has finished successfully", fg="green")
        elif run.status == RunStatus.ERROR:
            click.secho(f"Run has completed with an error", fg="red")
    else:
        click.secho(f"Run has been successfully started", fg="green")

    click.secho(f"Run ID: {run.id}")
    click.secho(run.url, bold=True)


@click.command(name="run")
@click.argument("suite_id", type=str, required=True)
@click.option("--model", type=str, required=True, help="Model to run the tests with")
@click.option(
    "--run-name",
    type=str,
    default=None,
    help="Name of the run",
)
@click.option(
    "--wait-for-completion",
    is_flag=True,
    default=False,
    help="Wait for the run to complete before returning",
)
@click.option(
    "--eval-model", type=str, default=None, help="Model to use for evaluation"
)
@click.option(
    "--parallelism",
    type=int,
    default=None,
    help="Maximum number of concurrent threads",
)
@click.option(
    "--run-golden-eval",
    is_flag=True,
    default=None,
    help="Run evaluation against golden output",
)
@click.option(
    "--run-confidence-evaluation",
    is_flag=True,
    default=None,
    help="Run confidence evaluation",
)
@click.option(
    "--heavyweight-factor",
    type=int,
    default=None,
    help="Heavyweight factor for evaluation",
)
@click.option(
    "--create-text-summary",
    is_flag=True,
    default=None,
    help="Create text summary of results",
)
@click.option(
    "--temperature", type=float, default=None, help="Temperature parameter for model"
)
@click.option(
    "--max-output-tokens", type=int, default=None, help="Maximum tokens in model output"
)
@click.option("--system-prompt", type=str, default=None, help="System prompt for model")
@click.option(
    "--new-line-stop-option", is_flag=True, default=None, help="Stop on new line"
)
def run_command(
    suite_id: str, model: str, run_name: str, wait_for_completion: bool, **params
):
    """
    Run a test suite
    """
    asyncio.run(
        run_command_async(suite_id, model, run_name, wait_for_completion, params)
    )


suite_group.add_command(create_command)
suite_group.add_command(update_command)
suite_group.add_command(list_command)
suite_group.add_command(pull_command)
suite_group.add_command(run_command)
