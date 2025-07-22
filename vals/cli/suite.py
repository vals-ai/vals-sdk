import asyncio
import json
import os
from io import TextIOWrapper
from typing import Any

import click
from tabulate import tabulate

from vals.cli.util import display_error_and_exit
from vals.sdk.exceptions import ValsException
from vals.sdk.suite import Suite
from vals.sdk.types import RunParameters, RunStatus


@click.group(name="suite")
def suite_group():
    """
    Start, create, or view tests and test suites
    """
    pass


async def create_commmand_async(file: TextIOWrapper, project_id: str | None):
    suite = await Suite.from_dict(json.loads(file.read()))
    if project_id:
        suite.project_id = project_id
    await suite.create()

    click.secho("Successfully created test suite.", fg="green")
    click.secho(f"ID: {suite.id}")
    click.secho(suite.url, bold=True)


@click.command(name="create")
@click.option("-f", "--file", type=click.File("r"), required=True)
@click.option("--project-id", type=str, help="Project ID to create the suite in")
def create_command(file: TextIOWrapper, project_id: str | None):
    """
    Creates a new test suite based on the json file provided.

    See the documentation for information on the format.
    """
    asyncio.run(create_commmand_async(file, project_id))


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
@click.option("-f", "--file", type=click.File("r"), required=True)
@click.option("-s", "--suite-id", type=str, required=True)
def update_command(file: TextIOWrapper, suite_id: str):
    """
    Update the test and checks of an already existing suite
    """
    asyncio.run(update_command_async(file, suite_id))


async def list_command_async(limit: int, offset: int, search: str, project_id: str):
    if project_id:
        click.echo(f"Listing suites for project: {project_id}")
    else:
        click.echo("Listing suites for default project")

    suites = await Suite.list_suites(
        limit=limit, offset=offset - 1, search=search, project_id=project_id
    )
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
@click.option(
    "--project-id",
    type=str,
    default="default-project",
    show_default=True,
    help="Project ID to filter suites by. If unset, uses the default project.",
)
def list_command(
    limit: int,
    offset: int,
    search: str,
    project_id: str,
):
    """
    List test suites associated with this organization
    """
    asyncio.run(list_command_async(limit, offset, search, project_id))


async def pull_command_async(
    file: str,
    suite_id: str,
    to_csv: bool,
    to_json: bool,
    download_files: bool,
    suite_download_path: str | None,
    max_concurrent_downloads: int = 50,
):
    if to_csv and to_json:
        display_error_and_exit(
            "Cannot specify both --csv and --json - they are mutually exclusive."
        )

    path_output = suite_download_path if suite_download_path else os.getcwd()
    path_output_suite = os.path.join(path_output, os.path.basename(file))
    path_documents = os.path.join(path_output, "documents")

    suite = await Suite.from_id(
        suite_id,
        download_files=download_files,
        download_path=path_documents,
        max_concurrent_downloads=max_concurrent_downloads,
    )

    if to_csv:
        with open(path_output_suite, "w") as f:
            f.write(suite.to_csv_string())
    else:
        if download_files:
            for test in suite.tests:
                test.files_under_test = [
                    file.path for file in test.files_under_test if file.path is not None
                ]
        suite_dict = suite.to_dict()
        json_string = json.dumps(suite_dict, indent=2)
        with open(path_output_suite, "w") as f:
            f.write(json_string)

    click.secho("Successfully pulled test suite.", fg="green")


@click.command(name="pull")
@click.option("-s", "--suite-id", type=str, required=True)
@click.option(
    "--file", type=str, required=True, help="Name of the file to save the suite to"
)
@click.option("--csv", is_flag=True, help="Output in CSV format")
@click.option("--json", is_flag=True, help="Output in JSON format")
@click.option(
    "--no-download-files", is_flag=True, help="Do not download files from the suite"
)
@click.option(
    "--download-path",
    type=str,
    default=None,
    help="Path to write the suite file and associated files to",
)
@click.option(
    "--max-concurrent-downloads",
    type=int,
    default=50,
    help="Maximum number of concurrent files to download.",
)
def pull_command(
    suite_id: str,
    file: str,
    csv: bool,
    json: bool,
    no_download_files: bool,
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
            not no_download_files,
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
            click.secho("Run has finished successfully", fg="green")
        elif run.status == RunStatus.ERROR:
            click.secho("Run has completed with an error", fg="red")
    else:
        click.secho("Run has been successfully started", fg="green")

    click.secho(f"Run ID: {run.id}")
    click.secho(run.url, bold=True)


@click.command(name="run")
@click.option("-s", "--suite-id", type=str, required=True)
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
@click.option("--as-batch", is_flag=True, default=False, help="Run suite as a batch")
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
