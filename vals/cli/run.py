import json
from datetime import datetime
from io import BytesIO

import click
from vals.sdk.run import (
    get_csv,
    get_run_results,
    get_run_url,
    run_status,
    run_summary,
    start_run,
)
from vals.sdk.util import fe_host

from ..sdk.exceptions import PrlException
from .util import display_error_and_exit, prompt_user_for_suite


@click.group(name="run")
def run_group():
    """
    Commands relating to starting or viewing runs
    """
    pass


@click.command(name="start")
@click.argument("param-file", type=click.File("r"), required=False, default=None)
@click.option("-s", "--suite-id", required=False)
def start_click_command(param_file, suite_id: str = None):
    """
    Start a new run of a test suite.
    """
    if param_file is not None:
        try:
            parameters = json.load(param_file)
        except Exception:
            display_error_and_exit("Config file was not valid JSON")
    else:
        parameters = {}

    if suite_id is None:
        suite_id = prompt_user_for_suite()

    try:
        run_id = start_run(suite_id, parameters)
    except PrlException as e:
        display_error_and_exit(e.message)

    click.secho("Successfully started run.", fg="green")
    click.secho(get_run_url(run_id), bold=True)


@click.command(name="get-csv")
@click.argument("run-id", type=click.STRING, required=True)
@click.argument("file", type=click.File("wb"), required=True)
def get_csv_click_command(run_id, file: BytesIO):
    """
    Get the CSV file with run results for a given run.

    This is equivalent to clicking 'Export to CSV' on the Run Results
    page within the website.

    Pass the path to a file where the CSV file should be downloaded.
    """

    try:
        csv_bytes = get_csv(run_id)
        file.write(csv_bytes)

    except PrlException as e:
        display_error_and_exit(e.message)

    click.secho("Successfully downloaded the result CSV.", fg="green")


@click.command(name="list-results")
@click.option(
    "--show-archived",
    is_flag=True,
    show_default=True,
    default=False,
    help="When enabled, archived runs are displayed in the output",
)
@click.option(
    "--ids-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="When enabled, displays a list of only run ids with no additional information",
)
@click.option("--suite-id", required=False, help="Display")
def list_results_click_command(show_archived: bool, ids_only: bool, suite_id: str):
    """
    Display a list of run results.
    """
    run_results = get_run_results(show_archived, suite_id)
    if ids_only:
        for result in run_results:
            print(result["runId"])

    else:
        click.secho(
            f"{'Title':40} {'Timestamp':24} {'Status':13} {'Pass %':8} {'Archived':9}",
            bold=True,
        )
        for result in run_results:
            localtz = datetime.now().astimezone().tzinfo
            timestamp = datetime.strptime(
                result["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
            ).astimezone(localtz)
            # Print the timestamp in our own arbitrary format
            timestamp_str = datetime.strftime(timestamp, "%Y-%m-%d %H:%M %Z")
            pass_percentage = (
                "N/A"
                if result["passPercentage"] is None
                else f"{result['passPercentage'] * 100:.2f}"
            )
            click.secho(
                f"{result['testSuite']['title'][:38]:40} {timestamp_str:24} {result['status']:13} {pass_percentage:8} {'yes' if result['archived'] else 'no':10}",
            )


@click.command(name="status")
@click.argument("run-id", type=click.STRING, required=True)
def status_click_command(run_id):
    "CLI command to get the current status of a single run (error, in_progress, success)"
    status = run_status(run_id)
    click.echo(status)


@click.command(name="summary")
@click.argument("run-id", type=click.STRING, required=True)
def summary_click_command(run_id: str):
    """CLI command to get the current top-line result of a run"""
    dict_result = run_summary(run_id)
    click.echo(json.dumps(dict_result))


run_group.add_command(start_click_command)
run_group.add_command(list_results_click_command)
run_group.add_command(get_csv_click_command)
run_group.add_command(status_click_command)
run_group.add_command(summary_click_command)
