import asyncio
import json
from datetime import datetime
from io import TextIOWrapper

import click
from vals.cli.util import display_table
from vals.sdk.v2.run import Run


@click.group(name="run")
def run_group():
    """
    Commands relating to starting or viewing runs
    """
    pass


async def pull_async(run_id: str, file: TextIOWrapper, csv: bool, _json: bool):
    run = await Run.from_id(run_id)

    if csv:
        file.write(await run.to_csv_string())
    elif _json:
        file.write(json.dumps(run.to_dict()))

    click.secho("Successfully pulled run results.", fg="green")


@click.command
@click.argument("run-id", type=click.STRING, required=True)
@click.argument("file", type=click.File("w"), required=True)
@click.option("--csv", is_flag=True, default=False, help="Save as a CSV")
@click.option("--json", is_flag=True, default=False, help="Save as a JSON")
def pull(run_id: str, file: TextIOWrapper, csv: bool, json: bool):
    """
    Pull results of a run and save it to a file.
    """
    asyncio.run(pull_async(run_id, file, csv, json))


async def list_async(
    limit: int, offset: int, suite_id: str | None, show_archived: bool
):
    run_results = await Run.list_runs(
        limit=limit, offset=offset, show_archived=show_archived
    )

    # f"{'Title':40} {'Timestamp':24} {'Status':13} {'Pass %':8} {'Archived':9}",
    column_names = ["Run Name", "Test Suite", "Status", "Pass %", "Timestamp"]
    test_suite_width = 40
    run_name_width = 40
    column_widths = [test_suite_width, run_name_width, 13, 8, 20]

    rows = []
    for run in run_results:
        date_str = run.timestamp.strftime("%Y/%m/%d %H:%M")
        # TODO: Change this to pass rate
        pass_percentage_str = (
            f"{run.pass_percentage * 100:.2f}" if run.pass_percentage else "N/A"
        )
        if len(run.test_suite_title) > test_suite_width:
            run.test_suite_title = run.test_suite_title[: test_suite_width - 3] + "..."
        if len(run.name) > run_name_width:
            run.name = run.name[: run_name_width - 3] + "..."
        rows.append(
            [
                run.test_suite_title[:38],
                run.name,
                run.status.value,
                pass_percentage_str,
                date_str,
            ]
        )

    display_table(column_names, column_widths, rows)


@click.command()
@click.option(
    "-l",
    "--limit",
    type=click.INT,
    default=25,
    help="Limit the number of runs to display",
)
@click.option("-o", "--offset", required=False, help="Filter runs by suite id")
@click.option("--suite-id", required=False, help="Filter runs by suite id")
@click.option(
    "--show-archived",
    is_flag=True,
    help="When enabled, archived runs are displayed in the output",
)
def list(limit: int, offset: int, suite_id: str | None, show_archived: bool):
    """
    List runs associated with this organization
    """
    asyncio.run(list_async(limit, offset, suite_id, show_archived))


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


run_group.add_command(pull)
run_group.add_command(list_results_click_command)
run_group.add_command(list)
