import asyncio
import json
import sys
from io import TextIOWrapper
from typing import Any, Dict

import click
from vals.cli.util import display_error_and_exit, prompt_user_for_suite
from vals.sdk.exceptions import ValsException
from vals.sdk.suite import create_suite, list_test_suites, pull_suite, update_suite
from vals.sdk.util import fe_host
from vals.sdk.v2.suite import Suite


@click.group(name="suite")
def suite_group():
    """
    Start, create, or view tests and test suites
    """
    pass


async def create_commmand_async(file: TextIOWrapper):
    try:
        suite = await Suite.from_dict(json.loads(file.read()))
        await suite.create()
        click.secho("Successfully created test suite.", fg="green")
        click.secho(f"ID: {suite.id}")
        click.secho(suite.url, bold=True)

    except ValsException as e:
        display_error_and_exit(e.message)


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
@click.argument("suite_id", type=click.File("r"))
async def update_command(file: TextIOWrapper, suite_id: str):
    """
    Update the test and checks of an already existing suite
    """
    asyncio.run(update_command_async(file, suite_id))


async def list_command_async(number: int, offset: int):
    suites = await Suite.list_suites(limit=number, offset=offset)

    number_width = 3
    title_width = 40
    id_width = 36
    last_modified_width = 20

    header = f"| {'#':{number_width}} | {'Title':{title_width}} | {'Suite ID':{id_width}} | {'Last Modified':{last_modified_width}} |"
    click.echo(header)
    line = (
        "+"
        + "+".join(
            [
                "-" * (width + 2)
                for width in [number_width, title_width, id_width, last_modified_width]
            ]
        )
        + "+"
    )
    click.echo(line)

    for i, suite in enumerate(suites):
        truncated_title = (
            suite.title[: title_width - 3] + "..."
            if len(suite.title) > title_width
            else suite.title
        )
        date_str = suite.last_modified_at.strftime("%Y/%m/%d %H:%M")
        row = f"| {i + 1:{number_width}} | {truncated_title:{title_width}} | {suite.id:{id_width}} | {date_str:{last_modified_width}} |"
        click.echo(row)


@click.command(name="list")
@click.option("-n", "--number", type=int, default=25, help="Number of rows to return")
@click.option("-o", "--offset", type=int, default=0, help="Start table at this row")
def list_command(
    number: int,
    offset: int,
):
    """
    List test suites associated with this organization
    """
    asyncio.run(list_command_async(number, offset))


async def pull_command_async(
    file: TextIOWrapper, suite_id: str, to_csv: bool, to_json: bool
):
    if to_csv and to_json:
        display_error_and_exit(
            "Cannot specify both --csv and --json - they are mutually exclusive."
        )

    suite = await Suite.from_id(suite_id)
    if to_json:
        file.write(json.dumps(suite.to_dict(), indent=2))
    elif to_csv:
        file.write(suite.to_csv_string())

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
def pull_command(file: TextIOWrapper, suite_id: str, csv: bool, json: bool):
    """
    Read a suite from the PRL server to a local JSON file.
    """
    asyncio.run(pull_command_async(file, suite_id, csv, json))


suite_group.add_command(create_command)
suite_group.add_command(update_command)
suite_group.add_command(list_command)
suite_group.add_command(pull_command)
