import json
import sys
from io import TextIOWrapper
from typing import Any, Dict

import click
from vals.cli.util import display_error_and_exit, prompt_user_for_suite
from vals.sdk.exceptions import PrlException
from vals.sdk.suite import create_suite, list_test_suites, pull_suite, update_suite
from vals.sdk.util import fe_host


@click.group(name="suite")
def suite_group():
    """
    Start, create, or view tests and test suites
    """
    pass


def parse_suite_interactive():
    title = click.prompt("Test Suite Title")
    while title == "":
        title = click.prompt("Title cannot be empty. Reenter")

    description = click.prompt("Test Suite Description")

    i = 1
    keep_generating_prompts = True
    tests = []
    while keep_generating_prompts:
        click.secho(f"---Test {i}---", bold=True)
        input_under_test = click.prompt("Input under test (e.g. the prompt)")

        keep_generating_criteria = True
        j = 1
        checks = []
        while keep_generating_criteria:
            operator = click.prompt(f"Operator {j}")
            criteria = click.prompt(f"Criteria {j}")
            checks.append({"criteria": criteria, "operator": operator})
            j += 1

            keep_generating_criteria = click.confirm("Keep Generating Checks?")

        i += 1

        tests.append({"input_under_test": input_under_test, "checks": checks})
        keep_generating_prompts = click.confirm("Keep generating tests?")

    return {"title": title, "description": description, "tests": tests}


def parse_suite_file(file):
    # TODO: Validate file format
    try:
        return json.load(file)
    except Exception as e:
        raise PrlException("The input file provided is not valid JSON")


def parse_suite(interactive: bool, file: TextIOWrapper) -> Dict[str, Any]:
    if not interactive and file is None:
        click.echo(
            "Either --interactive must be passed, or an input file should be specified"
        )
        sys.exit(1)

    if interactive:
        data = parse_suite_interactive()
    else:
        data = parse_suite_file(file)

    return data


@click.command(name="create")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive mode instead of reading from file",
)
@click.argument("file", type=click.File("r"), required=False)
def create_command(interactive: bool, file: str):
    """
    Creates a new test suite.

    There are two modes. In normal operation, inputs are read from a JSON file:

    \tprl suite create <filename>

    In interactive mode, the user is prompted for values:

    \tprl suite create --interactive

        Requires authentication to use.
    """
    # try:
    data = parse_suite(interactive, file)

    try:
        suite_id = create_suite(data)
    except PrlException as e:
        display_error_and_exit(f"Could not create suite. {e.message}")

    # Execute the query on the transport
    click.secho("Successfully created test suite.", fg="green")
    click.secho(f"{fe_host()}/view?test_suite_id={suite_id}", bold=True)


@click.command(name="update")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive mode instead of reading from file",
)
@click.argument("file", type=click.File("r"), required=False)
def update_command(interactive: bool, file: str):
    """
    Update the test and checks of an already existing suite
    """
    suite_id = prompt_user_for_suite()
    try:
        data = parse_suite(interactive, file)
        update_suite(suite_id, data)

        click.secho("Successfully updated test suite.", fg="green")
        click.secho(f"{fe_host()}/view?test_suite_id={suite_id}", bold=True)
    except PrlException as e:
        click.secho(e.message, fg="red")
    except Exception as e:
        click.secho("Suite Update Failed. Error:" + str(e), fg="red")


@click.command(name="list")
def list_command():
    """
    List test suites associated with this organization
    """
    suites = list_test_suites()

    suite_text = "\n".join([f"{i}: {s['title']}" for i, s in enumerate(suites)])
    click.echo(suite_text)


@click.command(name="pull")
@click.argument("file", type=click.File("w"), required=True)
def pull_command(file: TextIOWrapper):
    """
    Read a suite from the PRL server to a local JSON file.
    """
    suite_id = prompt_user_for_suite()
    try:
        output = pull_suite(suite_id)
        file.write(json.dumps(output, indent=2))

    except PrlException as e:
        display_error_and_exit(e.message)

    click.secho("Successfully pulled test suite.", fg="green")


suite_group.add_command(create_command)
suite_group.add_command(list_command)
suite_group.add_command(update_command)
suite_group.add_command(pull_command)
