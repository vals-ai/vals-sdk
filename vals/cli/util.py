import sys

import click
from vals.sdk.suite import list_test_suites
from vals.sdk.util import list_rag_suites


def prompt_user_for_rag_suite():
    suites = list_rag_suites()
    click.echo("Rag Suites:")
    click.echo(
        "\n".join([f"{i}: {s['id']} {s['query']}" for i, s in enumerate(suites)])
    )

    idx = click.prompt("Enter the number of the rag suite to run", type=int)
    while not 0 <= idx <= len(suites):
        idx = click.prompt("Invalid choice. Retry", type=int)
    suite_id = suites[idx]["id"]
    return suite_id


def display_error_and_exit(error_msg: str):
    click.secho("ERROR: " + error_msg, fg="red")
    sys.exit(1)


def prompt_user_for_suite():
    suites = list_test_suites()
    click.echo("Test Suites:")
    click.echo("\n".join([f"{i}: {s['title']}" for i, s in enumerate(suites)]))

    idx = click.prompt("Enter the number of the test suite to run", type=int)
    while not 0 <= idx <= len(suites):
        idx = click.prompt("Invalid choice. Retry", type=int)
    suite_id = suites[idx]["id"]
    return suite_id
