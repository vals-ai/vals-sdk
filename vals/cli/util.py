import click
import sys


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


def display_error_and_exit(message: str):
    click.secho(message, fg="red")
    sys.exit(1)
