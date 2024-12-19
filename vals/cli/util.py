import sys

import click
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


def display_table(
    column_headers: list[str], column_widths: list[int], rows: list[list[str]]
):
    header = (
        "|"
        + "|".join([f" {h:{w}} " for h, w in zip(column_headers, column_widths)])
        + "|"
    )
    click.echo(header)
    seperator_line = (
        "+" + "+".join(["-" * (width + 2) for width in column_widths]) + "+"
    )
    click.echo(seperator_line)
    for row in rows:
        row_str = (
            "| "
            + " | ".join([f"{str(r):{w}}" for r, w in zip(row, column_widths)])
            + " |"
        )
        click.echo(row_str)
