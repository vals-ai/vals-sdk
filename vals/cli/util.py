import sys

import click


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


def display_error_and_exit(error_msg: str):
    click.secho("ERROR: " + error_msg, fg="red")
    sys.exit(1)
