import click
import sys


def display_error_and_exit(message: str):
    click.secho(message, fg="red")
    sys.exit(1)
