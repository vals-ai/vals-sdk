import sys

import click

from .rag import rag_group
from .run import run_group
from .suite import suite_group
from .util import display_error_and_exit


class ExceptionHandlingWrapper(click.Group):
    """Custom class that overrides the default error handling for click"""

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            display_error_and_exit(e)


@click.group(cls=ExceptionHandlingWrapper)
def cli():
    pass


cli.add_command(suite_group)
cli.add_command(run_group)
cli.add_command(rag_group)
