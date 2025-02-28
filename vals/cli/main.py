import click
import sys

from .rag import rag_group
from .run import run_group
from .suite import suite_group
from .util import pretty_print_error


class ValsException(click.Group):
    """Custom class that overrides the default error handling for click"""

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            pretty_print_error(e)
            sys.exit(1)


@click.group(cls=ValsException)
def cli():
    pass


cli.add_command(suite_group)
cli.add_command(run_group)
cli.add_command(rag_group)
