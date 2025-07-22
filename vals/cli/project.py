"""Project management CLI commands."""

import asyncio
import click
from tabulate import tabulate
from vals.sdk.project import Project


@click.group(name="project")
def project_group():
    """
    Commands related to project management, e.g. creation, deletion, etc.
    """
    pass


async def list_command_async(limit: int, offset: int, search: str = ""):
    """List all projects in the organization."""
    projects = await Project.list_projects(limit=limit, offset=offset, search=search)
    headers = ["#", "Name", "ID", "Slug", "Default"]
    rows = []
    for i, project in enumerate(projects, start=offset):
        default_marker = "✓" if project.is_default else ""
        rows.append([i, project.name, project.id, project.slug, default_marker])

    table = tabulate(rows, headers=headers, tablefmt="tsv")
    click.echo(table)


@click.command(name="list")
@click.option("-l", "--limit", type=int, default=25, help="Number of rows to return")
@click.option(
    "-o", "--offset", type=int, default=0, help="Start table at this row (0-indexed)"
)
@click.option(
    "-s", "--search", type=str, default="", help="Search query to filter projects"
)
def list_command(limit: int, offset: int, search: str):
    """
    List all projects in the organization
    """
    asyncio.run(list_command_async(limit, offset, search))


project_group.add_command(list_command)
