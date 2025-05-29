"""Project management CLI commands."""

import asyncio
import click
from tabulate import tabulate
from vals.sdk.project import Project


@click.group(name="project")
def project_group():
    """
    Manage projects
    """
    pass


async def list_command_async(limit: int, offset: int):
    """List all projects in the organization."""
    projects = await Project.list_projects(limit=limit, offset=offset - 1)
    
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
    "-o", "--offset", type=int, default=1, help="Start table at this row (1-indexed)"
)
def list_command(limit: int, offset: int):
    """
    List all projects in the organization
    """
    asyncio.run(list_command_async(limit, offset))


async def default_command_async():
    """Show the default project for the organization."""
    project = await Project.get_default_project()
    
    click.echo(f"Default Project:")
    click.echo(f"  Name: {project.name}")
    click.echo(f"  ID: {project.id}")
    click.echo(f"  Slug: {project.slug}")


@click.command(name="default")
def default_command():
    """
    Show the default project for the organization
    """
    asyncio.run(default_command_async())


project_group.add_command(list_command)
project_group.add_command(default_command)