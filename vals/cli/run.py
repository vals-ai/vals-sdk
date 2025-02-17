import asyncio
from io import TextIOWrapper

import click
from tabulate import tabulate
from vals.sdk.run import Run


@click.group(name="run")
def run_group():
    """
    Commands relating to starting or viewing runs
    """
    pass


async def pull_async(run_id: str, file: TextIOWrapper, csv: bool, _json: bool):
    run = await Run.from_id(run_id)

    if csv:
        file.write(await run.to_csv_string())
    else:
        file.write(await run.to_json_string())

    click.secho("Successfully pulled run results.", fg="green")


@click.command
@click.argument("file", type=click.File("w"), required=True)
@click.argument("run-id", type=click.STRING, required=True)
@click.option("--csv", is_flag=True, default=False, help="Save as a CSV")
@click.option("--json", is_flag=True, default=False, help="Save as a JSON")
def pull(run_id: str, file: TextIOWrapper, csv: bool, json: bool):
    """
    Pull results of a run and save it to a file.
    """
    asyncio.run(pull_async(run_id, file, csv, json))


async def list_async(
    limit: int, offset: int, suite_id: str | None, show_archived: bool, search: str
):
    run_results = await Run.list_runs(
        limit=limit,
        offset=offset - 1,
        show_archived=show_archived,
        suite_id=suite_id,
        search=search,
    )

    column_names = ["#", "Run Name", "Id", "Status", "Model", "Pass Rate", "Timestamp"]

    rows = []
    for i, run in enumerate(run_results, start=offset):
        date_str = run.timestamp.strftime("%Y/%m/%d %H:%M")
        pass_percentage_str = (
            f"{run.pass_rate:.2f}%" if run.pass_rate is not None else "N/A"
        )
        rows.append(
            [
                i,
                run.name,
                run.id,
                run.status.value,
                run.model,
                pass_percentage_str,
                date_str,
            ]
        )

    table = tabulate(rows, headers=column_names, tablefmt="tsv")
    click.echo(table)


@click.command()
@click.option(
    "-l",
    "--limit",
    type=click.INT,
    default=25,
    help="Limit the number of runs to display",
)
@click.option(
    "-o",
    "--offset",
    required=False,
    default=1,
    help="Start table at this row (1-indexed)",
)
@click.option("--suite-id", required=False, help="Filter runs by suite id")
@click.option(
    "--show-archived",
    is_flag=True,
    default=False,
    help="When enabled, archived runs are displayed in the output",
)
@click.option(
    "--search",
    type=click.STRING,
    default="",
    help="Search for a run based off its name, model or test suite title",
)
def list(
    limit: int, offset: int, suite_id: str | None, show_archived: bool, search: str
):
    """
    List runs associated with this organization
    """
    asyncio.run(list_async(limit, offset, suite_id, show_archived, search))


run_group.add_command(pull)
run_group.add_command(list)
