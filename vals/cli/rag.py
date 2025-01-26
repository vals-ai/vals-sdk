import asyncio
import json
from typing import Any, Dict

import click
import requests
from vals.cli.util import display_error_and_exit
from vals.sdk.auth import _get_auth_token
from vals.sdk.util import be_host, get_ariadne_client


@click.group(name="rag")
def rag_group():
    """
    Commands relating to starting rag evaluations.
    """
    pass


async def list_rag_suites():
    client = get_ariadne_client()

    response = await client.get_rag_suites()

    # TODO: Error check
    return response.rag_suites


async def create_rag_suite(data: Dict[str, Any]) -> str:
    client = get_ariadne_client()
    result = await client.create_rag_suite(query=data["query"], file_path=data["id"])
    # Extract the 'id' of the created RagSuite from the result
    suite_id = result.update_rag_suite.rag_suite.id
    return suite_id


async def list_command_async():
    suites = await list_rag_suites()
    if len(suites) == 0:
        click.secho("No rag suites found", fg="red")
        return

    suite_text = "\n".join([f"{i}: {s.id} {s.query}" for i, s in enumerate(suites)])
    click.echo(suite_text)


@click.command(name="list")
def list_command():
    """
    List rag suites associated with this organization
    """
    asyncio.run(list_command_async())


async def upload_command_async(file: str, query: str):
    with open(file, "rb") as f:
        response = requests.post(
            f"{be_host()}/upload_rag_file/",
            files={"file": f},
            headers={"Authorization": _get_auth_token()},
        )
        if response.status_code != 200:
            display_error_and_exit(f"Failed to upload file {file}")

    suite_id = await create_rag_suite(
        {"id": response.json()["file_id"], "query": query}
    )

    click.secho(
        "Successfully Uploaded RAG. ID: {}".format(suite_id),
        fg="green",
    )


@click.command(name="upload")
@click.argument("file", type=click.STRING, required=True)
@click.argument("query", type=click.STRING, required=True)
def upload_command(file: str, query: str):
    """
    Uploads a CSV file to the server for reranking documents.

    Args:
        file (str): The path to the CSV file. The CSV file should
        have a column named "text" containing the documents to be
        reranked as well as a column named id containing the unique
        identifier for each document.

    Returns:
        str: The file ID that can be used to rerank the documents.
    """
    asyncio.run(upload_command_async(file, query))


async def rerank_command_async(
    id: str,
    query: str = "",
    light_ranking: bool = True,
    model: str = "gpt-3",
):
    suites = await list_rag_suites()

    # TODO: Error check
    file_id, original_query = next(
        ((suite.path, suite.query) for suite in suites if suite.id == id),
        None,
    )

    if str(query) == "":
        query = str(original_query)
        click.secho(
            f"Using original query: {query}.",
        )
    else:
        click.secho(
            f"Using custom query: {str(query)}.",
        )

    response = requests.get(
        f"{be_host()}/evaluate_rag_ranking/",
        params={
            "file_id": file_id,
            "query": query,
            "light_ranking": light_ranking,
            "model": model,
        },
        headers={"Authorization": _get_auth_token()},
    )

    if response.status_code != 200:
        display_error_and_exit(f"Failed to run reranking: {file_id}")

    scores = response.json()
    scores["query"] = query

    file_name = f"rag_scores_{id}.json"
    with open(file_name, "w") as f:
        json.dump(scores, f, indent=4)

    click.secho(
        f"Successfully Ran RAG Rankings. Scores saved to {file_name}.",
        fg="green",
    )


@click.command(name="rerank")
@click.argument("id", type=click.STRING, required=True)
@click.option("--query", type=click.STRING, default="")
@click.option("--light-ranking", type=click.STRING, default=True)
@click.option("--model", type=click.STRING, default="gpt-3")
def rerank_command(
    id: str | None = None,
    query: str = "",
    light_ranking: bool = True,
    model: str = "gpt-3",
):
    """
    Reranks the documents in the file based on the provided query.

    Args:
        file_id (str, optional): The ID of the file containing the documents to be reranked.
        query (str, optional): The query to be used for reranking, if not provided the original query will be used.
        light_ranking (bool, optional): Whether to use light ranking. Defaults to True.
        model (str, optional): The model to be used. Defaults to "gpt-3".

    Returns:
        None. Prints a list of scores that match the order of the documents in the file.
    """
    asyncio.run(rerank_command_async(id, query, light_ranking, model))


rag_group.add_command(upload_command)
rag_group.add_command(rerank_command)
rag_group.add_command(list_command, "list")
