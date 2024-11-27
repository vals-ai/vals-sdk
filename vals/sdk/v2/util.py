import hashlib
from io import BytesIO

import httpx
import requests
from vals.graphql_client.client import Client as AriadneClient
from vals.sdk.auth import _get_auth_token
from vals.sdk.util import be_host


def get_ariadne_client() -> AriadneClient:
    """
    Use the new codegen-based client
    """
    headers = {"Authorization": _get_auth_token()}
    return AriadneClient(
        url=f"{be_host()}/graphql/",
        http_client=httpx.AsyncClient(headers=headers, timeout=60),
    )


def md5_hash(file) -> str:
    """Produces an md5 hash of the file."""
    hasher = hashlib.md5()
    while chunk := file.read(8192):  # Read in 8 KB chunks
        hasher.update(chunk)
    hash = hasher.hexdigest()
    file.seek(0)
    return hash


def parse_file_id(file_id: str) -> tuple[str, str, str | None]:
    if len(file_id) >= 37 and file_id[-37] == ";":
        # This is a heuristic to check if we are in the old file id regime.
        # I checked and all previously uploaded files should match this
        # Counting the number of slashes ensures that it's not a new file id, that
        # happens to have a semicolon in the wrong spot
        org, rest = file_id.split("/")
        filename, test_suite_id = rest.split(";")
        return org, filename, None

    tokens = file_id.split("/")
    if len(tokens) != 2:
        raise Exception(f"Improperly formatted file_id: {file_id}")

    org, filename_with_hash = tokens

    hash, filename = filename_with_hash.split("-", 1)

    if len(hash) != 32:
        raise Exception(f"Improperly formatted file_id: {file_id}")

    return org, filename, hash


def read_file(file_id: str) -> BytesIO:
    response = requests.post(
        url=f"{be_host()}/download_file/?file_id={file_id}",
        headers={"Authorization": _get_auth_token()},
    )
    return BytesIO(response.content)
