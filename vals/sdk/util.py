import hashlib
import os
from io import BytesIO

import httpx
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from vals.graphql_client.client import Client as AriadneClient
from vals.sdk.auth import _get_auth_token, _get_region

VALS_ENV = os.getenv("VALS_ENV")


def read_pdf(file: BytesIO):
    """
    Convenience method to parse PDFs to strings
    """
    text = ""
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text


def read_docx(file: BytesIO):
    """
    Convenience method to parse docx files to strings
    """
    output = pypandoc.convert_text(file.read(), to="plain", format="docx")
    return output


def be_host():
    region = _get_region()
    if region == "eu-north-1":
        return "https://europebe.playgroundrl.com"
    if VALS_ENV == "LOCAL":
        return "http://localhost:8000"
    if VALS_ENV == "DEV":
        return "https://devbe.playgroundrl.com"

    return "https://prodbe.playgroundrl.com"


def fe_host():
    region = _get_region()
    if region == "eu-north-1":
        return "https://eu.platform.vals.ai"
    if VALS_ENV == "LOCAL":
        return "http://localhost:3000"
    if VALS_ENV == "DEV":
        return "https://dev.platform.vals.ai"

    return "https://platform.vals.ai"


def get_client_legacy():
    """Legacy client, only used for rag suites"""

    transport = RequestsHTTPTransport(
        url=f"{be_host()}/graphql/",
        headers={"Authorization": _get_auth_token()},
        verify=VALS_ENV != "LOCAL",
    )

    client_ = Client(transport=transport, fetch_schema_from_transport=True)
    return client_


def list_rag_suites():
    query = gql(
        f"""
        query getRagSuites {{
            ragSuites {{
            id
            org
            path
            query
            }}
            }}
        """
    )
    response = get_client_legacy().execute(query)

    # TODO: Error check
    return response["ragSuites"]


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
