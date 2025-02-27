import base64
import hashlib
import os
from io import BytesIO
from collections import defaultdict
import urllib.parse
import httpx
import requests
from vals.graphql_client.client import Client as AriadneClient
from vals.sdk.auth import _get_auth_token, _get_region

VALS_ENV = os.getenv("VALS_ENV")


def read_pdf(file: BytesIO):
    """
    Convenience method to parse PDFs to strings
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise Exception(
            "To use read_pdf and read_docx, please run `pip install vals[parsing]`"
        )

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
    try:
        import pypandoc
    except ImportError:
        raise Exception("To use read_docx, please run `pip install vals[parsing]`")

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
    if VALS_ENV == "BENCH":
        return "https://benchbe.playgroundrl.com"

    return "https://prodbe.playgroundrl.com"


def fe_host():
    region = _get_region()
    if region == "eu-north-1":
        return "https://eu.platform.vals.ai"
    if VALS_ENV == "LOCAL":
        return "http://localhost:3000"
    if VALS_ENV == "DEV":
        return "https://dev.platform.vals.ai"
    if VALS_ENV == "BENCH":
        return "https://bench.platform.vals.ai"

    return "https://platform.vals.ai"


def get_ariadne_client() -> AriadneClient:
    """
    Use the new codegen-based client
    """

    def append_auth_header(request: httpx.Request):
        request.headers["Authorization"] = _get_auth_token()
        return request

    return AriadneClient(
        url=f"{be_host()}/graphql/",
        http_client=httpx.AsyncClient(auth=append_auth_header, timeout=60),
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


def download_files_bulk(file_ids: list[str], download_path: str):

    if len(file_ids) == 0:
        raise Exception("No files to download")

    encoded_file_ids = [urllib.parse.quote(file_id) for file_id in file_ids]
    response = requests.get(
        f"{be_host()}/download_files_bulk/?file_ids={','.join(encoded_file_ids)}",
        headers={"Authorization": _get_auth_token()},
    )

    if response.status_code != 200:
        raise Exception(f"Failed to download files: {response.text}")

    files_data = response.json()["files"]

    os.makedirs(download_path, exist_ok=True)

    filename_groups = defaultdict(set)

    for file_id in file_ids:
        id, filename = file_id.split("-", 1)
        hash = id.split("/", 1)[-1]
        filename_groups[filename].add(hash)

    duplicate_files = {
        filename: ids for filename, ids in filename_groups.items() if len(ids) > 1
    }

    for filename, hashes in duplicate_files.items():
        for hash in hashes:
            download_dir = os.path.join(download_path, hash)
            os.makedirs(download_dir, exist_ok=True)

    for file_data in files_data:
        filename = file_data["filename"]
        hash = file_data["hash"]

        content = base64.b64decode(file_data["content"])

        if filename in duplicate_files:
            download_dir = os.path.join(download_path, hash)
            file_path = os.path.join(download_dir, filename)

            with open(file_path, "wb") as f:
                f.write(content)
        else:
            file_path = os.path.join(download_path, filename)

            with open(file_path, "wb") as f:
                f.write(content)
