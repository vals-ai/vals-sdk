import asyncio
import base64
import hashlib
import os
import urllib.parse
from collections import defaultdict
from io import BytesIO

import httpx
import requests
from tqdm import tqdm
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


async def _download_file_async(file_id: str, client: httpx.AsyncClient):
    """Helper function to download a single file asynchronously"""
    encoded_file_id = urllib.parse.quote(file_id)
    response = await client.get(
        f"{be_host()}/download_files_bulk/?file_ids={encoded_file_id}",
        headers={"Authorization": _get_auth_token()},
    )

    if response.status_code != 200:
        raise Exception(f"Failed to download file {file_id}: {response.text}")

    return response.json()["files"][0]


async def _download_files_chunk_async(file_ids_chunk: list[str]):
    """Download a chunk of files asynchronously"""
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [_download_file_async(file_id, client) for file_id in file_ids_chunk]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def download_files_bulk(
    file_ids: list[str], download_path: str, max_concurrent_downloads: int = 50
):
    """
    Download multiple files in parallel with a maximum of max_concurrent_downloads downloads at once.

    Args:
        file_ids: List of file IDs to download
        download_path: Path where to save the downloaded files
        max_concurrent_downloads: Maximum number of concurrent downloads (default: 5)
    """

    if len(file_ids) == 0:
        raise Exception("No files to download")

    filename_to_filepath_map = {}

    os.makedirs(download_path, exist_ok=True)

    # Prepare filename groups to handle duplicates
    filename_groups = defaultdict(set)
    for file_id in file_ids:
        id, filename = file_id.split("-", 1)
        hash = id.split("/", 1)[-1]
        filename_groups[filename].add(hash)

    duplicate_files = {
        filename: ids for filename, ids in filename_groups.items() if len(ids) > 1
    }

    # Create directories for duplicate files
    for filename, hashes in duplicate_files.items():
        for hash in hashes:
            download_dir = os.path.join(download_path, hash)
            os.makedirs(download_dir, exist_ok=True)

    # Split file_ids into chunks of max_concurrent_downloads size
    chunks = [
        file_ids[i : i + max_concurrent_downloads]
        for i in range(0, len(file_ids), max_concurrent_downloads)
    ]

    total_files = len(file_ids)
    all_files_data = []

    # Progress bar for downloading
    progress_bar = tqdm(total=total_files, desc="Downloading files", unit="file")

    for chunk in chunks:
        # Use asyncio to download files in parallel
        chunk_results = await _download_files_chunk_async(chunk)

        # Handle any exceptions
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                print(f"Error downloading {chunk[i]}: {result}")
            else:
                all_files_data.append(result)

            progress_bar.update(1)

    progress_bar.close()

    # Progress bar for saving files
    progress_bar = tqdm(total=len(all_files_data), desc="Saving files", unit="file")

    # Save downloaded files
    for file_data in all_files_data:
        filename = file_data["filename"]
        hash = file_data["hash"]
        content = base64.b64decode(file_data["content"])

        if filename in duplicate_files:
            download_dir = os.path.join(download_path, hash)
            file_path = os.path.join(download_dir, filename)
        else:
            file_path = os.path.join(download_path, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        filename_to_filepath_map[filename] = os.path.abspath(file_path)

        progress_bar.update(1)

    progress_bar.close()

    return filename_to_filepath_map
