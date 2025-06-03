import asyncio
import base64
import hashlib
import os
from collections import defaultdict
from io import BytesIO

import httpx
import requests
from tqdm import tqdm

from vals.graphql_client.client import Client as AriadneClient
from vals.sdk.auth import _get_auth_token, _get_region
from vals.sdk.types import File

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


def read_files(file_ids: list[str]) -> dict[str, BytesIO]:
    response = requests.post(
        url=f"{be_host()}/download_files_bulk/",
        headers={"Authorization": _get_auth_token()},
        json={"file_ids": file_ids},
    )

    return {
        file["filename"]: BytesIO(base64.b64decode(file["content"]))
        for file in response.json()["files"]
    }


async def _download_file_async(file_id: str, client: httpx.AsyncClient):
    """Helper function to download a single file asynchronously"""
    response = await client.post(
        f"{be_host()}/download_files_bulk/",
        headers={"Authorization": _get_auth_token()},
        json={"file_ids": [file_id]},
    )

    if response.status_code != 200:
        raise Exception(f"Failed to download file {file_id}: {response.text}")

    result = response.json()["files"][0]

    result["file_id"] = file_id

    return result


async def _download_files_chunk_async(file_ids_chunk: list[str]):
    """Download a chunk of files asynchronously"""
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [_download_file_async(file_id, client) for file_id in file_ids_chunk]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def download_files_bulk(
    files: list[File], documents_download_path: str, max_concurrent_downloads: int = 50
) -> dict[str, str]:
    """
    Download multiple files in parallel with a maximum of max_concurrent_downloads downloads at once.
    Process files as they are downloaded to minimize memory usage.

    Args:
        files: List of File objects to download
        download_path: Path where to save the downloaded files
        max_concurrent_downloads: Maximum number of concurrent downloads (default: 50)
    """
    if len(files) == 0:
        raise Exception("No files to download")

    file_id_to_file_path = {}
    os.makedirs(documents_download_path, exist_ok=True)

    # Pre-process files to identify duplicates by filename
    filename_count = defaultdict(set)
    for file in files:
        filename_count[file.file_name].add(file.hash)

    # Split files into chunks
    chunks = [
        files[i : i + max_concurrent_downloads]
        for i in range(0, len(files), max_concurrent_downloads)
    ]

    # Progress bar for the entire operation
    with tqdm(
        total=len(files), desc="Downloading and saving files", unit="file"
    ) as progress_bar:
        for chunk in chunks:
            # Download chunk of files
            chunk_results = await _download_files_chunk_async(
                [file.file_id for file in chunk]
            )

            # Process each result as it comes
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    print(f"Error downloading {chunk[i]}: {result}")
                    progress_bar.update(1)
                    continue

                # Extract file info
                filename = result["filename"]
                hash = result["hash"]
                content = base64.b64decode(result["content"])

                # Determine file path - use hash directory if filename has duplicates
                file_path = os.path.join(documents_download_path, filename)
                relative_file_path = os.path.join(
                    os.path.basename(documents_download_path), filename
                )

                if len(filename_count[filename]) > 1 or os.path.exists(file_path):
                    hash_dir = os.path.join(documents_download_path, hash)
                    os.makedirs(hash_dir, exist_ok=True)
                    file_path = os.path.join(hash_dir, filename)
                    relative_file_path = os.path.join(
                        os.path.basename(documents_download_path), hash, filename
                    )

                # Write the file
                with open(file_path, "wb") as f:
                    f.write(content)

                file_id_to_file_path[result["file_id"]] = relative_file_path

                # Update progress
                progress_bar.update(1)

    progress_bar.close()

    return file_id_to_file_path
