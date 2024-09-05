import os

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

from .auth import _get_auth_token, _get_region

VALS_ENV = os.getenv("VALS_ENV")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "jsonschemas")
SUITE_SCHEMA_PATH = os.path.join(SCHEMA_PATH, "suiteschema.json")
RUN_SCHEMA_PATH = os.path.join(SCHEMA_PATH, "run_params_schema.json")


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


def get_client():
    # We do not share clients because it kills multithreading, instead we just
    # create a new client each request.
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
    response = get_client().execute(query)

    # TODO: Error check
    return response["ragSuites"]
