from vals.graphql_client.client import Client as AriadneClient
from vals.sdk.auth import _get_auth_token
from vals.sdk.util import be_host


def get_ariadne_client() -> AriadneClient:
    """
    Use the new codegen-based client
    """

    return AriadneClient(
        url=f"{be_host()}/graphql/",
        headers={"Authorization": _get_auth_token()},
    )
