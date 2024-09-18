import json
import os
import time

from descope import DescopeClient

VALS_ENV = os.getenv("VALS_ENV", "PROD")

DEFAULT_REGION = "us-east-1"

global_api_key = None
global_in_eu = None
global_auth_dict = {}


def configure_credentials(api_key: str, in_eu: bool = False):
    """
    Configure the Vals API Key to be used with requests.
    This will take precedence over any credentials set in environment variables, or with vals login.

    API key can be generated in the Web App. If you are using the EU platform, make sure to set
    in_eu = True, otherwise leave it as the default.
    """
    global global_api_key, global_in_eu
    global_api_key = api_key.strip()
    global_in_eu = in_eu


def _get_region():
    global global_in_eu
    if global_in_eu is not None:
        return "eu-north-1" if global_in_eu else "us-east-1"

    if "VALS_REGION" in os.environ:
        vals_region = os.environ["VALS_REGION"].lower()
        if vals_region not in ["europe", "us"]:
            raise ValueError(
                f"Invalid region: {vals_region}. Must be 'europe' or 'us'."
            )

        return "eu-north-1" if vals_region == "europe" else "us-east-1"

    return DEFAULT_REGION


def get_descope_client():
    # Needs to be in this file because of circular imports
    region = _get_region()
    if region == "eu-north-1":
        project_id = "P2lXkjgPTaW5f8ZlhBzCpnxeqlpj"
    elif VALS_ENV == "LOCAL" or VALS_ENV == "DEV":
        project_id = "P2ktNOjz5Tgzs9wwS3VpShnCbmik"
    elif VALS_ENV == "PROD":
        project_id = "P2lXkZaPuDqCzGxoxGHseomQi7ac"
    else:
        raise Exception(f"Unrecognized VALS_ENV: {VALS_ENV}")
    return DescopeClient(project_id=project_id)


def _get_auth_token():
    """
    Get a new session token that can be used in Authorization header.

    Internally, reads the api key from either the VALS_ENV
    variable or a configured value, then uses the Descope SDK to
    exchange the api key for a session token.
    """
    global global_api_key, global_in_eu, global_auth_dict

    # API key was specified with configure_credentials
    if global_api_key is not None:
        api_key = global_api_key

    # API Key is specified in environment
    elif "VALS_API_KEY" in os.environ:
        api_key = os.environ["VALS_API_KEY"]
    else:
        raise Exception(
            "Either the `VALS_API_KEY` environment variable should be set, or the API key should be set with configure_credentials (in vals.sdk.auth.)."
        )

    if (
        "access_expiry" not in global_auth_dict
        or time.time() > global_auth_dict["access_expiry"]
    ):
        descopeClient = get_descope_client()

        response = descopeClient.exchange_access_key(api_key)

        global_auth_dict = {
            **global_auth_dict,
            "access_token": response["sessionToken"]["jwt"],
            "access_expiry": response["sessionToken"]["exp"],
        }

    return global_auth_dict["access_token"]
