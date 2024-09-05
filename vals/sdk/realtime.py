import requests
from vals.sdk.auth import _get_auth_token
from vals.sdk.util import be_host


def evaluate(output: str, checks: list[dict[str, str]]):
    """
    Evaluate an output in realtime, against a set of checks

    Each check should have this format:
    {"operator": <operator>, "criteria": <criteria>}
    """

    body = {"output": output, "checks": checks}

    response = requests.post(
        url=f"{be_host()}/live/",
        headers={"Authorization": _get_auth_token()},
        json=body,
    )

    response_body = response.json()

    keys_to_keep = ["auto_eval", "feedback"]

    return [{k: v for k, v in d.items() if k in keys_to_keep} for d in response_body[0]]
