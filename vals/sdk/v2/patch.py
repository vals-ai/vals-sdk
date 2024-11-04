from functools import wraps
from typing import Callable

from openai import OpenAI

# TODO: Is there a better way to do this besides global variables?
in_tokens = 0
out_tokens = 0


def _wrap_chatcompletion(func: Callable):
    @wraps(func)
    def wrapper(**kwargs):
        response = func(**kwargs)
        global in_tokens, out_tokens
        in_tokens += response.usage.prompt_tokens
        out_tokens += response.usage.completion_tokens

        return response

    return wrapper


# External Facing
def patch(client: OpenAI):
    """
    Calling this function allows the Vals SDK to collect token metadata from any calls to OpenAI
    or models that use the OpenAI API.
    """
    client.chat.completions.create = _wrap_chatcompletion(  # type: ignore
        client.chat.completions.create
    )
    return client
