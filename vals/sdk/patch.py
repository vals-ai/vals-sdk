from functools import wraps
from typing import Callable

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
def patch(openai_client):
    """
    Calling this function allows the Vals SDK to collect token metadata from any calls to OpenAI
    or models that use the OpenAI API.
    """
    openai_client.chat.completions.create = _wrap_chatcompletion(  # type: ignore
        openai_client.chat.completions.create
    )
    return openai_client
