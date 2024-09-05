import os

from openai import OpenAI
from vals.sdk.sdk import patch, run_evaluations

client = patch(OpenAI(api_key=os.environ.get("OPEN_AI_KEY")))


def test_function(test_input: str):

    prompt = (
        f"Answer as if you are a pirate. "
        + f"Prompt: {test_input}\n\n"
    )
    temp = 0.2

    gpt_client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    response = gpt_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt + test_input}],
        temperature=temp,
    )
    return response.choices[0].message.content


run_id = run_evaluations(
    # Replace with your suite link
    "https://www.platform.vals.ai/view?test_suite_id=ea00a4f2-c1d3-4d5f-89b6-be1e76fd344f",
    test_function,
)
