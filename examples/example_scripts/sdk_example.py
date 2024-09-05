import os

from openai import OpenAI
from vals.sdk.sdk import patch, run_evaluations

client = patch(OpenAI(api_key=os.environ.get("OPEN_AI_KEY")))

# Before running this, create a test suite on the Vals AI platform


def test_function(test_input: str):
    prompt = "You are a pirate, answer in the speaking style of a pirate.\n\n"
    temp = 0.2

    gpt_client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    response = gpt_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt + test_input}],
        temperature=temp,
    )
    return response.choices[0].message.content


run_id = run_evaluations(
    # Replace with the link suite link
    "https://platform.vals.ai/view?test_suite_id=xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx",
    test_function,
)
