"""
This example is similar to sdk_example.py, except we create our test suite in-line as well.
"""

import os

from openai import OpenAI
from vals.sdk.sdk import patch, run_evaluations
from vals.sdk.suite import create_suite

client = patch(OpenAI(api_key=os.environ.get("OPEN_AI_KEY")))

# This is the definition of our test suite, the same way
# one would define it in the platform
suite_data = {
    "title": "Example Suite [SDK]",
    "description": "An example suite created from the Vals SDK.",
    "tests": [
        {
            "input_under_test": "Who was the third president?",
            "checks": [
                {"operator": "grammar"},
                {"operator": "includes", "criteria": "Thomas Jefferson"},
            ],
        }
    ],
}

# This function creates a suite on the platform
suite_id = create_suite(suite_data=suite_data)


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
    f"https://platform.vals.ai/view?test_suite_id={suite_id}",
    test_function,
)
print(f"Run Id: {run_id}")
