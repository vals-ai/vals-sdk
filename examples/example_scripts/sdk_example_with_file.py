"""
This example is similar to sdk_example.py, except we create our test suite in-line as well.

We also add an additional file, and use consistent_with_docs to check it. 

NOTE: You should run this from the `examples/` folder. 
"""

import os
from io import BytesIO

from openai import OpenAI
from vals.sdk.sdk import patch, read_docx, run_evaluations
from vals.sdk.suite import create_suite

client = patch(OpenAI(api_key=os.environ.get("OPEN_AI_KEY")))

# This is the definition of our test suite, the same way
# one would define it in the platform
suite_data = {
    "title": "Example Suite - Files [SDK]",
    "description": "An example suite created from the Vals SDK.",
    "tests": [
        {
            "input_under_test": "Does this SAFE have an MFN clause?",
            "file_under_test": "data_files/postmoney_safe.docx",
            "checks": [
                {"operator": "grammar"},
                {"operator": "negative_answer"},
                {"operator": "consistent_with_docs"},
            ],
        }
    ],
}

# This function creates a suite on the platform
suite_id = create_suite(suite_data=suite_data)


def test_function(test_input: str, files: dict[str, BytesIO]):

    prompt = f"You are a pirate, answer in the speaking style of a pirate.\n\n Use the following documents to inform your answer:\n\n\n"

    for filename, file in files.items():
        # We could equivalently use read_pdf here, based on the extension.
        prompt += filename + ":\n"
        prompt += read_docx(file) + "\n\n"

    temp = 0.1

    gpt_client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    response = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt + test_input}],
        temperature=temp,
    )
    return response.choices[0].message.content


run_id = run_evaluations(
    f"https://dev.platform.vals.ai/view?test_suite_id={suite_id}",
    test_function,
)
print(f"Run Id: {run_id}")
