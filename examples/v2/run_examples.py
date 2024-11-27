import asyncio
import json
from io import BytesIO
from typing import Any

from vals.sdk.sdk import read_docx
from vals.sdk.v2.suite import Check, Suite, Test
from vals.sdk.v2.types import QuestionAnswerPair


async def run_with_model_under_test():
    """Run the suite on a stock model, gpt-4o-mini"""
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()
    run = await suite.run(model="gpt-4o-mini", wait_for_completion=True)

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")

    # Can save the results to a CSV file.
    run.to_csv("out.csv")


async def run_with_function():
    """Run the suite on a custom model function."""
    suite = Suite(
        title="Test Suite",
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()

    def function(input_under_test: str) -> str:
        # This would be replaced with your custom model.
        return input_under_test + "!!!"

    run = await suite.run(model=function, wait_for_completion=True)

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")


async def run_with_function_context_and_files():
    """Run the suite with context and files."""
    suite = Suite(
        title="Test Suite",
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
                context={
                    "message_history": [
                        {"role": "user", "content": "What is QSBS?"},
                        {"role": "assistant", "content": "QSBS is a company."},
                    ]
                },
                files_under_test=["data_files/postmoney_safe.docx"],
            ),
        ],
    )
    await suite.create()

    def function(
        input_under_test: str, files: dict[str, BytesIO], context: dict[str, Any]
    ) -> str:
        # Your LLM would leverage the context, the files, and the input_under_test
        # to return a response.
        return input_under_test

    run = await suite.run(model=function, wait_for_completion=True)

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")


async def run_with_qa_pairs():
    """Run the suite with QA pairs."""
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )

    await suite.create()

    qa_pairs = [QuestionAnswerPair(input_under_test="What is QSBS?", llm_output="QSBS")]

    run = await suite.run(
        model=qa_pairs, model_name="test-model", wait_for_completion=True
    )

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")


async def run_with_qa_pairs_and_context():
    """Run the suite with QA pairs and context."""
    context = {
        "message_history": [
            {"role": "user", "content": "What is QSBS?"},
            {"role": "assistant", "content": "QSBS is a company."},
        ]
    }
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
                context=context,
            ),
        ],
    )

    await suite.create()

    qa_pairs = [
        QuestionAnswerPair(
            input_under_test="What is QSBS?", llm_output="QSBS", context=context
        )
    ]

    run = await suite.run(
        model=qa_pairs, model_name="test-model", wait_for_completion=True
    )

    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")


async def all():
    await run_with_model_under_test()
    await run_with_function()
    await run_with_function_context_and_files()
    await run_with_qa_pairs()
    await run_with_qa_pairs_and_context()


if __name__ == "__main__":
    asyncio.run(all())
