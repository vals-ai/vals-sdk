"""
Set of examples on how to use the Suite class.

Note: The file paths assume this is run from the `examples/` directory.
"""

import asyncio
import json

from vals import Check, Suite, Test


async def list_suites():
    """List all suites."""
    suites = await Suite.list_suites()
    for suite in suites:
        print(f"Suite: {suite.title}")


async def create_suite():
    """Create a single, basic test suite."""
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
            Test(
                input_under_test="What is an 83 election?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()


async def create_suite_with_files():
    """
    Create a test suite that has a file upload as part of the test input.
    """
    # Try creating a new suite.
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is the MFN clause?",
                files_under_test=["data_files/postmoney_safe.docx"],
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()


async def update_suite():
    """Create a suite, then update its contents."""
    # Create the suite
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
            Test(
                input_under_test="What is an 83 election?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()

    # Update the suite
    suite.tests[0] = Test(
        input_under_test="Who was the third President?",
        checks=[Check(operator="equals", criteria="John Adams")],
    )
    await suite.update()


async def create_and_delete_suite():
    """Create a suite, then delete it."""
    suite = Suite(
        title="Test Suite to Delete", global_checks=[Check(operator="grammar")]
    )
    await suite.create()
    print(f"Created suite: {suite.id}")
    await suite.delete()


async def pull_suite():
    """
    Example of pulling a suite that already exists.
    """
    suite = Suite(
        title="Test Suite to Delete", global_checks=[Check(operator="grammar")]
    )
    await suite.create()
    # TODO: Replace this with your own suite id.
    suite = await Suite.from_id(suite.id)

    print(f"Pulling: Suite Title: {suite.title}")
    print(f"Global Checks: {suite.global_checks}")
    # Can update the suite and push it back up to the server.
    suite.title = suite.title + " - UPDATED TITLE"
    await suite.update()
    print(json.dumps(suite.to_dict(), indent=2))


async def pull_suite_with_files():
    suite = Suite(
        title="Test Suite",
        global_checks=[Check(operator="grammar")],
        tests=[
            Test(
                input_under_test="What is the MFN clause?",
                files_under_test=["data_files/postmoney_safe.docx"],
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
        ],
    )
    await suite.create()

    suite2 = await Suite.from_id(suite.id, download_files=True)
    print(suite2)


async def load_from_json():
    """Create a suite from a json file."""
    suite = await Suite.from_json_file("suites/example_suite.json")
    print(f"Loaded from JSON: {suite}")


async def move_test_between_suites():
    suite1 = Suite(
        title="Suite 1",
        tests=[
            Test(
                input_under_test="What is QSBS?",
                checks=[Check(operator="equals", criteria="QSBS")],
            )
        ],
    )
    await suite1.create()

    suite2 = Suite(
        title="Suite 2",
        tests=[
            Test(
                input_under_test="What is an 83 election?",
                checks=[Check(operator="equals", criteria="QSBS")],
            ),
            suite1.tests[0],
        ],
    )
    await suite2.create()

    print(f"Suite 1: {suite1.url}")
    print(f"Suite 2: {suite2.url}")


async def all():
    await pull_suite_with_files()
    await list_suites()
    await create_suite()
    await create_and_delete_suite()
    await create_suite_with_files()
    await update_suite()
    await pull_suite()
    await load_from_json()
    await move_test_between_suites()


if __name__ == "__main__":
    asyncio.run(all())
