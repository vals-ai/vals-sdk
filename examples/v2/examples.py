import asyncio

from vals.sdk.v2.suite import Check, Suite, Test


async def create_suite():
    # Try creating a new suite.
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
    # NOTE: It doesn't push until we actually call the update() method.
    await suite.update()


async def pull_suite():
    """
    Example of pulling a suite that already exists.
    """
    # TODO: Replace this with your own suite id.
    suite = await Suite.from_id("79479ea0-5aed-4ff4-9545-6099bdf446f3")

    print(suite)

    # Can update the suite and push it back up to the server.
    suite.title = suite.title + " - UPDATED"
    await suite.update()


async def run_examples():
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

    # Run the suite.
    run = await suite.run(model_under_test="gpt-4o-mini", wait_for_completion=True)

    # Basic fields
    print(f"Run URL: {run.url}")
    print(f"Pass percentage: {run.pass_percentage}")

    # Get CSV of results.
    with open("out.csv", "wb") as f:
        f.write(await run.get_csv())

    # Can look at individual test results
    print(run.test_results[0].check_results[0])

    # Refresh the local data based on what's on the server.
    # run.pull()

    # If you don't set wait for run completion to be true when you do suite.run(),
    # you can also call run.wait_for_completion() to block until the run is complete.
    # await run.wait_for_completion()


if __name__ == "__main__":
    asyncio.run(run_examples())
