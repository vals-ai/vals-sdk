import asyncio

from vals.sdk.v2.suite import Check, Suite, Test


async def main():
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

    # List all suites
    print("List of suites: ", await Suite.list_suites())

    # Start a run
    run_id = await suite.run(model_under_test="gpt-4o-mini", wait_for_completion=True)

    # Update the suite
    suite.tests[0] = Test(
        input_under_test="Who was the third President?",
        checks=[Check(operator="equals", criteria="John Adams")],
    )
    await suite.update()

    # Start a second run
    run_id = await suite.run(model_under_test="gpt-4o-mini", wait_for_completion=True)

    # Delete the suite
    await suite.delete()


if __name__ == "__main__":
    asyncio.run(main())
