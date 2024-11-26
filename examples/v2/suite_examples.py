"""
Set of examples on how to use the Suite class.

Note: The file paths assume this is run from the `examples/` directory.
"""

import asyncio
import json

from vals.graphql_client.input_types import MetadataType, QuestionAnswerPairInputType
from vals.sdk.v2.suite import Check, Suite, Test
from vals.sdk.v2.types import Metadata, QuestionAnswerPair


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

    # Generate 1000 QA pairs with realistic inputs and outputs
    qa_pairs = []
    for i in range(1000):
        input_text = f"Please analyze the following complex tax scenario and provide detailed guidance. A client has a startup that was founded in 2019 and has received multiple rounds of funding. They're considering a stock sale that could qualify for QSBS treatment under Section 1202. The company has maintained C-corporation status since inception and has assets under $50 million. The client acquired their shares through a combination of founder's stock and subsequent purchases. They're approaching the 5-year holding period for some shares but not others. Key considerations include: 1) The impact of previous funding rounds on the company's asset test compliance 2) Whether any disqualifying redemptions occurred 3) How to calculate the excluded gain given mixed acquisition dates 4) Implications of recent changes to state tax treatment of QSBS gains 5) Strategic timing of sales to maximize tax benefits. Case #{i}"

        llm_output = f"""Based on the complex tax scenario provided, here is my detailed analysis and guidance:

1. QSBS Qualification Analysis:
- The company appears to meet the basic C-corporation requirement since inception
- The $50M gross asset test is currently satisfied
- The active business requirement likely met as a startup

2. Holding Period Considerations:
- Shares acquired in 2019 are approaching but not yet at 5-year mark
- Recommend tracking each share lot separately
- Consider structured sales to optimize timing

3. Funding Round Impact:
- Need to verify asset test at time of each funding round
- Previous rounds may have affected qualification
- Important to maintain detailed records

4. Redemption Analysis:
- No disqualifying redemptions mentioned but should verify
- Look back period rules apply
- Document all stock transactions

5. Gain Exclusion Calculation:
- Will vary by share lot acquisition date
- Maximum exclusion is greater of $10M or 10x basis
- State treatment requires separate analysis

Recommendations:
1. Maintain detailed share lot tracking
2. Consider phased sale approach
3. Document asset test compliance
4. Consult state-specific guidance
5. Engage qualified tax counsel

Case #{i} analysis complete. Let me know if you need any clarification on specific points."""

        qa_pairs.append(
            QuestionAnswerPairInputType(
                input_under_test=input_text,
                llm_output=llm_output,
                file_ids=[],
            )
        )
    await suite._create_qa_set(qa_pairs, {}, "")


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
    print(f"Created suite: {suite._id}")
    await suite.delete()


async def pull_suite():
    """
    Example of pulling a suite that already exists.
    """
    # TODO: Replace this with your own suite id.
    suite = await Suite.from_id("de9860f4-bf22-43ee-85a6-279a282e7cb4")

    print(f"Pulling: Suite Title: {suite.title}")
    print(f"Global Checks: {suite.global_checks}")
    # Can update the suite and push it back up to the server.
    suite.title = suite.title + " - UPDATED TITLE"
    await suite.update()
    print(json.dumps(suite.to_dict(), indent=2))


async def load_from_json():
    """Create a suite from a json file."""
    suite = await Suite.from_json_file("example_suites/example_suite.json")
    print(f"Loaded from JSON: {suite}")


async def all():
    # await list_suites()
    await create_suite()
    # await create_and_delete_suite()
    # await create_suite_with_files()
    # await update_suite()
    # await pull_suite()
    # await load_from_json()


if __name__ == "__main__":
    asyncio.run(all())
