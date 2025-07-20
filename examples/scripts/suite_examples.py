"""
Suite Examples - Comprehensive demonstrations of Suite class functionality.

This file shows all operations you can perform with the Suite class:
- Creating suites with various test types
- CRUD operations (Create, Read, Update, Delete)
- Working with files
- Using tags and metadata
- Global checks vs test-specific checks
- Loading/saving from JSON
- Error handling

Note: File paths assume this is run from the `examples/` directory.
"""

import asyncio
import json
import os
from io import BytesIO
from vals import Suite, Test, Check


async def create_simple_suite():
    """Create a basic test suite with a few tests."""
    print("\n=== Creating Simple Suite ===")

    suite = Suite(
        title="Basic Math Test Suite",
        description="Tests for basic arithmetic operations",
        tests=[
            Test(
                input_under_test="What is 2+2?",
                checks=[
                    Check(operator="equals", criteria="4"),
                    Check(operator="excludes", criteria="5"),
                ],
            ),
            Test(
                input_under_test="What is 10 divided by 2?",
                checks=[
                    Check(operator="equals", criteria="5"),
                    Check(operator="includes", criteria="5"),
                ],
            ),
        ],
    )

    await suite.create()
    print(f"Created suite: {suite.title} (ID: {suite.id})")
    print(f"URL: {suite.url}")
    return suite


async def create_suite_with_global_checks():
    """Create suite with global checks that apply to all tests."""
    print("\n=== Creating Suite with Global Checks ===")

    suite = Suite(
        title="Professional Response Suite",
        description="Tests that require professional, error-free responses",
        global_checks=[
            Check(operator="grammar", criteria=""),
            Check(operator="excludes", criteria="error"),
            Check(operator="excludes", criteria="sorry"),
            Check(operator="excludes", criteria="undefined"),
        ],
        tests=[
            Test(
                input_under_test="What is the capital of France?",
                checks=[Check(operator="equals", criteria="Paris")],
                # This test will have both its specific check AND all global checks
            ),
            Test(
                input_under_test="Explain quantum computing in simple terms",
                checks=[],  # This test will only have the global checks
            ),
        ],
    )

    await suite.create()
    print(f"Created suite with {len(suite.global_checks)} global checks")
    return suite


async def create_suite_with_files():
    """Create a test suite with file uploads."""
    print("\n=== Creating Suite with Files ===")

    # Create in-memory file
    sample_doc = BytesIO(b"This is a sample document for testing file uploads.")
    sample_doc.name = "sample.txt"

    # Check if local file exists
    local_file_path = "data_files/postmoney_safe.docx"
    tests = [
        Test(
            input_under_test="What is the content of this text file?",
            files_under_test=[sample_doc.name],
            checks=[
                Check(operator="includes", criteria="sample document"),
                Check(operator="includes", criteria="testing"),
            ],
        )
    ]

    if os.path.exists(local_file_path):
        with open(local_file_path, "rb") as f:
            docx_file = BytesIO(f.read())
            docx_file.name = "postmoney_safe.docx"

            tests.append(
                Test(
                    input_under_test="What type of legal document is this?",
                    files_under_test=[docx_file.name],
                    checks=[
                        Check(operator="includes", criteria="SAFE"),
                        Check(
                            operator="includes_any", criteria="agreement|contract|legal"
                        ),
                    ],
                )
            )

    suite = Suite(
        title="Document Analysis Suite",
        description="Tests that analyze uploaded documents",
        tests=tests,
    )

    await suite.create()
    print(f"Created suite with {len(suite.tests)} file-based tests")
    return suite


async def create_suite_with_tags_and_metadata():
    """Create suite with tags for categorization and filtering."""
    print("\n=== Creating Suite with Tags and Metadata ===")

    suite = Suite(
        title="Categorized Knowledge Tests",
        description="Tests organized by category and difficulty",
        tests=[
            Test(
                input_under_test="What is 2+2?",
                tags=["category:math", "difficulty:easy", "priority:high", "subject:arithmetic"],
                context={"instruction": "This is a basic arithmetic question"},
                checks=[Check(operator="equals", criteria="4")],
            ),
            Test(
                input_under_test="Solve for x: 2x + 5 = 13",
                tags=["category:math", "difficulty:medium", "priority:medium", "subject:algebra"],
                checks=[Check(operator="equals", criteria="4")],
            ),
            Test(
                input_under_test="Who wrote 'Romeo and Juliet'?",
                tags=["category:literature", "difficulty:easy", "priority:low", "subject:shakespeare"],
                checks=[Check(operator="includes", criteria="Shakespeare")],
            ),
            Test(
                input_under_test="Explain the theory of relativity",
                tags=["category:science", "difficulty:hard", "priority:medium", "subject:physics"],
                context={"instruction": "Focus on the key concepts in simple language"},
                checks=[
                    Check(operator="includes", criteria="Einstein"),
                    Check(
                        operator="includes_any", criteria="space|time|speed of light"
                    ),
                ],
            ),
        ],
    )

    await suite.create()
    print(f"Created suite with {len(suite.tests)} tagged tests")
    
    # Show test distribution
    categories = {}
    for test in suite.tests:
        for tag in test.tags:
            if tag.startswith("category:"):
                cat = tag.split(":", 1)[1]
                categories[cat] = categories.get(cat, 0) + 1

    print("\nTest distribution by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} tests")
    
    return suite


async def create_suite_with_context():
    """Create suite where tests have context information."""
    print("\n=== Creating Suite with Context ===")

    suite = Suite(
        title="Context-Based Q&A Suite",
        description="Tests that require understanding provided context",
        tests=[
            Test(
                input_under_test="What year was the company founded?",
                context={"background": "TechCorp was established in 2015 by three Stanford graduates. The company focuses on AI solutions."},
                checks=[Check(operator="includes", criteria="2015")],
            ),
            Test(
                input_under_test="How many founders were there?",
                context={"background": "TechCorp was established in 2015 by three Stanford graduates. The company focuses on AI solutions."},
                checks=[
                    Check(operator="includes", criteria="three"),
                    Check(operator="includes", criteria="3"),
                ],
            ),
            Test(
                input_under_test="Summarize the patient's main symptoms",
                context={
                    "patient_info": """Patient: John Doe, Age: 45
                Chief complaints: Persistent headache for 3 days, mild fever (99.5F), fatigue
                Medical history: Hypertension, controlled with medication
                Current medications: Lisinopril 10mg daily"""
                },
                checks=[
                    Check(operator="includes", criteria="headache"),
                    Check(operator="includes", criteria="fever"),
                    Check(operator="includes", criteria="fatigue"),
                ],
            ),
        ],
    )

    await suite.create()
    print(f"Created context-based suite with {len(suite.tests)} tests")
    return suite


async def update_existing_suite():
    """Create a suite, then update its contents."""
    print("\n=== Updating Existing Suite ===")

    # Create initial suite
    suite = Suite(
        title="Suite to Update",
        description="Original description",
        tests=[
            Test(
                input_under_test="Original question 1",
                checks=[Check(operator="grammar", criteria="")],
            )
        ],
    )
    await suite.create()
    print(f"Created suite: {suite.title}")

    # Update various aspects
    suite.title = "Updated Suite Name"
    suite.description = "This description has been updated with more details"

    # Add new tests
    suite.tests.append(
        Test(
            input_under_test="New question added during update",
            checks=[Check(operator="includes", criteria="important")],
        )
    )

    # Modify existing test
    suite.tests[0].input_under_test = "Updated question 1"
    suite.tests[0].checks.append(Check(operator="excludes", criteria="error"))

    # Add global checks
    suite.global_checks = [
        Check(operator="grammar", criteria=""),
        Check(operator="excludes", criteria="inappropriate"),
    ]

    # Update the suite
    await suite.update()
    print(f"Updated suite: {suite.title}")
    print(
        f"Now has {len(suite.tests)} tests and {len(suite.global_checks)} global checks"
    )
    return suite


async def pull_and_analyze_suite():
    """Pull an existing suite and analyze its contents."""
    print("\n=== Pulling and Analyzing Suite ===")

    # First create a suite to pull
    suite = Suite(
        title="Suite for Analysis",
        tests=[Test(input_under_test=f"Question {i}", checks=[]) for i in range(5)],
    )
    await suite.create()

    # Pull the suite
    if suite.id is None:
        raise ValueError("Suite ID is None")
    pulled_suite = await Suite.from_id(suite.id)

    print(f"Pulled suite: {pulled_suite.title}")
    print(f"ID: {pulled_suite.id}")
    print(f"Number of tests: {len(pulled_suite.tests)}")

    # Analyze test distribution
    if pulled_suite.tests:
        print("\nTest details:")
        for i, test in enumerate(pulled_suite.tests[:3]):  # Show first 3
            print(f"  Test {i + 1}: {test.input_under_test}")
            print(f"    Checks: {len(test.checks)}")

    return pulled_suite


async def pull_suite_with_files():
    """Pull a suite and download its files."""
    print("\n=== Pulling Suite with File Downloads ===")

    # Create a suite with files
    file_content = BytesIO(b"Content to be downloaded")
    file_content.name = "download_test.txt"

    suite = Suite(
        title="Suite with Downloadable Files",
        tests=[Test(input_under_test="Analyze this file", files_under_test=[file_content.name], checks=[])],
    )
    await suite.create()

    # Pull with file download
    os.makedirs("downloaded_files", exist_ok=True)
    if suite.id is None:
        raise ValueError("Suite ID is None")
    pulled_suite = await Suite.from_id(
        suite.id, download_files=True, download_path="downloaded_files"
    )

    print(f"Pulled suite: {pulled_suite.title}")

    # Check downloaded files
    if os.path.exists("downloaded_files"):
        files = os.listdir("downloaded_files")
        print(f"Downloaded {len(files)} file(s):")
        for file in files:
            print(f"  - {file}")

    return pulled_suite


async def load_and_save_json():
    """Load suite from JSON and save suite to JSON."""
    print("\n=== Loading and Saving JSON ===")

    # Load from JSON if file exists
    json_path = "suites/example_suite.json"
    if os.path.exists(json_path):
        loaded_suite = await Suite.from_json_file(json_path)
        print(f"Loaded suite from JSON: {loaded_suite.title}")

        # Create it on the platform
        await loaded_suite.create()
        print(f"Created suite: {loaded_suite.id}")
    else:
        # Create a new suite
        loaded_suite = Suite(
            title="Suite for JSON Export",
            description="This suite will be exported to JSON",
            tests=[
                Test(
                    input_under_test="Sample question",
                    checks=[Check(operator="grammar", criteria="")],
                )
            ],
        )
        await loaded_suite.create()

    # Save to JSON
    export_path = "exported_suite.json"
    loaded_suite.to_json_file(export_path)
    print(f"Exported suite to: {export_path}")

    # Show JSON structure
    with open(export_path, "r") as f:
        data = json.load(f)
        print(f"JSON structure has keys: {list(data.keys())}")

    return loaded_suite


async def clone_and_modify_suite():
    """Clone an existing suite with modifications."""
    print("\n=== Cloning and Modifying Suite ===")

    # Create original
    original = Suite(
        title="Original Suite",
        tests=[Test(input_under_test=f"Question {i}", checks=[]) for i in range(3)],
    )
    await original.create()
    print(f"Created original: {original.title}")

    # Pull to get fresh copy
    if original.id is None:
        raise ValueError("Original suite ID is None")
    cloned = await Suite.from_id(original.id)

    # Modify the clone
    cloned.title = "Cloned and Enhanced Suite"
    cloned.description = "This is a modified version of the original"
    cloned.id = None  # Clear ID to create as new suite

    # Add new test
    cloned.tests.append(
        Test(
            input_under_test="New question in cloned suite",
            checks=[Check(operator="grammar", criteria="")],
        )
    )

    # Create as new suite
    await cloned.create()
    print(f"Created clone: {cloned.title}")
    print(f"Original has {len(original.tests)} tests")
    print(f"Clone has {len(cloned.tests)} tests")

    return original, cloned


async def delete_suite_example():
    """Create and delete a suite."""
    print("\n=== Deleting Suite ===")

    # Create suite
    suite = Suite(
        title="Suite to Delete",
        tests=[Test(input_under_test="This suite will be deleted", checks=[])],
    )
    await suite.create()
    print(f"Created suite: {suite.id}")

    # Delete it
    await suite.delete()
    print(f"Deleted suite: {suite.id}")


async def list_all_suites():
    """List all test suites with filtering."""
    print("\n=== Listing All Suites ===")

    # List first 10 suites
    suites = await Suite.list_suites(limit=10)

    print(f"Found {len(suites)} suites:")
    for suite in suites[:5]:  # Show first 5
        print(f"  - {suite.title}")
        print(f"    Created: {suite.created}")
        if suite.description:
            print(f"    Description: {suite.description[:50]}...")


async def error_handling_example():
    """Demonstrate error handling best practices."""
    print("\n=== Error Handling Example ===")

    # Try to create suite with invalid data
    try:
        suite = Suite(
            title="",  # Empty title should cause error
            tests=[],
        )
        await suite.create()
    except Exception as e:
        print(f"Expected error for empty title: {type(e).__name__}: {e}")

    # Try to pull non-existent suite
    try:
        await Suite.from_id("non-existent-id-12345")
    except Exception as e:
        print(f"Expected error for non-existent suite: {type(e).__name__}: {e}")

    # Try to update deleted suite
    try:
        suite = Suite(title="To Delete", tests=[])
        await suite.create()
        await suite.delete()

        # This should fail
        suite.title = "Updated after delete"
        await suite.update()
    except Exception as e:
        print(f"Expected error for updating deleted suite: {type(e).__name__}: {e}")


async def main():
    """Run all suite examples."""
    print("=" * 50)
    print("SUITE EXAMPLES - Comprehensive Demonstrations")
    print("=" * 50)

    try:
        # Basic operations
        await create_simple_suite()
        await create_suite_with_global_checks()

        # Files and context
        await create_suite_with_files()
        await create_suite_with_context()

        # Tags and metadata
        await create_suite_with_tags_and_metadata()

        # CRUD operations
        await update_existing_suite()
        await pull_and_analyze_suite()
        await pull_suite_with_files()

        # Import/Export
        await load_and_save_json()
        await clone_and_modify_suite()

        # Cleanup
        await delete_suite_example()

        # List and error handling
        await list_all_suites()
        await error_handling_example()

        print("\n" + "=" * 50)
        print("All suite examples completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError in examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
