"""
Run Examples - Comprehensive demonstrations of Run class functionality.

This file shows all operations you can perform with the Run class:
- Running suites with stock models (GPT-4, Claude, etc.)
- Running with custom Python functions
- Using custom operators
- Monitoring run progress
- Analyzing results
- Exporting to CSV
- Error handling

Note: File paths assume this is run from the `examples/` directory.
"""

import asyncio
import csv
from io import BytesIO
from typing import Any, List

from vals import Run, Suite, Test, Check, RunParameters, OutputObject


# Custom operator example
class LengthCheckOperator:
    """Custom operator that checks if output length meets criteria."""

    async def evaluate(self, output: str, criteria: str) -> dict:
        """Check if output length matches criteria (e.g., '>100', '<50', '=20')."""
        if not criteria:
            return {"passed": False, "details": "No length criteria specified"}

        # Parse criteria
        if criteria.startswith(">="):
            required_length = int(criteria[2:])
            passed = len(output) >= required_length
        elif criteria.startswith("<="):
            required_length = int(criteria[2:])
            passed = len(output) <= required_length
        elif criteria.startswith(">"):
            required_length = int(criteria[1:])
            passed = len(output) > required_length
        elif criteria.startswith("<"):
            required_length = int(criteria[1:])
            passed = len(output) < required_length
        elif criteria.startswith("="):
            required_length = int(criteria[1:])
            passed = len(output) == required_length
        else:
            required_length = int(criteria)
            passed = len(output) == required_length

        details = f"Output length: {len(output)}, Required: {criteria}"
        return {"passed": passed, "details": details}


# Simple custom model
def simple_custom_model(input: str) -> str:
    """A simple model that echoes input with modifications."""
    return f"You asked: '{input}'. Here's my response based on that question."


# Advanced custom model with context and files
def advanced_custom_model(
    input: str, context: dict = None, files: dict = None
) -> OutputObject:
    """Advanced model that uses context and files, returning OutputObject."""
    response_parts = [f"Processing: {input}"]
    metadata = {
        "input_length": len(input),
        "has_context": bool(context),
        "file_count": 0,
    }

    if context:
        response_parts.append(f"Context provided: {context[:50]}...")
        metadata["context_length"] = len(context)

    if files:
        metadata["file_count"] = len(files)
        file_info = []
        for file in files:
            if hasattr(file, "name"):
                file_info.append(file.name)
            if hasattr(file, "read"):
                content = file.read()
                if hasattr(file, "seek"):
                    file.seek(0)  # Reset file pointer
                metadata[f"file_size_{file.name}"] = len(content)

        response_parts.append(f"Files provided: {', '.join(file_info)}")

    response = " ".join(response_parts)

    return OutputObject(llm_output=response, output_context=metadata)


# Async custom model
async def async_custom_model(input: str) -> str:
    """Async model that simulates processing delay."""
    await asyncio.sleep(0.5)  # Simulate processing time
    return f"Async processed: {input}"


# Model for QA pairs
def qa_pair_model(input: str) -> str:
    """Model that returns predefined answers for specific questions."""
    qa_map = {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4",
        "Who wrote Romeo and Juliet?": "William Shakespeare",
    }
    return qa_map.get(input, f"No predefined answer for: {input}")


async def run_with_stock_model():
    """Run a test suite with a stock model."""
    print("\n=== Running with Stock Model (GPT-4o-mini) ===")

    suite = Suite(
        title="Stock Model Test Suite",
        tests=[
            Test(
                input_under_test="What is the capital of Japan?",
                checks=[Check(operator="equals", criteria="Tokyo")],
            ),
            Test(
                input_under_test="Translate 'Hello' to Spanish",
                checks=[Check(operator="includes", criteria="Hola")],
            ),
            Test(
                input_under_test="What is 15 * 8?",
                checks=[Check(operator="equals", criteria="120")],
            ),
        ],
    )
    await suite.create()

    # Run with GPT-4o-mini
    run = await suite.run(model="gpt-4o-mini")
    print(f"Started run: {run.id}")
    print(f"Run URL: {run.url}")

    # Wait for completion
    await run.wait_for_completion(poll_interval=2)

    print("Run completed!")
    print(f"Pass rate: {run.pass_percentage:.1f}%")

    return run


async def run_with_custom_parameters():
    """Run with custom parameters like temperature and max tokens."""
    print("\n=== Running with Custom Parameters ===")

    suite = Suite(
        title="Creative Writing Test",
        tests=[
            Test(
                input_under_test="Write a haiku about coding",
                checks=[
                    Check(operator="grammar", criteria=""),
                    Check(
                        operator="includes_any", criteria="code|program|software|debug"
                    ),
                ],
            ),
            Test(
                input_under_test="Create a metaphor for machine learning",
                checks=[
                    Check(operator="includes", criteria="like"),
                    Check(operator="excludes", criteria="literally"),
                ],
            ),
        ],
    )
    await suite.create()

    # Custom parameters for creative tasks
    params = RunParameters(
        temperature=0.9,  # High temperature for creativity
        max_output_tokens=200,
        parallelism=2,
    )

    run = await suite.run(model="gpt-4o", parameters=params)
    print(f"Started creative run: {run.id}")
    print(f"Parameters: temp={params.temperature}, max_tokens={params.max_output_tokens}")

    await run.wait_for_completion()
    print(f"Completed with {run.pass_percentage:.1f}% pass rate")

    return run


async def run_with_custom_function():
    """Run with a custom Python function as the model."""
    print("\n=== Running with Custom Function ===")

    suite = Suite(
        title="Custom Function Test",
        tests=[
            Test(
                input_under_test="Test input 1",
                checks=[Check(operator="includes", criteria="You asked")],
            ),
            Test(
                input_under_test="Another test",
                checks=[Check(operator="includes", criteria="response")],
            ),
        ],
    )
    await suite.create()

    # Run with custom function
    run = await suite.run(
        model=simple_custom_model, model_name="simple-echo-model"
    )
    print(f"Started run with custom model: {run.id}")

    await run.wait_for_completion()
    print(f"Custom model pass rate: {run.pass_percentage:.1f}%")

    return run


async def run_with_context_and_files():
    """Run with a model that uses context and files."""
    print("\n=== Running with Context and Files ===")

    # Create test file
    test_file = BytesIO(b"This is test file content for the model to analyze.")
    test_file.name = "test_document.txt"

    suite = Suite(
        title="Context and Files Test",
        tests=[
            Test(
                input_under_test="What information is provided in the context?",
                context={"background": "The company was founded in 2020 and has 50 employees."},
                checks=[
                    Check(operator="includes", criteria="2020"),
                    Check(operator="includes", criteria="50"),
                ],
            ),
            Test(
                input_under_test="Analyze the uploaded file",
                files_under_test=[test_file.name],
                checks=[Check(operator="includes", criteria="test_document.txt")],
            ),
        ],
    )
    await suite.create()

    # Run with advanced model
    run = await suite.run(
        model=advanced_custom_model, model_name="context-aware-model"
    )
    print(f"Started context-aware run: {run.id}")

    await run.wait_for_completion()
    print(f"Completed with {run.pass_percentage:.1f}% pass rate")

    return run


async def run_with_custom_operators():
    """Run with custom operators for specialized checks."""
    print("\n=== Running with Custom Operators ===")

    suite = Suite(
        title="Custom Operator Test",
        tests=[
            Test(
                input_under_test="Generate a short response",
                checks=[
                    Check(operator="custom.length_check", criteria="<100"),
                    Check(operator="grammar", criteria=""),
                ],
            ),
            Test(
                input_under_test="Generate a detailed explanation of photosynthesis",
                checks=[
                    Check(operator="custom.length_check", criteria=">200"),
                    Check(operator="includes", criteria="plants"),
                ],
            ),
        ],
    )
    await suite.create()

    # Register custom operator
    custom_operators = {"custom.length_check": LengthCheckOperator()}

    # Run with custom operators
    run = await suite.run(
        model="gpt-4o-mini", custom_operators=custom_operators
    )
    print(f"Started run with custom operators: {run.id}")

    await run.wait_for_completion()
    print(f"Custom operator run completed: {run.pass_percentage:.1f}%")

    return run


async def run_with_local_evaluation():
    """Run with local evaluation instead of server-side."""
    print("\n=== Running with Local Evaluation ===")

    suite = Suite(
        title="Local Evaluation Test",
        tests=[
            Test(
                input_under_test="What is the meaning of life?",
                checks=[
                    Check(operator="custom.length_check", criteria=">10"),
                    Check(operator="excludes", criteria="error"),
                ],
            )
        ],
    )
    await suite.create()

    # Custom operators for local eval
    custom_operators = {"custom.length_check": LengthCheckOperator()}

    # Run with local evaluation
    run = await suite.run(
        model="gpt-4o-mini", custom_operators=custom_operators, use_server_eval=False
    )
    print(f"Started local evaluation run: {run.id}")

    await run.wait_for_completion()
    print(f"Local eval completed: {run.pass_percentage:.1f}%")

    return run


async def run_with_qa_pairs():
    """Run using pre-computed question-answer pairs."""
    print("\n=== Running with QA Pairs ===")

    # Define QA pairs
    qa_pairs = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    ]

    suite = Suite(
        title="QA Pairs Test",
        tests=[
            Test(
                input_under_test=qa["question"],
                checks=[Check(operator="equals", criteria=qa["answer"])],
            )
            for qa in qa_pairs
        ],
    )
    await suite.create()

    # Run with QA model
    run = await suite.run(
        model=qa_pair_model, model_name="qa-pairs-model"
    )
    print(f"Started QA pairs run: {run.id}")

    await run.wait_for_completion()
    print(f"QA pairs run completed: {run.pass_percentage:.1f}%")

    return run


async def monitor_run_progress():
    """Monitor the progress of a running test suite."""
    print("\n=== Monitoring Run Progress ===")

    # Create larger suite to see progress
    tests = []
    for i in range(10):
        tests.append(
            Test(
                input_under_test=f"Calculate {i} squared",
                checks=[Check(operator="includes", criteria=str(i * i))],
            )
        )

    suite = Suite(title="Progress Monitoring Test", tests=tests)
    await suite.create()

    # Start run
    run = await suite.run(model="gpt-4o-mini")
    print(f"Started run with {len(tests)} tests: {run.id}")

    # Monitor progress
    completed_count = 0
    while run.status not in ["COMPLETED", "FAILED", "CANCELLED"]:
        await asyncio.sleep(2)
        run = await Run.from_id(run.id)

        # Count completed tests
        new_completed = sum(1 for r in run.test_results if r.llm_output is not None)
        if new_completed > completed_count:
            completed_count = new_completed
            print(f"Progress: {completed_count}/{len(tests)} tests completed")

    print("\nRun finished!")
    print(f"Final status: {run.status}")
    print(f"Pass rate: {run.pass_percentage:.1f}%")

    return run


async def analyze_run_results():
    """Pull and analyze detailed run results."""
    print("\n=== Analyzing Run Results ===")

    # Create test suite with tags
    suite = Suite(
        title="Detailed Analysis Test",
        tests=[
            Test(
                input_under_test="What is the capital of France?",
                tags=["category:geography", "difficulty:easy"],
                checks=[Check(operator="equals", criteria="Paris")],
            ),
            Test(
                input_under_test="Explain quantum entanglement",
                tags=["category:physics", "difficulty:hard"],
                checks=[
                    Check(operator="includes", criteria="quantum"),
                    Check(operator="includes", criteria="particles"),
                ],
            ),
            Test(
                input_under_test="What is 25 * 4?",
                tags=["category:math", "difficulty:easy"],
                checks=[Check(operator="equals", criteria="100")],
            ),
        ],
    )
    await suite.create()

    # Run the suite
    run = await suite.run(model="gpt-4o-mini")
    await run.wait_for_completion()

    # Pull detailed results
    detailed_run = await Run.from_id(run.id)

    print(f"\nDetailed Analysis for Run {detailed_run.id}")
    print(f"Overall pass rate: {detailed_run.pass_percentage:.1f}%")

    # Analyze by test
    print("\nTest Results:")
    for i, result in enumerate(detailed_run.test_results):
        print(f"\nTest {i + 1}: {result.test.input_under_test[:50]}...")
        print(f"  Passed: {result.pass_percentage == 100} ({result.pass_percentage:.0f}%)")

        if result.llm_output:
            print(f"  Output: {result.llm_output[:100]}...")

        # Show failed checks
        for check_result in result.check_results:
            if check_result.auto_eval != 1:
                print(
                    f"  Failed check: {check_result.operator} - {check_result.criteria}"
                )

    # Analyze by tags
    tag_performance = {}
    for result in detailed_run.test_results:
        for tag in result.test.tags:
            if tag.startswith("category:"):
                category = tag.split(":", 1)[1]
                if category not in tag_performance:
                    tag_performance[category] = {"total": 0, "passed": 0}
                tag_performance[category]["total"] += 1
                if result.pass_percentage == 100:
                    tag_performance[category]["passed"] += 1

    print("\nPerformance by Category:")
    for category, stats in tag_performance.items():
        pass_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {category}: {pass_rate:.1f}% pass rate")

    return detailed_run


async def export_run_to_csv():
    """Export run results to CSV format."""
    print("\n=== Exporting Run to CSV ===")

    # Create and run a suite
    suite = Suite(
        title="CSV Export Test",
        tests=[
            Test(input_under_test=f"Question {i}", checks=[Check(operator="grammar", criteria="")])
            for i in range(5)
        ],
    )
    await suite.create()
    run = await suite.run(model="gpt-4o-mini")
    await run.wait_for_completion()

    # Pull detailed results
    detailed_run = await Run.from_id(run.id)

    # Export to CSV
    csv_filename = "run_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "test_input",
            "model_output",
            "passed",
            "pass_percentage",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in detailed_run.test_results:
            writer.writerow(
                {
                    "test_input": result.test.input_under_test,
                    "model_output": result.llm_output or "",
                    "passed": result.pass_percentage == 100,
                    "pass_percentage": result.pass_percentage,
                }
            )

    print(f"Exported results to {csv_filename}")
    print(f"Exported {len(detailed_run.test_results)} test results")

    return detailed_run


async def compare_model_performance():
    """Compare performance of different models."""
    print("\n=== Comparing Model Performance ===")

    # Create benchmark suite
    suite = Suite(
        title="Model Benchmark Suite",
        tests=[
            Test(
                input_under_test="What is the capital of Japan?",
                checks=[Check(operator="equals", criteria="Tokyo")],
            ),
            Test(
                input_under_test="What is 15 * 8?",
                checks=[Check(operator="equals", criteria="120")],
            ),
            Test(
                input_under_test="Translate 'Good morning' to French",
                checks=[Check(operator="includes_any", criteria="Bonjour|bonjour")],
            ),
        ],
    )
    await suite.create()

    # Test different models
    models = ["gpt-4o-mini", "gpt-4o"]
    results = {}

    for model in models:
        print(f"\nTesting {model}...")
        try:
            run = await suite.run(model=model)
            await run.wait_for_completion()

            results[model] = {
                "pass_rate": run.pass_percentage,
                "run_id": run.id,
            }
            print(f"  Pass rate: {run.pass_percentage:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
            results[model] = {"error": str(e)}

    # Summary
    print("\nModel Comparison Summary:")
    for model, result in results.items():
        if "error" not in result:
            print(f"  {model}: {result['pass_rate']:.1f}%")
        else:
            print(f"  {model}: Failed - {result['error']}")

    return results


async def run_with_retry_logic():
    """Demonstrate retry logic for handling failures."""
    print("\n=== Running with Retry Logic ===")

    suite = Suite(
        title="Retry Logic Test",
        tests=[
            Test(input_under_test="What is 2+2?", checks=[Check(operator="equals", criteria="4")])
        ],
    )
    await suite.create()

    max_retries = 3
    retry_count = 0
    successful_run = None

    while retry_count < max_retries and not successful_run:
        try:
            print(f"\nAttempt {retry_count + 1} of {max_retries}")
            run = await suite.run(model="gpt-4o-mini")
            await run.wait_for_completion()

            if run.pass_percentage == 100:
                print("Run succeeded with 100% pass rate!")
                successful_run = run
            else:
                print(f"Run completed with {run.pass_percentage:.1f}% pass rate")
                retry_count += 1
                if retry_count < max_retries:
                    print("Retrying...")
                    await asyncio.sleep(2)  # Wait before retry
        except Exception as e:
            print(f"Run failed with error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(2)

    if successful_run:
        print(f"\nSuccessful run achieved: {successful_run.id}")
    else:
        print(f"\nNo fully successful run after {max_retries} attempts")

    return successful_run


async def list_recent_runs():
    """List and analyze recent runs."""
    print("\n=== Listing Recent Runs ===")

    # List last 5 runs
    runs = await Run.list_runs(limit=5)

    if not runs:
        print("No runs found")
        return []

    print(f"Found {len(runs)} recent runs:")
    for run in runs:
        print(f"\nRun ID: {run.id}")
        print(f"  Suite: {run.test_suite_title}")
        print(f"  Status: {run.status}")
        print(f"  Model: {run.model}")
        print(f"  Pass Rate: {run.pass_percentage:.1f}%")
        print(f"  Created: {run.timestamp}")

    return runs


async def pull_run(run_id: str = None):
    """Pull and display a run by ID. If no ID provided, uses the most recent run."""
    if run_id:
        run = await Run.from_id(run_id)
    else:
        # Use the most recent run if no ID provided
        runs = await Run.list_runs(limit=1)
        if not runs:
            print("No runs found")
            return
        run = await Run.from_id(runs[0].id)
        print(f"Using most recent run: {run.id}")

    print(run.model_dump())


async def error_handling_example():
    """Demonstrate error handling in runs."""
    print("\n=== Error Handling Example ===")

    # Try to run non-existent suite
    try:
        fake_suite = Suite(title="Fake", tests=[])
        fake_suite.id = "non-existent-suite-id"
        await fake_suite.run(model="gpt-4o-mini")
    except Exception as e:
        print(f"Expected error for non-existent suite: {type(e).__name__}: {e}")

    # Try to run with invalid model
    try:
        suite = Suite(title="Invalid Model Test", tests=[Test(input_under_test="Test", checks=[])])
        await suite.create()
        await suite.run(model="non-existent-model-xyz")
    except Exception as e:
        print(f"Expected error for invalid model: {type(e).__name__}: {e}")

    # Handle timeout
    print("\nHandling potential timeout...")
    suite = Suite(title="Timeout Test", tests=[Test(input_under_test="Test", checks=[])])
    await suite.create()
    run = await suite.run(model="gpt-4o-mini")

    # Set a short timeout for demonstration
    try:
        await asyncio.wait_for(
            run.wait_for_completion(poll_interval=1),
            timeout=0.1,  # Very short timeout
        )
    except asyncio.TimeoutError:
        print("Run timed out (expected for demo)")
        # In real scenario, you might want to cancel or handle differently

    # Wait for actual completion
    await run.wait_for_completion()
    print(f"Run eventually completed: {run.status}")


async def main():
    """Run all examples."""
    print("=" * 50)
    print("RUN EXAMPLES - Comprehensive Demonstrations")
    print("=" * 50)

    try:
        # Basic runs
        await run_with_stock_model()
        await run_with_custom_parameters()

        # Custom models
        await run_with_custom_function()
        await run_with_context_and_files()
        await run_with_qa_pairs()

        # Advanced features
        await run_with_custom_operators()
        await run_with_local_evaluation()

        # Monitoring and analysis
        await monitor_run_progress()
        await analyze_run_results()

        # Export and comparison
        await export_run_to_csv()
        await compare_model_performance()

        # Error handling and retries
        await run_with_retry_logic()
        await error_handling_example()

        # List recent runs
        await list_recent_runs()
        await pull_run()  # Will use the most recent run

        print("\n" + "=" * 50)
        print("All run examples completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError in examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
