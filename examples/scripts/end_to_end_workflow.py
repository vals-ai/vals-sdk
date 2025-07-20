"""
End-to-End Workflow Example

This comprehensive example demonstrates a complete workflow:
1. Create a test suite with various test types
2. Run the suite with different models
3. Monitor and analyze results
4. Export to CSV and JSON formats
5. Clean up resources

This example simulates a real-world testing scenario for a customer service chatbot.
"""

import asyncio
import csv
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any

from vals import Suite, Test, Check, Run, RunParameters, OutputObject


# Custom model for the chatbot
class CustomerServiceBot:
    """Simulated customer service chatbot model."""

    def __init__(self):
        self.knowledge_base = {
            "return_policy": "Our return policy allows returns within 30 days of purchase with receipt.",
            "shipping": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.",
            "hours": "Customer service is available Monday-Friday 9AM-5PM EST.",
            "contact": "You can reach us at support@example.com or 1-800-EXAMPLE.",
            "warranty": "All products come with a 1-year manufacturer warranty.",
            "payment": "We accept credit cards, debit cards, PayPal, and Apple Pay.",
        }

    def __call__(self, input: str, context: str | None = None) -> OutputObject:
        """Process customer inquiry."""
        input_lower = input.lower()
        response = ""
        topics_matched = []
        confidence = 0.5

        # Check knowledge base
        for topic, info in self.knowledge_base.items():
            if topic in input_lower or any(
                word in input_lower for word in topic.split("_")
            ):
                response = info
                topics_matched.append(topic)
                confidence = 0.9
                break

        # Use context if no direct match
        if not response and context:
            if "product" in input_lower and "warranty" in context.lower():
                response = "Based on the context, " + self.knowledge_base["warranty"]
                topics_matched.append("warranty_from_context")
                confidence = 0.7
            else:
                response = "I understand your question. Let me connect you with a human agent for better assistance."
                confidence = 0.3

        # Default response
        if not response:
            response = "I'm sorry, I don't have information about that. Please contact our support team."
            confidence = 0.1

        return OutputObject(
            llm_output=response,
            output_context={
                "topics_matched": topics_matched,
                "confidence": confidence,
                "used_context": bool(context and "context" in topics_matched),
                "response_type": "direct_match" if topics_matched else "default",
            },
        )


async def step1_create_test_suite():
    """Step 1: Create a comprehensive test suite for the chatbot."""
    print("\n" + "=" * 60)
    print("STEP 1: Creating Customer Service Chatbot Test Suite")
    print("=" * 60)

    # Create sample files for testing
    faq_doc = BytesIO(b"""
    Frequently Asked Questions:
    
    Q: What is your return policy?
    A: Returns are accepted within 30 days with receipt.
    
    Q: How long does shipping take?
    A: Standard: 5-7 days, Express: 2-3 days.
    
    Q: What payment methods do you accept?
    A: Credit cards, debit cards, PayPal, and Apple Pay.
    """)
    faq_doc.name = "faq.txt"

    product_info = BytesIO(b"""
    Product Information:
    - All electronics come with 1-year warranty
    - Extended warranty available for purchase
    - Free technical support included
    """)
    product_info.name = "product_info.txt"

    # Create test suite
    suite = Suite(
        title="Customer Service Chatbot Test Suite",
        description="Comprehensive tests for customer service chatbot responses",
        global_checks=[
            Check(operator="grammar", criteria=""),
            Check(operator="excludes", criteria="error"),
            Check(operator="excludes", criteria="undefined"),
            Check(operator="excludes", criteria="null"),
        ],
        tests=[
            # Simple FAQ tests
            Test(
                input_under_test="What is your return policy?",
                tags=["category:faq", "priority:high", "complexity:simple"],
                checks=[
                    Check(operator="includes", criteria="30 days"),
                    Check(operator="includes", criteria="receipt"),
                ],
            ),
            Test(
                input_under_test="How long does standard shipping take?",
                tags=["category:faq", "priority:high", "complexity:simple"],
                checks=[
                    Check(operator="includes", criteria="5-7"),
                    Check(operator="includes", criteria="business days"),
                ],
            ),
            Test(
                input_under_test="What are your customer service hours?",
                tags=["category:faq", "priority:medium", "complexity:simple"],
                checks=[
                    Check(operator="includes", criteria="Monday-Friday"),
                    Check(operator="includes", criteria="9AM-5PM"),
                ],
            ),
            # Context-based tests
            Test(
                input_under_test="How long is the warranty?",
                context={"customer_info": "Customer is asking about a laptop purchase"},
                tags=["category:warranty", "priority:high", "complexity:medium"],
                checks=[
                    Check(operator="includes", criteria="1-year"),
                    Check(operator="includes", criteria="warranty"),
                ],
            ),
            Test(
                input_under_test="Can I extend the warranty?",
                context={"customer_info": "Customer recently purchased electronics"},
                files_under_test=[product_info.name],
                tags=["category:warranty", "priority:medium", "complexity:complex"],
                checks=[
                    Check(
                        operator="includes_any", criteria="warranty|support|available"
                    )
                ],
            ),
            # Payment and contact tests
            Test(
                input_under_test="What payment methods do you accept?",
                tags=["category:payment", "priority:high", "complexity:simple"],
                checks=[
                    Check(operator="includes", criteria="credit"),
                    Check(operator="includes", criteria="PayPal"),
                    Check(operator="includes_any", criteria="Apple Pay|ApplePay"),
                ],
            ),
            Test(
                input_under_test="How can I contact customer service?",
                tags=["category:contact", "priority:high", "complexity:simple"],
                checks=[
                    Check(
                        operator="includes_any", criteria="support@example.com|1-800"
                    ),
                    Check(operator="excludes", criteria="unavailable"),
                ],
            ),
            # Edge cases
            Test(
                input_under_test="Do you ship to Mars?",
                tags=["category:edge_case", "priority:low", "complexity:simple"],
                checks=[
                    Check(operator="grammar", criteria=""),
                    Check(operator="excludes", criteria="yes"),
                ],
            ),
            Test(
                input_under_test="I want to return a product I bought 6 months ago",
                tags=["category:edge_case", "priority:medium", "complexity:medium"],
                checks=[
                    Check(operator="includes_any", criteria="30 days|policy|cannot")
                ],
            ),
            # File-based tests
            Test(
                input_under_test="What does the FAQ say about returns?",
                files_under_test=[faq_doc.name],
                tags=["category:document", "priority:medium", "complexity:complex"],
                checks=[
                    Check(operator="includes_any", criteria="30 days|return|receipt")
                ],
            ),
        ],
    )

    # Create the suite
    await suite.create()
    print(f"\nCreated test suite: {suite.title}")
    print(f"Suite ID: {suite.id}")
    print(f"Total tests: {len(suite.tests)}")
    print(f"Global checks: {len(suite.global_checks)}")

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


async def step2_run_with_multiple_configurations(suite: Suite):
    """Step 2: Run the suite with different models and configurations."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Tests with Multiple Configurations")
    print("=" * 60)

    runs = []

    # Configuration 1: GPT-4 (baseline)
    print("\n1. Running with GPT-4o-mini (baseline)...")
    params1 = RunParameters(temperature=0.3, max_output_tokens=150)
    run1 = await suite.run(model="gpt-4o-mini", parameters=params1)
    runs.append(("GPT-4o-mini", run1))
    print(f"   Started run: {run1.id}")

    # Configuration 2: Custom chatbot
    print("\n2. Running with custom chatbot model...")
    chatbot = CustomerServiceBot()
    run2 = await suite.run(model=chatbot, model_name="customer-service-bot-v1")
    runs.append(("Custom Bot", run2))
    print(f"   Started run: {run2.id}")

    # Configuration 3: GPT-4 with higher temperature for edge cases
    print("\n3. Running with GPT-4o (creative mode for edge cases)...")
    params3 = RunParameters(temperature=0.7, max_output_tokens=200)
    run3 = await suite.run(model="gpt-4o", parameters=params3)
    runs.append(("GPT-4o Creative", run3))
    print(f"   Started run: {run3.id}")

    # Monitor progress
    print("\n4. Monitoring run progress...")
    for name, run in runs:
        print(f"\n   Monitoring {name}...")
        while run.status not in ["COMPLETED", "FAILED", "CANCELLED"]:
            await asyncio.sleep(3)
            run = await Run.from_id(run.id)
            completed = sum(1 for r in run.test_results if r.llm_output is not None)
            total = len(run.test_results)
            print(f"   {name}: {completed}/{total} tests completed", end="\r")
        print(f"   {name}: Completed with {run.pass_percentage:.1f}% pass rate")

    return runs


async def step3_analyze_results(runs: list[tuple[str, Run]]):
    """Step 3: Analyze and compare results."""
    print("\n" + "=" * 60)
    print("STEP 3: Analyzing Test Results")
    print("=" * 60)

    analysis = {
        "model_comparison": [],
        "category_performance": {},
        "failed_tests_by_model": {},
        "performance_metrics": {},
    }

    for name, run in runs:
        # Pull detailed results
        detailed = await Run.from_id(run.id)

        # Overall metrics
        model_metrics = {
            "model": name,
            "pass_rate": detailed.pass_percentage,
            "duration": 0,  # Duration not available in current API
            "avg_response_time": 0,
        }
        analysis["model_comparison"].append(model_metrics)

        # Category performance by tags
        category_pass_counts = {}
        category_total_counts = {}
        
        for result in detailed.test_results:
            for tag in result.test.tags:
                if tag.startswith("category:"):
                    category = tag.split(":", 1)[1]
                    if category not in category_pass_counts:
                        category_pass_counts[category] = 0
                        category_total_counts[category] = 0
                    
                    category_total_counts[category] += 1
                    if result.pass_percentage == 100:
                        category_pass_counts[category] += 1
        
        # Calculate pass rates per category
        for category in category_total_counts:
            if category not in analysis["category_performance"]:
                analysis["category_performance"][category] = {}
            pass_rate = (category_pass_counts[category] / category_total_counts[category]) * 100
            analysis["category_performance"][category][name] = pass_rate

        # Failed tests
        failed_tests = []
        for result in detailed.test_results:
            if result.pass_percentage < 100:
                category = "unknown"
                for tag in result.test.tags:
                    if tag.startswith("category:"):
                        category = tag.split(":", 1)[1]
                        break
                
                failed_tests.append(
                    {
                        "input": result.test.input_under_test,
                        "category": category,
                        "expected_checks": [
                            f"{c.operator}: {c.criteria}" for c in result.test.checks
                        ],
                        "output": result.llm_output[:100] + "..."
                        if result.llm_output
                        else "No output",
                    }
                )
        analysis["failed_tests_by_model"][name] = failed_tests

        # Performance metrics from output context
        context_metrics = []
        for result in detailed.test_results:
            if result.output_context:
                context_metrics.append(result.output_context)
        analysis["performance_metrics"][name] = context_metrics

    # Print analysis summary
    print("\n1. Model Comparison:")
    print(f"{'Model':<20} {'Pass Rate':<12} {'Duration':<12} {'Avg Response':<12}")
    print("-" * 56)
    for metrics in analysis["model_comparison"]:
        print(
            f"{metrics['model']:<20} {metrics['pass_rate']:<12.1f}% {metrics['duration']:<12.2f}s {metrics['avg_response_time']:<12.3f}s"
        )

    print("\n2. Performance by Category:")
    for category, model_scores in analysis["category_performance"].items():
        print(f"\n   {category}:")
        for model, score in model_scores.items():
            print(f"     - {model}: {score:.1f}%")

    print("\n3. Failed Tests Summary:")
    for model, failed in analysis["failed_tests_by_model"].items():
        print(f"\n   {model}: {len(failed)} failed tests")
        if failed:
            # Show first 2 failures
            for test in failed[:2]:
                print(f"     - Category: {test['category']}")
                print(f"       Input: {test['input'][:50]}...")
                print(f"       Output: {test['output'][:80]}...")

    # Identify best model for each category
    print("\n4. Best Model by Category:")
    for category, scores in analysis["category_performance"].items():
        best_model = max(scores.items(), key=lambda x: x[1])
        print(f"   - {category}: {best_model[0]} ({best_model[1]:.1f}%)")

    return analysis


async def step4_export_results(
    suite: Suite, runs: list[tuple[str, Run]], analysis: dict[str, Any]
):
    """Step 4: Export results to various formats."""
    print("\n" + "=" * 60)
    print("STEP 4: Exporting Results")
    print("=" * 60)

    # Create exports directory
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Export detailed test results to CSV
    csv_filename = f"exports/chatbot_test_results_{timestamp}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "model",
            "test_input",
            "category",
            "priority",
            "complexity",
            "passed",
            "pass_rate",
            "checks_passed",
            "total_checks",
            "model_output",
            "confidence",
            "response_type",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for name, run in runs:
            detailed = await Run.from_id(run.id)
            for result in detailed.test_results:
                # Extract metadata from tags
                category = priority = complexity = ""
                for tag in result.test.tags:
                    if tag.startswith("category:"):
                        category = tag.split(":", 1)[1]
                    elif tag.startswith("priority:"):
                        priority = tag.split(":", 1)[1]
                    elif tag.startswith("complexity:"):
                        complexity = tag.split(":", 1)[1]
                
                output_context = result.output_context or {}
                
                # Count passed checks
                passed_checks = sum(1 for cr in result.check_results if cr.auto_eval == 1)
                total_checks = len(result.check_results)

                writer.writerow(
                    {
                        "model": name,
                        "test_input": result.test.input_under_test,
                        "category": category,
                        "priority": priority,
                        "complexity": complexity,
                        "passed": result.pass_percentage == 100,
                        "pass_rate": f"{passed_checks}/{total_checks}",
                        "checks_passed": passed_checks,
                        "total_checks": total_checks,
                        "model_output": result.llm_output or "",
                        "confidence": output_context.get("confidence", ""),
                        "response_type": output_context.get("response_type", ""),
                    }
                )

    print(f"\n1. Exported detailed results to: {csv_filename}")

    # 2. Export analysis summary to JSON
    json_filename = f"exports/chatbot_analysis_{timestamp}.json"
    export_data = {
        "test_suite": {
            "id": suite.id,
            "name": suite.title,
            "total_tests": len(suite.tests),
            "categories": list(analysis["category_performance"].keys()),
        },
        "execution_summary": {
            "timestamp": timestamp,
            "models_tested": len(runs),
            "total_runs": len(runs),
        },
        "results": analysis,
        "recommendations": generate_recommendations(analysis),
    }

    with open(json_filename, "w") as jsonfile:
        json.dump(export_data, jsonfile, indent=2)

    print(f"2. Exported analysis to: {json_filename}")

    # 3. Export suite definition
    suite_filename = f"exports/chatbot_suite_{timestamp}.json"
    suite.to_json_file(suite_filename)
    print(f"3. Exported suite definition to: {suite_filename}")

    # 4. Generate summary report
    report_filename = f"exports/chatbot_test_report_{timestamp}.md"
    with open(report_filename, "w") as report:
        report.write("# Customer Service Chatbot Test Report\n\n")
        report.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.write(f"**Test Suite**: {suite.title}\n\n")
        report.write(f"**Total Tests**: {len(suite.tests)}\n\n")

        report.write("## Executive Summary\n\n")
        best_model = max(analysis["model_comparison"], key=lambda x: x["pass_rate"])
        report.write(
            f"- **Best Overall Model**: {best_model['model']} ({best_model['pass_rate']:.1f}% pass rate)\n"
        )
        report.write(f"- **Models Tested**: {len(runs)}\n")
        report.write(
            f"- **Categories Covered**: {len(analysis['category_performance'])}\n\n"
        )

        report.write("## Model Performance\n\n")
        report.write("| Model | Pass Rate | Duration | Avg Response Time |\n")
        report.write("|-------|-----------|----------|------------------|\n")
        for metrics in analysis["model_comparison"]:
            report.write(
                f"| {metrics['model']} | {metrics['pass_rate']:.1f}% | {metrics['duration']:.2f}s | {metrics['avg_response_time']:.3f}s |\n"
            )

        report.write("\n## Recommendations\n\n")
        for rec in export_data["recommendations"]:
            report.write(f"- {rec}\n")

    print(f"4. Generated test report: {report_filename}")

    return {
        "csv": csv_filename,
        "json": json_filename,
        "suite": suite_filename,
        "report": report_filename,
    }


def generate_recommendations(analysis: dict[str, Any]) -> list[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Overall performance
    best_model = max(analysis["model_comparison"], key=lambda x: x["pass_rate"])
    if best_model["pass_rate"] < 80:
        recommendations.append(
            "Overall pass rates are below 80%. Consider improving model training or adjusting test criteria."
        )

    # Category-specific recommendations
    for category, scores in analysis["category_performance"].items():
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        if avg_score < 70:
            recommendations.append(
                f"'{category}' category has low performance ({avg_score:.1f}%). Focus on improving responses for this area."
            )

    # Model-specific recommendations
    for model, failed in analysis["failed_tests_by_model"].items():
        if len(failed) > 3:
            categories = [test["category"] for test in failed]
            most_failed = max(set(categories), key=categories.count)
            recommendations.append(
                f"{model} struggles with '{most_failed}' category tests. Consider specialized training."
            )

    # Speed recommendations
    slowest = max(analysis["model_comparison"], key=lambda x: x["avg_response_time"])
    if slowest["avg_response_time"] > 1.0:
        recommendations.append(
            f"{slowest['model']} has slow response times ({slowest['avg_response_time']:.2f}s). Consider optimization."
        )

    return recommendations


async def step5_cleanup(suite: Suite, runs: list[tuple], export_files: dict[str, str]):
    """Step 5: Optional cleanup of resources."""
    print("\n" + "=" * 60)
    print("STEP 5: Cleanup")
    print("=" * 60)

    print("\nWorkflow completed successfully!")
    print("\nCreated resources:")
    print(f"  - Test Suite: {suite.id}")
    for name, run in runs:
        print(f"  - Run ({name}): {run.id}")

    print("\nExported files:")
    for file_type, filename in export_files.items():
        print(f"  - {file_type.capitalize()}: {filename}")

    # Ask user about cleanup
    user_input = (
        input("\nDo you want to delete the test suite? (y/N): ").strip().lower()
    )
    if user_input == "y":
        await suite.delete()
        print("Test suite deleted.")
    else:
        print("Test suite preserved for future use.")
        print(f"Suite URL: {suite.url}")


async def main():
    """Run the complete end-to-end workflow."""
    print("=" * 60)
    print("END-TO-END WORKFLOW: Customer Service Chatbot Testing")
    print("=" * 60)
    print("\nThis workflow demonstrates:")
    print("1. Creating a comprehensive test suite")
    print("2. Running tests with multiple models")
    print("3. Analyzing and comparing results")
    print("4. Exporting data in multiple formats")
    print("5. Resource cleanup")

    try:
        # Execute workflow steps
        suite = await step1_create_test_suite()
        runs = await step2_run_with_multiple_configurations(suite)
        analysis = await step3_analyze_results(runs)
        export_files = await step4_export_results(suite, runs, analysis)
        await step5_cleanup(suite, runs, export_files)

        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during workflow: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
