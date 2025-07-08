"""
Example demonstrating the use of OutputObject for returning output context.

This example shows how to use the new OutputObject class to return
structured output with metadata from model functions.
"""

import asyncio
from vals import Suite, Test, Check, configure_credentials, OutputObject


# Example 1: Simple model with OutputObject
def simple_model_with_context(input_text: str) -> OutputObject:
    """A simple model that returns output with context."""
    # Simulate some processing
    response = f"Processed: {input_text.upper()}"

    return OutputObject(
        llm_output=response,
        output_context={
            "original_length": len(input_text),
            "processed_length": len(response),
            "transformation": "uppercase",
        },
        in_tokens=len(input_text.split()),
        out_tokens=len(response.split()),
        duration=0.1,  # simulated duration
    )


# Example 2: RAG model with OutputObject
def rag_model(input_text: str) -> OutputObject:
    """A RAG model that includes retrieved documents in output context."""
    # Simulate document retrieval
    retrieved_docs = [
        {"id": "doc1", "text": "Paris is the capital of France.", "score": 0.95},
        {"id": "doc2", "text": "France is in Western Europe.", "score": 0.87},
    ]

    # Generate response based on retrieved docs
    response = "Based on the retrieved documents, Paris is the capital of France, which is located in Western Europe."

    return OutputObject(
        llm_output=response,
        output_context={
            "retrieved_documents": [doc["id"] for doc in retrieved_docs],
            "retrieval_scores": [doc["score"] for doc in retrieved_docs],
            "excerpts": [doc["text"] for doc in retrieved_docs],
            "retrieval_method": "semantic_search",
        },
        in_tokens=15,
        out_tokens=20,
        duration=0.5,
    )


# Example 3: Model with files and context using OutputObject
def model_with_files_and_context(
    input_text: str, files: dict, context: dict
) -> OutputObject:
    """A model that processes files and context, returning OutputObject."""
    # Process files
    file_info = {name: len(content.read()) for name, content in files.items()}

    # Generate response
    response = (
        f"Processed {len(files)} files with context: {context.get('task', 'unknown')}"
    )

    return OutputObject(
        llm_output=response,
        output_context={
            "files_processed": list(files.keys()),
            "file_sizes": file_info,
            "context_keys": list(context.keys()),
            "processing_status": "success",
        },
        in_tokens=50,
        out_tokens=30,
    )


# Example 4: Backward compatible - still works with dict return
def legacy_model_with_dict(input_text: str) -> dict:
    """Legacy model that returns a dict - still supported."""
    return {
        "llm_output": f"Legacy response: {input_text}",
        "output_context": {"legacy": True},
        "metadata": {"in_tokens": 10, "out_tokens": 5, "duration_seconds": 0.1},
    }


# Example 5: Backward compatible - still works with string return
def legacy_model_with_string(input_text: str) -> str:
    """Legacy model that returns a string - still supported."""
    return f"Simple string response: {input_text}"


async def main():
    # Configure credentials (assumes VALS_API_KEY is set)
    configure_credentials()

    # Create test suite
    suite = Suite(
        title="OutputObject Feature Demo",
        description="Demonstrates the new OutputObject feature for returning output context",
        tests=[
            Test(
                input_under_test="What is the capital of France?",
                checks=[
                    Check(operator="includes", criteria="Processed"),
                    Check(operator="includes", criteria="CAPITAL"),
                ],
            ),
            Test(
                input_under_test="Tell me about Paris",
                checks=[
                    Check(operator="includes", criteria="Paris"),
                    Check(operator="includes", criteria="France"),
                ],
            ),
        ],
    )

    # Create the suite
    print("Creating test suite...")
    await suite.create()
    print(f"Suite created: {suite.url}")

    # Run with different model types
    print("\n1. Running with OutputObject model...")
    run1 = await suite.run(
        model=simple_model_with_context, model_name="simple_output_object_model"
    )
    print(f"Run 1 completed: {run1.url}")

    print("\n2. Running with RAG OutputObject model...")
    run2 = await suite.run(model=rag_model, model_name="rag_output_object_model")
    print(f"Run 2 completed: {run2.url}")

    print("\n3. Running with legacy dict model (backward compatibility)...")
    run3 = await suite.run(model=legacy_model_with_dict, model_name="legacy_dict_model")
    print(f"Run 3 completed: {run3.url}")

    print("\n4. Running with legacy string model (backward compatibility)...")
    run4 = await suite.run(
        model=legacy_model_with_string, model_name="legacy_string_model"
    )
    print(f"Run 4 completed: {run4.url}")

    # Display some results with output context
    print("\n--- Output Context Examples ---")

    # Pull the RAG run to see output context
    await run2.pull()
    for test_result in run2.test_results[:2]:
        print(f"\nTest: {test_result.input_under_test}")
        print(f"Output: {test_result.llm_output}")
        if test_result.output_context:
            print(f"Output Context: {test_result.output_context}")

    print("\n✅ All examples completed successfully!")
    print("\nKey takeaways:")
    print("1. OutputObject provides type-safe output with metadata")
    print("2. output_context is now easily discoverable and documented")
    print("3. All legacy return types (string, dict) still work")
    print("4. IDE autocomplete and validation work with OutputObject")


if __name__ == "__main__":
    asyncio.run(main())
