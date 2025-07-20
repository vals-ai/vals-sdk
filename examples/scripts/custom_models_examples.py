"""
Custom Models Examples - Comprehensive demonstrations of custom model implementations.

This file shows various ways to implement custom models:
- Simple synchronous models
- Async models
- Models with context and files
- Models using OutputObject for metadata
- RAG (Retrieval Augmented Generation) models
- Domain-specific models
- Stateful models with caching
- Error handling in models

Note: These models can be used with Suite.run(custom_model=your_model)
"""

import asyncio
import json
import random
from typing import Any
import time

from vals import Suite, Test, Check, OutputObject


# ====================
# SIMPLE MODELS
# ====================


def echo_model(input: str) -> str:
    """Simplest possible model - just echoes the input."""
    return f"Echo: {input}"


def uppercase_model(input: str) -> str:
    """Simple transformation model."""
    return input.upper()


def word_count_model(input: str) -> str:
    """Model that counts words in the input."""
    word_count = len(input.split())
    return f"The input contains {word_count} words."


# ====================
# MATHEMATICAL MODELS
# ====================


def calculator_model(input: str) -> str:
    """Simple calculator that evaluates basic math expressions."""
    try:
        # Extract mathematical expression
        # Warning: eval() is dangerous in production - use ast.literal_eval or a proper parser
        expression = input.lower().replace("what is", "").replace("?", "").strip()

        # Basic safety check (only allow numbers and basic operators)
        allowed_chars = "0123456789+-*/() ."
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        else:
            return "I can only evaluate basic mathematical expressions."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def statistics_model(input: str) -> OutputObject:
    """Model that calculates statistics from numbers in the input."""
    import re

    # Extract all numbers from input
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", input)]

    if not numbers:
        return OutputObject(
            llm_output="No numbers found in the input.",
            output_context={"error": "no_numbers"},
        )

    # Calculate statistics
    stats = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "numbers": numbers,
    }

    output = f"Found {stats['count']} numbers. Sum: {stats['sum']}, Mean: {stats['mean']:.2f}, Min: {stats['min']}, Max: {stats['max']}"

    return OutputObject(llm_output=output, output_context=stats)


# ====================
# ASYNC MODELS
# ====================


async def async_delay_model(input: str) -> str:
    """Async model that simulates processing delay."""
    processing_time = random.uniform(0.5, 2.0)
    await asyncio.sleep(processing_time)
    return f"Processed '{input}' in {processing_time:.2f} seconds"


async def async_api_model(input: str) -> OutputObject:
    """Async model that simulates API calls."""
    # Simulate multiple API calls
    api_results = []

    for i in range(3):
        await asyncio.sleep(0.3)  # Simulate API latency
        api_results.append(
            {
                "api": f"service_{i}",
                "response": f"Data for '{input}' from service {i}",
                "latency_ms": random.randint(100, 500),
            }
        )

    # Aggregate results
    total_latency = sum(r["latency_ms"] for r in api_results)
    output = f"Gathered data from {len(api_results)} APIs in {total_latency}ms total"

    return OutputObject(
        llm_output=output,
        output_context={
            "api_results": api_results,
            "total_latency_ms": total_latency,
            "apis_called": len(api_results),
        },
    )


# ====================
# CONTEXT-AWARE MODELS
# ====================


def context_qa_model(input: str, context: str | None = None) -> OutputObject:
    """Model that answers questions based on provided context."""
    if not context:
        return OutputObject(
            llm_output="No context provided. Please provide context for me to answer your question.",
            output_context={"error": "no_context"},
        )

    # Simple keyword matching
    input_words = set(input.lower().split())
    context_sentences = context.split(".")

    relevant_sentences = []
    for sentence in context_sentences:
        sentence_words = set(sentence.lower().split())
        overlap = input_words.intersection(sentence_words)
        if len(overlap) > 1:  # At least 2 words match
            relevant_sentences.append(sentence.strip())

    if relevant_sentences:
        answer = ". ".join(relevant_sentences[:2]) + "."
        confidence = min(len(relevant_sentences) * 0.3, 0.9)
    else:
        answer = "I couldn't find specific information about that in the context."
        confidence = 0.1

    return OutputObject(
        llm_output=answer,
        output_context={
            "relevant_sentences": len(relevant_sentences),
            "confidence": confidence,
            "context_length": len(context),
            "question_words": list(input_words),
        },
    )


def entity_extraction_model(input: str, context: str | None = None) -> OutputObject:
    """Model that extracts entities from input and context."""
    import re

    entities = {
        "dates": [],
        "numbers": [],
        "capitalized_words": [],
        "emails": [],
        "urls": [],
    }

    text = input
    if context:
        text += " " + context

    # Extract various entities
    entities["dates"] = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    entities["numbers"] = re.findall(r"\b\d+\.?\d*\b", text)
    entities["capitalized_words"] = re.findall(r"\b[A-Z][a-z]+\b", text)
    entities["emails"] = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
    )
    entities["urls"] = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        text,
    )

    # Count total entities
    total_entities = sum(len(v) for v in entities.values())

    output_lines = [f"Found {total_entities} entities:"]
    for entity_type, values in entities.items():
        if values:
            output_lines.append(
                f"- {entity_type}: {', '.join(values[:3])}{'...' if len(values) > 3 else ''}"
            )

    return OutputObject(
        llm_output="\n".join(output_lines),
        output_context={
            "entities": entities,
            "total_entities": total_entities,
            "text_length": len(text),
        },
    )


# ====================
# FILE PROCESSING MODELS
# ====================


def file_analyzer_model(input: str, files: list[Any] | None = None) -> OutputObject:
    """Model that analyzes uploaded files."""
    if not files:
        return OutputObject(
            llm_output="No files provided to analyze.",
            output_context={"error": "no_files"},
        )

    file_analysis = []
    total_size = 0

    for file in files:
        analysis = {
            "name": getattr(file, "name", "unknown"),
            "size": 0,
            "type": "unknown",
            "preview": "",
        }

        if hasattr(file, "read"):
            content = file.read()
            analysis["size"] = len(content)
            total_size += len(content)

            # Reset file pointer
            if hasattr(file, "seek"):
                file.seek(0)

            # Determine file type and preview
            file_name = str(analysis["name"])
            if file_name.endswith(".txt"):
                analysis["type"] = "text"
                analysis["preview"] = content[:100].decode("utf-8", errors="ignore")
            elif file_name.endswith(".json"):
                analysis["type"] = "json"
                try:
                    json_data = json.loads(content)
                    analysis["preview"] = f"JSON with {len(json_data)} keys"
                except (json.JSONDecodeError, ValueError):
                    analysis["preview"] = "Invalid JSON"
            elif file_name.endswith(".csv"):
                analysis["type"] = "csv"
                lines = content.decode("utf-8", errors="ignore").split("\n")
                analysis["preview"] = f"CSV with {len(lines)} rows"

        file_analysis.append(analysis)

    # Generate output
    output_lines = [f"Analyzed {len(files)} file(s):"]
    for analysis in file_analysis:
        output_lines.append(
            f"\n- {analysis['name']} ({analysis['type']}, {analysis['size']} bytes)"
        )
        if analysis["preview"]:
            output_lines.append(f"  Preview: {analysis['preview']}")

    output_lines.append(f"\nTotal size: {total_size} bytes")

    return OutputObject(
        llm_output="\n".join(output_lines),
        output_context={
            "file_count": len(files),
            "total_size": total_size,
            "file_analysis": file_analysis,
        },
    )


def csv_processor_model(input: str, files: list[Any] | None = None) -> OutputObject:
    """Model that processes CSV files."""
    import csv
    import io

    if not files:
        return OutputObject(
            llm_output="No CSV files provided.", output_context={"error": "no_files"}
        )

    # Find CSV files
    csv_files = [f for f in files if hasattr(f, "name") and f.name.endswith(".csv")]

    if not csv_files:
        return OutputObject(
            llm_output="No CSV files found in the uploaded files.",
            output_context={"error": "no_csv_files"},
        )

    results = []
    for csv_file in csv_files:
        content = csv_file.read()
        if hasattr(csv_file, "seek"):
            csv_file.seek(0)

        # Parse CSV
        csv_text = content.decode("utf-8", errors="ignore")
        csv_reader = csv.DictReader(io.StringIO(csv_text))

        rows = list(csv_reader)
        if rows:
            results.append(
                {
                    "filename": csv_file.name,
                    "rows": len(rows),
                    "columns": list(rows[0].keys()),
                    "sample": rows[0],
                }
            )

    # Generate response based on input question
    if "sum" in input.lower() or "total" in input.lower():
        # Try to sum numeric columns
        output = "Calculated sums for numeric columns:\n"
        for result in results:
            output += f"\n{result['filename']}:\n"
            rows = result.get("rows", [])
            for col in result["columns"]:
                try:
                    total = sum(float(row.get(col, 0)) for row in rows if row.get(col))
                    output += f"  - {col}: {total}\n"
                except (ValueError, TypeError):
                    pass
    else:
        # Default: show summary
        output = "CSV File Summary:\n"
        for result in results:
            output += f"\n{result['filename']}:"
            output += f"\n  - Rows: {result['rows']}"
            output += f"\n  - Columns: {', '.join(result['columns'])}"

    return OutputObject(llm_output=output, output_context={"csv_analysis": results})


# ====================
# RAG MODELS
# ====================


class SimpleRAGModel:
    """Simple Retrieval Augmented Generation model."""

    def __init__(self):
        # Simulate a document store
        self.documents = [
            {
                "id": 1,
                "content": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            },
            {
                "id": 2,
                "content": "Python is a high-level programming language. It was created by Guido van Rossum.",
            },
            {
                "id": 3,
                "content": "Machine learning is a subset of artificial intelligence. It enables systems to learn from data.",
            },
            {
                "id": 4,
                "content": "The Earth orbits around the Sun. It takes approximately 365.25 days to complete one orbit.",
            },
            {
                "id": 5,
                "content": "Water boils at 100 degrees Celsius at sea level. The boiling point changes with altitude.",
            },
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Simple keyword-based retrieval."""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in self.documents:
            content = str(doc["content"])
            doc_words = set(content.lower().split())
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored_docs.append(
                    {
                        "document": doc,
                        "score": score,
                        "matched_words": list(query_words.intersection(doc_words)),
                    }
                )

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def __call__(self, input: str) -> OutputObject:
        """Process input using RAG approach."""
        # Retrieve relevant documents
        retrieved = self.retrieve(input)

        if not retrieved:
            return OutputObject(
                llm_output="I couldn't find any relevant information about that topic.",
                output_context={"retrieved_count": 0, "confidence": 0.0},
            )

        # Generate answer based on retrieved documents
        context = " ".join([r["document"]["content"] for r in retrieved])

        # Simple answer generation (in real RAG, this would use an LLM)
        answer_sentences = []
        for r in retrieved:
            sentences = r["document"]["content"].split(".")
            for sentence in sentences:
                if any(word in sentence.lower() for word in r["matched_words"]):
                    answer_sentences.append(sentence.strip())

        if answer_sentences:
            answer = ". ".join(answer_sentences[:2]) + "."
        else:
            answer = f"Based on the documents: {context[:200]}..."

        return OutputObject(
            llm_output=answer,
            output_context={
                "retrieved_count": len(retrieved),
                "top_document_id": retrieved[0]["document"]["id"],
                "confidence": retrieved[0]["score"] / len(input.split()),
                "matched_keywords": retrieved[0]["matched_words"],
                "retrieval_scores": [r["score"] for r in retrieved],
            },
        )


# ====================
# STATEFUL MODELS
# ====================


class StatefulChatModel:
    """Model that maintains conversation state."""

    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.context_window = 5  # Remember last 5 exchanges

    def __call__(self, input: str) -> OutputObject:
        """Process input with conversation history."""
        # Add to history
        self.conversation_history.append({"role": "user", "content": input})

        # Trim history to context window
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[
                -self.context_window * 2 :
            ]

        # Simple response generation based on history
        response = self._generate_response(input)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return OutputObject(
            llm_output=response,
            output_context={
                "conversation_length": len(self.conversation_history),
                "user_preferences": self.user_preferences,
                "context_used": min(
                    len(self.conversation_history), self.context_window
                ),
            },
        )

    def _generate_response(self, input: str) -> str:
        """Generate response based on input and history."""
        input_lower = input.lower()

        # Check for preference setting
        if "my name is" in input_lower:
            name = input.split("my name is")[-1].strip().rstrip(".")
            self.user_preferences["name"] = name
            return f"Nice to meet you, {name}! I'll remember that."

        # Check for preference usage
        if "name" in self.user_preferences and "who am i" in input_lower:
            return f"You are {self.user_preferences['name']}."

        # Check conversation continuity
        if len(self.conversation_history) > 1:
            last_exchange = self.conversation_history[-2]["content"]
            if "?" in last_exchange and "yes" in input_lower:
                return "Great! What else would you like to know?"
            elif "?" in last_exchange and "no" in input_lower:
                return "Understood. Is there anything else I can help with?"

        # Default response
        return f"I understand you said: '{input}'. How can I help you further?"


class CachedModel:
    """Model with response caching for efficiency."""

    def __init__(self, base_model=None):
        self.cache = {}
        self.cache_hits = 0
        self.total_calls = 0
        self.base_model = base_model or (lambda x: f"Processed: {x}")

    def __call__(self, input: str) -> OutputObject:
        """Process input with caching."""
        self.total_calls += 1

        # Check cache
        cache_key = input.lower().strip()
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_response = self.cache[cache_key]

            return OutputObject(
                llm_output=cached_response,
                output_context={
                    "cache_hit": True,
                    "cache_size": len(self.cache),
                    "cache_hit_rate": self.cache_hits / self.total_calls,
                    "total_calls": self.total_calls,
                },
            )

        # Generate new response
        start_time = time.time()

        if isinstance(self.base_model, type) and hasattr(self.base_model, "__call__"):
            response = self.base_model(input)
        else:
            response = self.base_model(input)

        processing_time = time.time() - start_time

        # Cache the response
        self.cache[cache_key] = response

        return OutputObject(
            llm_output=response,
            output_context={
                "cache_hit": False,
                "cache_size": len(self.cache),
                "cache_hit_rate": self.cache_hits / self.total_calls,
                "total_calls": self.total_calls,
                "processing_time_ms": processing_time * 1000,
            },
        )


# ====================
# SPECIALIZED MODELS
# ====================


def code_analyzer_model(input: str) -> OutputObject:
    """Model that analyzes code snippets."""
    # Detect programming language
    language = "unknown"
    if "def " in input or "import " in input:
        language = "python"
    elif "function " in input or "const " in input:
        language = "javascript"
    elif "public class" in input or "public static" in input:
        language = "java"

    # Count lines and analyze structure
    lines = input.strip().split("\n")
    code_stats = {
        "lines": len(lines),
        "empty_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(
            1 for line in lines if line.strip().startswith(("#", "//", "/*"))
        ),
        "language": language,
    }

    # Simple complexity estimate
    complexity_indicators = ["if", "else", "for", "while", "try", "catch"]
    complexity_score = sum(
        1 for indicator in complexity_indicators if indicator in input
    )

    output = "Code Analysis:\n"
    output += f"- Language: {language}\n"
    total_lines = int(code_stats.get('lines', 0))
    empty_lines = int(code_stats.get('empty_lines', 0))
    comment_lines = int(code_stats.get('comment_lines', 0))
    code_lines = total_lines - empty_lines - comment_lines
    output += f"- Lines: {total_lines} (Code: {code_lines}, Comments: {comment_lines})\n"
    output += f"- Complexity indicators: {complexity_score}"

    return OutputObject(
        llm_output=output,
        output_context={
            **code_stats,
            "complexity_score": complexity_score,
            "has_functions": "def " in input or "function " in input,
            "has_classes": "class " in input,
        },
    )


def sentiment_analyzer_model(input: str) -> OutputObject:
    """Simple sentiment analysis model."""
    # Simple keyword-based sentiment analysis
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "love",
        "best",
        "happy",
    }
    negative_words = {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "hate",
        "worst",
        "sad",
        "angry",
        "disappointed",
    }

    words = set(input.lower().split())
    positive_count = len(words.intersection(positive_words))
    negative_count = len(words.intersection(negative_words))

    # Determine sentiment
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = (
            positive_count / (positive_count + negative_count)
            if negative_count > 0
            else 1.0
        )
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = (
            negative_count / (positive_count + negative_count)
            if positive_count > 0
            else 1.0
        )
    else:
        sentiment = "neutral"
        confidence = 0.5

    output = f"Sentiment: {sentiment.capitalize()} (confidence: {confidence:.2f})"

    return OutputObject(
        llm_output=output,
        output_context={
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_words_found": positive_count,
            "negative_words_found": negative_count,
            "total_words": len(words),
        },
    )


# ====================
# ERROR HANDLING MODELS
# ====================


def robust_model(input: str) -> OutputObject:
    """Model with comprehensive error handling."""
    errors = []
    warnings = []

    try:
        # Validate input
        if not input:
            errors.append("Empty input received")
            return OutputObject(
                llm_output="Error: No input provided.",
                output_context={"errors": errors, "status": "failed"},
            )

        if len(input) > 10000:
            warnings.append("Input truncated to 10000 characters")
            input = input[:10000]

        # Process input (with potential errors)
        if "error" in input.lower():
            raise ValueError("User requested error simulation")

        if "warning" in input.lower():
            warnings.append("User input contains 'warning' keyword")

        # Normal processing
        result = f"Successfully processed: {input[:50]}..."

        return OutputObject(
            llm_output=result,
            output_context={
                "status": "success",
                "warnings": warnings,
                "input_length": len(input),
                "processing_time_ms": random.randint(10, 100),
            },
        )

    except ValueError as e:
        errors.append(f"ValueError: {str(e)}")
        return OutputObject(
            llm_output=f"Processing failed: {str(e)}",
            output_context={
                "status": "error",
                "errors": errors,
                "error_type": "ValueError",
            },
        )
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return OutputObject(
            llm_output="An unexpected error occurred during processing.",
            output_context={
                "status": "error",
                "errors": errors,
                "error_type": type(e).__name__,
            },
        )


# ====================
# DEMO FUNCTIONS
# ====================


async def demo_simple_models():
    """Demonstrate simple model usage."""
    print("\n=== Simple Models Demo ===")

    suite = Suite(
        title="Simple Models Test",
        tests=[
            Test(
                input_under_test="Hello, world!",
                checks=[Check(operator="includes", criteria="HELLO")],
            ),
            Test(
                input_under_test="Count the words in this sentence",
                checks=[Check(operator="includes", criteria="6")],
            ),
            Test(
                input_under_test="What is 25 + 17?",
                checks=[Check(operator="equals", criteria="42")],
            ),
        ],
    )

        # Test different simple models
    models = [
        ("uppercase", uppercase_model),
        ("word_count", word_count_model),
        ("calculator", calculator_model),
    ]
    
    for model_name, model_func in models:
        print(f"\nTesting {model_name} model...")
        run = await suite.run(model_func, model_name=model_name)
        await run.wait_for_run_completion()
        print(f"Pass rate: {run.pass_percentage:.1f}%")


async def demo_rag_model():
    """Demonstrate RAG model usage."""
    print("\n=== RAG Model Demo ===")

    rag_model = SimpleRAGModel()

    suite = Suite(
        title="RAG Model Test",
        tests=[
            Test(
                input_under_test="What is the capital of France?",
                checks=[Check(operator="includes", criteria="Paris")],
            ),
            Test(
                input_under_test="Who created Python?",
                checks=[Check(operator="includes", criteria="Guido")],
            ),
            Test(
                input_under_test="How long does Earth take to orbit the Sun?",
                checks=[Check(operator="includes", criteria="365")],
            ),
        ],
    )

    run = await suite.run(rag_model, model_name="simple-rag")
    await run.wait_for_run_completion()

    # Show detailed results
    detailed = await Run.from_id(run.id)
    print("\nRAG Model Results:")
    for result in detailed.test_results:
        if result.output_context:
            print(f"\nQuestion: {result.input_under_test}")
            print(f"Retrieved docs: {result.output_context.get('retrieved_count', 0)}")
            print(f"Confidence: {result.output_context.get('confidence', 0):.2f}")


async def demo_stateful_model():
    """Demonstrate stateful model usage."""
    print("\n=== Stateful Model Demo ===")

    chat_model = StatefulChatModel()

    suite = Suite(
        title="Stateful Model Test",
        tests=[
            Test(
                input_under_test="My name is Alice",
                checks=[Check(operator="includes", criteria="Alice")],
            ),
            Test(
                input_under_test="Who am I?",
                checks=[Check(operator="includes", criteria="Alice")],
            ),
        ],
    )

    run = await suite.run(chat_model, model_name="stateful-chat")
    await run.wait_for_run_completion()
    print(f"Stateful model pass rate: {run.pass_percentage:.1f}%")


async def main():
    """Run all custom model demonstrations."""
    print("=" * 50)
    print("CUSTOM MODELS EXAMPLES")
    print("=" * 50)

    try:
        await demo_simple_models()
        await demo_rag_model()
        await demo_stateful_model()

        print("\n" + "=" * 50)
        print("All custom model examples completed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError in examples: {e}")
        raise


if __name__ == "__main__":
    # Import Run for detailed results
    from vals import Run

    asyncio.run(main())
