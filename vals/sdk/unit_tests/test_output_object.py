"""Tests for the OutputObject feature."""

import pytest
from vals.sdk.types import OutputObject
from vals.graphql_client.input_types import QuestionAnswerPairInputType
from vals.sdk.suite import Suite
from vals.sdk.types import Test


class TestOutputObject:
    """Test suite for OutputObject functionality."""

    def test_output_object_creation(self):
        """Test creating an OutputObject with all fields."""
        output = OutputObject(
            llm_output="Test response",
            output_context={"key": "value", "number": 42},
            duration=1.5,
            in_tokens=10,
            out_tokens=20,
        )

        assert output.llm_output == "Test response"
        assert output.output_context == {"key": "value", "number": 42}
        assert output.duration == 1.5
        assert output.in_tokens == 10
        assert output.out_tokens == 20

    def test_output_object_minimal(self):
        """Test creating an OutputObject with only required fields."""
        output = OutputObject(llm_output="Minimal response")

        assert output.llm_output == "Minimal response"
        assert output.output_context is None
        assert output.duration is None
        assert output.in_tokens is None
        assert output.out_tokens is None

    def test_output_object_validation(self):
        """Test that OutputObject validates input properly."""
        # Should raise error without llm_output
        with pytest.raises(ValueError):
            OutputObject()

    @pytest.mark.asyncio
    async def test_process_model_output_with_output_object(self):
        """Test that _process_model_output correctly handles OutputObject."""
        suite = Suite(title="Test", description="Test suite")
        test = Test(input_under_test="test input")
        test._id = "test-123"

        output = OutputObject(
            llm_output="Test response",
            output_context={"source": "test"},
            duration=2.0,
            in_tokens=5,
            out_tokens=10,
        )

        result = await suite._process_model_output(
            output=output,
            test=test,
            file_ids=["file1", "file2"],
            time_start=0.0,
            time_end=3.0,  # Will be overridden by output.duration
            in_tokens_start=0,
            out_tokens_start=0,
            in_tokens_end=20,  # Will be overridden by output.in_tokens
            out_tokens_end=30,  # Will be overridden by output.out_tokens
        )

        assert isinstance(result, QuestionAnswerPairInputType)
        assert result.llm_output == "Test response"
        assert result.output_context == {"source": "test"}
        assert result.file_ids == ["file1", "file2"]
        assert result.context == {}
        assert result.input_under_test == "test input"
        assert result.test_id == "test-123"

        # Check metadata was properly set from OutputObject
        assert result.metadata.duration_seconds == 2.0
        assert result.metadata.in_tokens == 5
        assert result.metadata.out_tokens == 10

    @pytest.mark.asyncio
    async def test_process_model_output_backwards_compatibility_dict(self):
        """Test that _process_model_output still handles dict correctly."""
        suite = Suite(title="Test", description="Test suite")
        test = Test(input_under_test="test input")
        test._id = "test-123"

        output = {
            "llm_output": "Dict response",
            "output_context": {"format": "dict"},
            "metadata": {"in_tokens": 15, "out_tokens": 25, "duration_seconds": 1.0},
        }

        result = await suite._process_model_output(
            output=output,
            test=test,
            file_ids=[],
            time_start=0.0,
            time_end=2.0,
            in_tokens_start=0,
            out_tokens_start=0,
            in_tokens_end=100,
            out_tokens_end=200,
        )

        assert result.llm_output == "Dict response"
        assert result.output_context == {"format": "dict"}
        assert result.metadata.in_tokens == 15
        assert result.metadata.out_tokens == 25
        assert result.metadata.duration_seconds == 1.0

    @pytest.mark.asyncio
    async def test_process_model_output_backwards_compatibility_string(self):
        """Test that _process_model_output still handles string correctly."""
        suite = Suite(title="Test", description="Test suite")
        test = Test(input_under_test="test input")
        test._id = "test-123"

        output = "Simple string response"

        result = await suite._process_model_output(
            output=output,
            test=test,
            file_ids=[],
            time_start=0.0,
            time_end=1.0,
            in_tokens_start=0,
            out_tokens_start=0,
            in_tokens_end=10,
            out_tokens_end=20,
        )

        assert result.llm_output == "Simple string response"
        assert result.output_context is None
        assert result.metadata.in_tokens == 10
        assert result.metadata.out_tokens == 20
        assert result.metadata.duration_seconds == 1.0

    def test_model_function_simple_with_output_object(self):
        """Test that a simple model function can return OutputObject."""

        def model(input_text: str) -> OutputObject:
            return OutputObject(
                llm_output=f"Processed: {input_text}",
                output_context={"input_length": len(input_text)},
            )

        result = model("Hello")
        assert isinstance(result, OutputObject)
        assert result.llm_output == "Processed: Hello"
        assert result.output_context == {"input_length": 5}

    def test_model_function_with_context_and_output_object(self):
        """Test that a context model function can return OutputObject."""

        def model(input_text: str, files: dict, context: dict) -> OutputObject:
            return OutputObject(
                llm_output=f"Files: {len(files)}, Context: {context.get('key', 'none')}",
                output_context={"file_count": len(files), "has_context": bool(context)},
            )

        result = model("test", {"file1": b"data"}, {"key": "value"})
        assert isinstance(result, OutputObject)
        assert "Files: 1" in result.llm_output
        assert "Context: value" in result.llm_output
        assert result.output_context["file_count"] == 1
        assert result.output_context["has_context"] is True

    @pytest.mark.asyncio
    async def test_output_object_with_none_values(self):
        """Test OutputObject with None values uses defaults correctly."""
        suite = Suite(title="Test", description="Test suite")
        test = Test(input_under_test="test input")
        test._id = "test-123"

        output = OutputObject(
            llm_output="Test",
            output_context=None,  # Explicitly None
            duration=None,
            in_tokens=None,
            out_tokens=None,
        )

        result = await suite._process_model_output(
            output=output,
            test=test,
            file_ids=[],
            time_start=0.0,
            time_end=5.0,
            in_tokens_start=10,
            out_tokens_start=20,
            in_tokens_end=30,
            out_tokens_end=50,
        )

        # Should use calculated defaults when OutputObject fields are None
        assert result.metadata.duration_seconds == 5.0
        assert result.metadata.in_tokens == 20  # 30 - 10
        assert result.metadata.out_tokens == 30  # 50 - 20
        assert result.output_context is None
