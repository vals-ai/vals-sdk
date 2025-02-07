import json
import time
from io import BytesIO
from typing import Any, Callable, List, Optional, Union
from inspect_ai import Task
from inspect_ai.solver import TaskState, Solver, Generate, generate
from inspect_ai.scorer import Target, Score, Scorer
from inspect_ai.model import ModelOutput
from copy import deepcopy

from vals.sdk.types import (
    QuestionAnswerPair,
    OperatorInput,
    OperatorOutput,
    CustomModelInput,
    CustomModelOutput,
    ModelCustomOperatorFunctionType,
)


class InspectWrapper:
    """Wrapper to convert between inspect Task components and Vals framework components."""

    model = "vals-sdk"

    def __init__(self, task: Task):
        self.task = task

    def custom_model_input_to_task_state(
        self, custom_model_input: CustomModelInput
    ) -> TaskState:
        metadata = (
            custom_model_input.context.get("inspect_context", {})
            .get("task_state", {})
            .get("metadata", {})
        )

        if "uploaded_files" in custom_model_input.context:
            metadata["uploaded_files"] = custom_model_input.context["uploaded_files"]

        return TaskState(
            model=self.model,
            sample_id=0,
            epoch=0,
            input=custom_model_input.input_under_test,
            messages=[],
            metadata=metadata,
        )

    def operator_input_to_task_state(self, operator_input: OperatorInput) -> TaskState:
        metadata = (
            operator_input.context.get("inspect_context", {})
            .get("task_state", {})
            .get("metadata", {})
        )

        if "uploaded_files" in operator_input.context:
            metadata["uploaded_files"] = operator_input.context["uploaded_files"]

        model_output = ModelOutput()
        model_output.completion = operator_input.model_output

        return TaskState(
            model=self.model,
            sample_id=0,
            epoch=0,
            input=operator_input.input,
            messages=[],
            metadata=metadata,
            output=model_output,
        )

    def operator_input_to_target(self, operator_input: OperatorInput) -> Target:
        return Target(target=operator_input.context["inspect_context"]["target"])

    def to_custom_model(
        self, solver: Solver | None = generate(), generate: Generate | None = None
    ) -> Callable[[str], Union[str, dict]]:
        """Convert inspect Solver to Vals CustomModel format."""
        solver = solver if solver is not None else self.task.solver
        if generate is None:
            raise ValueError("Generate function is required")

        async def wrapped_custom_model(
            input_under_test: str,
            files: dict[str, BytesIO],
            context: dict[str, Any],
        ) -> CustomModelOutput:
            custom_model_input = CustomModelInput(
                input_under_test=input_under_test,
                files=files,
                context=context,
            )
            # Create TaskState from input
            task_state = self.custom_model_input_to_task_state(custom_model_input)
            # Run solver
            start_time = time.time()
            result_state = await solver(task_state, generate)
            end_time = time.time()

            # Convert to CustomModelOutput format
            return {
                "llm_output": result_state.output.completion,
                "metadata": {
                    "in_tokens": result_state.output.usage.input_tokens,
                    "out_tokens": result_state.output.usage.output_tokens,
                    "duration_seconds": end_time - start_time,
                },
            }

        return wrapped_custom_model

    def to_single_custom_operator(self, scorer: Scorer):
        async def wrapped_custom_operator(
            operator_input: OperatorInput,
        ) -> OperatorOutput:
            task_state = self.operator_input_to_task_state(operator_input)
            target = self.operator_input_to_target(operator_input)

            score: Score = await scorer(task_state, target)

            explanation = score.explanation or ""
            if score.metadata is not None:
                explanation += "\n" + json.dumps(score.metadata)

            output = OperatorOutput(
                name=scorer.__qualname__.split(".")[0],
                score=int(score.as_float()),
                explanation=explanation,
            )
            return output

        return wrapped_custom_operator

    def to_custom_operators(
        self, scorer: Scorer | list[Scorer] | None = None
    ) -> List[ModelCustomOperatorFunctionType]:
        """Convert inspect Scorers to Vals CustomOperator format."""
        scorer = scorer if scorer is not None else self.task.scorer
        if not isinstance(scorer, list):
            scorer = [scorer]

        return [self.to_single_custom_operator(r) for r in scorer]
