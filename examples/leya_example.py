import random
from dataclasses import dataclass
from typing import List

from vals.sdk.run import (
    Metadata,
    QuestionAnswerPair,
    _create_question_answer_set,
    get_run_url,
    pull_run_results_json,
    start_run,
    wait_for_run_completion,
)
from vals.sdk.suite import update_suite


@dataclass
class CheckClass:
    operator: str
    criteria: str


@dataclass
class ExampleClass:
    model_input: str

    checks: List[CheckClass]

    @property
    def dict(self) -> dict:
        return {
            "checks": [
                {"operator": check.operator, "criteria": check.criteria}
                for check in self.checks
            ]
        }


@dataclass
class OutputClass:
    response: str


def extract_context_from_output(output: OutputClass):
    return {"context": output.response + "!!!"}


def handle_leya_examples(
    name: str, id: str, examples: list[ExampleClass], outputs: list[OutputClass]
):
    vals_dict = {
        "title": f"{self.name} evaluation",
        "description": f"The outputs of evaluating {self.name} on a batch of examples",
        "tests": [
            {
                "input_under_test": (
                    example.model_input
                    if example.model_input
                    else f"[{idx + 1}]: No input provided"
                ),  # TODO: add something here for the test input
                "context": await extract_context_from_output(output),
                **example.dict,
            }
            for idx, (example, output) in enumerate(zip(self.examples, outputs))
        ],
    }
    # Normally, would not be necessary now, but we need to update the context.
    update_suite(self.id, vals_dict)

    question_answer_pairs = [
        QuestionAnswerPair(
            input_under_test=(
                example.model_input
                if example.model_input
                else f"[{idx + 1}]: No input provided"
            ),
            llm_output=output.response,
            context=extract_context_from_output(output),
            file_ids=[],
            metadata=Metadata(
                in_tokens=0,
                out_tokens=0,
                duration_seconds=0,
            ),
            test_id=None,
        )
        for idx, (example, output) in enumerate(zip(examples, outputs))
    ]

    parameters = {
        "eval_model": "gpt-4o",
        "model_under_test": self.name,
        "description": f"Evals for {self.name}.",
        "temperature": 0.1,
        "maximum_threads": 10,
        "run_golden_eval": False,
        "use_fixed_output": True,
        "max_output_tokens": 512,
        "heavyweight_factor": 1,
        "run_golden_eval_style": True,
        "run_golden_eval_format": True,
        "run_golden_eval_content": True,
    }

    question_answer_set_id = _create_question_answer_set(
        test_suite_id=self.id,
        question_answer_pairs=question_answer_pairs,
        model_id=name,
        parameters=parameters,
    )

    run_id = start_run(
        suite_id=id,
        parameters=parameters,
        qa_set_id=question_answer_set_id,
    )

    wait_for_run_completion(run_id)
    results = pull_run_results_json(run_id)
    run_url = get_run_url(run_id)


handle_leya_examples(
    name="test name",
    id="abc0e6d5-67d2-46fb-bba5-994f5f686026",
    examples=[
        ExampleClass(
            model_input="What is QSBS?",
            checks=[CheckClass(operator="includes", criteria="C Corp")],
        ),
        ExampleClass(
            model_input="What is an 83b election?",
            checks=[CheckClass(operator="includes", criteria="early exercise")],
        ),
    ],
    outputs=[
        OutputClass(
            response="QSBS is a company that provides software solutions for the financial industry."
        ),
        OutputClass(
            response="An 83b election is a type of election that is used to elect the President of the United States."
        ),
    ],
)
