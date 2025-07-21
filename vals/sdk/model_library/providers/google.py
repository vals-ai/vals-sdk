from __future__ import annotations

import base64
import io
import json
from collections.abc import Iterator
from typing import Any, Literal, cast
from urllib.parse import urlparse

from google.cloud import storage
from google.genai import Client, types
from google.genai.types import (
    BatchJob,
    CreateBatchJobConfig,
    FinishReason,
    GenerateContentResponse,
    JobState,
    Part,
)
from openai._client import AsyncOpenAI
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    BatchResult,
    FileInput,
    FileWithBase64,
    LLMBatchMixin,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import MaxOutputTokensExceededError
from vals.sdk.model_library.openai.openai import OpenAIModel
from vals.sdk.model_library.utils import deep_model_dump


class GoogleBatchMixin(LLMBatchMixin):
    completed_batch_statuses: list[JobState] = [
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    ]

    def __init__(self, google: GoogleModel):
        self._root: GoogleModel = google
        self.client: Client = self._root.vertex_client
        self.storage_client: storage.Client = storage.Client(
            credentials=sdk.model_library_settings.GCP_CREDS
        )
        self.GS_URI: str = sdk.model_library_settings.GS_URI

    @override
    def create_batch_query_request(
        self,
        custom_id: str,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        request = {
            "request": {
                "labels": {"qa_pair_id": custom_id},
                **self._root.query_body(prompt, images=images, files=files, **kwargs),  # type: ignore
            }
        }
        dumped = deep_model_dump(request)
        if not isinstance(dumped, dict):
            raise Exception("Failed to dump request")
        return dumped

    @override
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        input_file_name = f"{batch_name}/input.jsonl"
        output_file_name = f"{batch_name}/output"

        input_uri = self.GS_URI + "/" + input_file_name
        output_uri = self.GS_URI + "/" + output_file_name

        try:
            input_jsonl_str = "\n".join(json.dumps(req) for req in requests)
            input_jsonl_bytes = io.BytesIO(input_jsonl_str.encode("utf-8"))
            bucket = self.storage_client.bucket(
                urlparse(self.GS_URI).netloc
            )  # function wants bucket name not bucket uri
            blob = bucket.blob(f"{batch_name}/input.jsonl")
            blob.upload_from_file(input_jsonl_bytes)
            self._root.logger.info(f"Uploaded input files to GCS: {input_uri}")
        except Exception as e:
            raise Exception(f"Failed to upload input files to GCS: {e}")

        job: BatchJob = await self.client.aio.batches.create(
            model=self._root.model_name,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_uri),
        )

        if job.name is None:
            raise Exception("Failed to create batch job")

        return job.name

    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        try:
            # Note: This approach is necessary because the Vertex AI API
            # doesn't provide a direct way to access batch output files
            prefix = f"run-{batch_id}/output/"
            bucket = self.storage_client.bucket(
                urlparse(self.GS_URI).netloc
            )  # function wants bucket name not bucket uri
            blobs = bucket.list_blobs(prefix=prefix)
            jsonl_blob = next(
                (b for b in blobs if b.name.endswith("predictions.jsonl")), None
            )

            if not jsonl_blob:
                raise Exception(f"No predictions.jsonl file found for batch {batch_id}")

            output_jsonl_str = jsonl_blob.download_as_text()
        except Exception as e:
            raise Exception(f"Failed to download output file: {e}")

        try:
            output_lines = [
                json.loads(line) for line in output_jsonl_str.split("\n") if line
            ]
        except Exception as e:
            raise Exception(f"Failed to decode output file: {e}")

        results: list[BatchResult] = []
        for line in output_lines:
            error_message = line["status"]
            custom_id = line["request"]["labels"]["qa_pair_id"]

            metadata: QueryResultMetadata | None = None
            try:
                response = line["response"]
                llm_output = response["candidates"][0]["content"]["parts"][0]["text"]
                if response["usageMetadata"]:
                    metadata = QueryResultMetadata(
                        in_tokens=response["usageMetadata"]["promptTokenCount"],
                        out_tokens=response["usageMetadata"]["candidatesTokenCount"],
                    )
            except Exception:
                llm_output = ""

            results.append(
                BatchResult(
                    custom_id=custom_id,
                    llm_output=llm_output,
                    live_metadata=metadata,
                    error_message=error_message,
                )
            )

        return results

    @override
    async def cancel_batch_request(self, batch_id: str):
        self._root.logger.info(f"Cancelling batch {batch_id}")
        self.client.batches.cancel(name=batch_id)

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        raise NotImplementedError("Gemini does not support batch progress")

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        job = await self.client.aio.batches.get(name=batch_id)

        if not job.state:
            raise Exception("Failed to get batch job status")
        return job.state

    @override
    @classmethod
    def is_status_completed(cls, batch_status: str) -> bool:
        return batch_status in cls.completed_batch_statuses

    @override
    @classmethod
    def is_status_failed(cls, batch_status: str) -> bool:
        return batch_status == JobState.JOB_STATE_FAILED

    @override
    @classmethod
    def is_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == JobState.JOB_STATE_CANCELLED


class GoogleModel(LLM):
    SAFETY_CONFIG: list[dict[str, str]] = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    _client: Client | None = None

    @override
    def get_client(self) -> Client:
        if not GoogleModel._client:
            GoogleModel._client = Client(
                api_key=self._key,
            )
        return GoogleModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["google"] = "google",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.model_name: str = self.model_name.replace("-thinking", "")

        self._key: str = cast(
            str, json.loads(sdk.model_library_settings.GEMINI_KEYS)[0]
        )

        self.vertex_client: Client = Client(
            vertexai=True,
            project=sdk.model_library_settings.GCP_PROJECT_ID,
            location=sdk.model_library_settings.GCP_REGION,
            credentials=sdk.model_library_settings.GCP_CREDS,
        )

        self.batch: LLMBatchMixin | None = (
            GoogleBatchMixin(self) if self.supports_batch else None
        )

        # https://ai.google.dev/gemini-api/docs/openai
        self.delegate: OpenAIModel | None = (
            None
            if self.native
            else OpenAIModel(
                model_name=model_name,
                provider=provider,
                config=config,
                custom_client=AsyncOpenAI(
                    api_key=self._key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                use_completions=True,
            )
        )

    def _append_images(
        self,
        images: list[FileInput],
    ) -> Iterator[Part]:
        """Append images to the request body"""
        if not images:
            return iter(())
        for image in images:
            match image:
                case FileWithBase64():
                    image_bytes = base64.b64decode(image.base64)
                    yield types.Part.from_bytes(
                        mime_type=f"image/{image.mime}",
                        data=image_bytes,
                    )
                case _:
                    raise Exception("Unsupported image type")

    def query_body(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        content_user: list[Part] = []
        content_user.extend(self._append_images(images))
        content_user.append(Part.from_text(text=prompt))

        body: dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "safety_settings": GoogleModel.SAFETY_CONFIG,
        }

        if "system_prompt" in kwargs and self.model_name != "gemini-2.0-flash-exp":
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                body["system_instruction"] = system_prompt

        body.update(kwargs)

        # Handle thinking models by checking if "-thinking" is in the model name
        if self.reasoning:
            # thinking budget
            body["thinking_config"] = types.ThinkingConfig(thinking_budget=24576)

        # For non-thinking flash preview models, explicitly set thinking_budget to 0
        elif "flash-preview" in self.model_name:
            body["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        return {
            "contents": types.Content(parts=content_user, role="user"),
            "config": body,
        }

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate.query(
                prompt, images=images, files=files, **kwargs
            )

        body: dict[str, Any] = self.query_body(
            prompt, images=images, files=files, **kwargs
        )

        response: GenerateContentResponse = (
            await self.get_client().aio.models.generate_content(
                model=self.model_name,
                **body,  # type: ignore
            )
        )

        if (
            not response.candidates
            or response.candidates[0].finish_reason == FinishReason.MAX_TOKENS
        ):
            raise MaxOutputTokensExceededError()

        content = response.candidates[0].content
        if not content or not content.parts:
            raise Exception("Model returned no content")

        metadata: QueryResultMetadata | None = None
        if response.usage_metadata:
            metadata = QueryResultMetadata(
                in_tokens=response.usage_metadata.prompt_token_count or 0,
                out_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        answer: str = ""
        # NOTE: thought appears to always be empty
        thought: str = ""
        if self.reasoning and content.parts and metadata:
            for part in content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thought += part.text
                else:
                    answer += part.text
            if thought:
                metadata["reasoning"] = thought

        return answer, metadata
