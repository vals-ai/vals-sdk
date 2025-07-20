import base64
import json
from collections.abc import Iterator
from typing import Any, Literal, cast

# from google.cloud import storage
from google.genai import Client, types
from google.genai.types import FinishReason, GenerateContentResponse
from openai._client import AsyncOpenAI
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)
from vals.sdk.model_library.exceptions import MaxOutputTokensExceededError
from vals.sdk.model_library.openai.openai import OpenAIModel

# class GoogleBatchMixin(LLMBatchMixin):
#     completed_batch_statuses = [
#         JobState.JOB_STATE_SUCCEEDED,
#         JobState.JOB_STATE_FAILED,
#         JobState.JOB_STATE_CANCELLED,
#         JobState.JOB_STATE_PAUSED,
#     ]


# self.storage_client = storage.Client(credentials=settings.GCP_CREDS)
#
#     def create_batch_query_request(self, custom_id: str, query_args: dict[str, Any]):
#         """Creates request to gemini prediction API."""
#         parameters = query_args["parameters"]
#         temperature = 0
#         stopoptions: list[str] = []
#
#         temperature = parameters.get("temperature", 0)
#         max_tokens = parameters.get("max_tokens", 512)
#         system_prompt = parameters.get("system_prompt", "").strip()
#
#         if "new_line_stop_option" in parameters and parameters["new_line_stop_option"]:
#             stopoptions = stopoptions + ["\n"]
#
#         safety_config = [
#             {
#                 "category": "HARM_CATEGORY_HARASSMENT",
#                 "threshold": "BLOCK_NONE",
#             },
#             {
#                 "category": "HARM_CATEGORY_HATE_SPEECH",
#                 "threshold": "BLOCK_NONE",
#             },
#             {
#                 "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                 "threshold": "BLOCK_NONE",
#             },
#             {
#                 "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                 "threshold": "BLOCK_NONE",
#             },
#         ]
#
#         generation_config = {
#             "maxOutputTokens": max_tokens,
#             "temperature": temperature,
#             "topP": 1,
#         }
#
#         if stopoptions != []:
#             generation_config["stopSequences"] = stopoptions
#
#         system_instruction = {
#             "role": "user",
#             "parts": {"text": system_prompt},
#         }
#
#         parts = []
#         parts.append({"text": query_args["prompt"]})
#         if self.supports_images and "images" in query_args:
#             images = query_args["images"]
#             for image_base64, image_ext, _ in images:
#                 parts.append(
#                     {
#                         "mime_type": f"image/{image_ext}",
#                         "data": image_base64,
#                     }
#                 )
#
#         return {
#             "request": {
#                 "contents": [
#                     {
#                         "role": "user",
#                         "parts": parts,
#                     }
#                 ],
#                 "generationConfig": generation_config,
#                 "safetySettings": safety_config,
#                 "systemInstruction": system_instruction,
#                 "labels": {
#                     "qa_pair_id": custom_id,
#                 },
#             }
#         }
#
#     async def batch_query(
#         self,
#         batch_name: str,
#         requests: list[dict[str, Any]],
#     ) -> str:
#         input_file_name = f"{batch_name}/input.jsonl"
#         output_file_name = f"{batch_name}/output"
#
#         input_uri = settings.GS_URI + "/" + input_file_name
#         output_uri = settings.GS_URI + "/" + output_file_name
#
#         try:
#             input_jsonl_str = "\n".join(json.dumps(req) for req in requests)
#             input_jsonl_bytes = io.BytesIO(input_jsonl_str.encode("utf-8"))
#             bucket = self.storage_client.bucket(
#                 urlparse(settings.GS_URI).netloc
#             )  # function wants bucket name not bucket uri
#             blob = bucket.blob(f"{batch_name}/input.jsonl")
#             blob.upload_from_file(input_jsonl_bytes)
#             logger.info(f"Uploaded input files to GCS: {input_uri}")
#         except Exception as e:
#             raise Exception(f"Failed to upload input files to GCS: {e}")
#
#         job = await self.vertex_client.aio.batches.create(
#             model=self.model_name,
#             src=input_uri,
#             config=CreateBatchJobConfig(dest=output_uri),
#         )
#
#         if job.name is None:
#             raise Exception("Failed to create batch job")
#
#         return job.name
#
#     async def get_batch_status(self, batch_id: str) -> str:
#         job = await self.vertex_client.aio.batches.get(name=batch_id)
#
#         if job is None or job.state is None:
#             raise Exception("Failed to get batch job status")
#
#         return job.state
#
#     def is_status_completed(self, batch_status: str) -> bool:
#         return batch_status in self.completed_batch_statuses
#
#     def is_status_failed(self, batch_status: str) -> bool:
#         return batch_status == JobState.JOB_STATE_FAILED
#
#     def is_status_cancelled(self, batch_status: str) -> bool:
#         return batch_status == JobState.JOB_STATE_CANCELLED
#
#     def get_batch_progress(self, batch_id: str) -> int:
#         raise NotImplementedError("Gemini does not support batch progress")
#
#     async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
#         try:
#             # Note: This approach is necessary because the Vertex AI API
#             # doesn't provide a direct way to access batch output files
#             prefix = f"run-{batch_id}/output/"
#             bucket = self.storage_client.bucket(
#                 urlparse(settings.GS_URI).netloc
#             )  # function wants bucket name not bucket uri
#             blobs = bucket.list_blobs(prefix=prefix)
#             jsonl_blob = next(
#                 (b for b in blobs if b.name.endswith("predictions.jsonl")), None
#             )
#
#             if not jsonl_blob:
#                 raise Exception(f"No predictions.jsonl file found for batch {batch_id}")
#
#             output_jsonl_str = jsonl_blob.download_as_text()
#         except Exception as e:
#             raise Exception(f"Failed to download output file: {e}")
#
#         try:
#             output_lines = [
#                 json.loads(line) for line in output_jsonl_str.split("\n") if line
#             ]
#         except Exception as e:
#             raise Exception(f"Failed to decode output file: {e}")
#
#         results = []
#
#         for line in output_lines:
#             error_message = line["status"]
#             custom_id = line["request"]["labels"]["qa_pair_id"]
#
#             try:
#                 response = line["response"]
#                 in_tokens = response["usageMetadata"]["promptTokenCount"]
#                 out_tokens = response["usageMetadata"]["candidatesTokenCount"]
#                 llm_output = response["candidates"][0]["content"]["parts"][0]["text"]
#             except Exception:
#                 in_tokens = 0
#                 out_tokens = 0
#                 llm_output = ""
#
#             results.append(
#                 BatchResult(
#                     custom_id=custom_id,
#                     llm_output=llm_output,
#                     live_metadata={
#                         "in_tokens": in_tokens,
#                         "out_tokens": out_tokens,
#                     },
#                     error_message=error_message,
#                 )
#             )
#
#         return results
#
#     def cancel_batch_request(self, batch_id: str):
#         logger.info(f"Cancelling batch {batch_id}")
#         self.vertex_client.batches.cancel(name=batch_id)


class GoogleModel(LLM):
    SAFETY_CONFIG = [
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
        self.model_name = self.model_name.replace("-thinking", "")

        self._key = cast(str, json.loads(sdk.model_library_settings.GEMINI_KEYS)[0])

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
    ) -> Iterator[dict[str, Any]]:
        """Append images to the request body"""
        if not images:
            return iter(())
        for image in images:
            match image:
                case FileWithBase64():
                    yield {
                        "mime_type": f"image/{image.mime}",
                        "data": image.base64,
                    }
                case _:
                    raise Exception("Unsupported image type")

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

        history = [{"role": "user", "parts": [*self._append_images(images)]}]

        body: dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if "system_prompt" in kwargs and self.model_name != "gemini-2.0-flash-exp":
            system_prompt = str(kwargs.get("system_prompt", ""))
            del kwargs["system_prompt"]

            if system_prompt.strip():
                body["system_instruction"] = system_prompt

        body.update(kwargs)

        # Create content based on whether there are images or not
        text_part = types.Part.from_text(text=prompt)
        image_parts = []
        if history and history[0].get("parts"):
            image_parts = [
                types.Part.from_bytes(
                    mime_type=part["mime_type"], data=base64.b64decode(part["data"])
                )
                for part in history[0]["parts"]
            ]
        full_content = types.Content(parts=image_parts + [text_part], role="user")

        # Handle thinking models by checking if "-thinking" is in the model name
        if self.reasoning:
            # thinking budget
            body["thinking_config"] = types.ThinkingConfig(thinking_budget=24576)

        # For non-thinking flash preview models, explicitly set thinking_budget to 0
        elif "flash-preview" in self.model_name:
            body["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response: GenerateContentResponse = (
            await self.get_client().aio.models.generate_content(
                model=self.model_name,
                contents=full_content,
                config=body,
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
