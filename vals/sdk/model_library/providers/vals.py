from __future__ import annotations

import random
from typing import Any, Literal

from typing_extensions import override

from vals.sdk.model_library.base import (
    LLM,
    FileInput,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)

# class DummyAIBatchMixin(LLMBatchMixin):
#     BATCH_EXP = 60 * 10  # 10 minutes
#     redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
#
#     def __init__(self, openai: DummyAIModel):
#         self._root: DummyAIModel = openai
#
#     def create_batch_query_request(
#         self,
#         custom_id: str,
#         prompt: str,
#         *,
#         images: list[FileInput],
#         files: list[FileInput],
#         **kwargs: object,
#     ) -> dict[str, Any]:
#         return {
#             "custom_id": custom_id,
#             "method": "",
#             "url": "",
#             "body": self._root.query_body(prompt, images=images, files=files, **kwargs),
#         }
#
#     async def batch_query(self, batch_name: str, requests: list[dict[str, Any]]) -> str:
#         """Sends a batch api query and returns batch_id"""
#         if random.random() < FAIL_RATE and "evaluator" not in self.model_name:
#             raise Exception("Something went wrong in batch query")
#
#         random_id = "".join(random.choices("0123456789", k=8))
#         batch_id = f"dumbar_batch_{random_id}"
#         batch_obj = {
#             "status": "in_progress",
#             "batch_name": batch_name,
#             "requests": requests,
#         }
#         redis_client.setex(f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj))
#         return batch_id
#
#     async def get_batch_status(self, batch_id: str) -> str:
#         batch_obj = self._get_batch_obj(batch_id)
#         if random.random() < FAIL_RATE and "evaluator" not in self.model_name:
#             batch_obj["status"] = "failed"
#         elif batch_obj["status"] != "cancelled":
#             batch_obj["status"] = random.choice(["completed", "in_progress"])
#
#         redis_client.setex(f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj))
#         return batch_obj["status"]
#
#     @override
#     @classmethod
#     def is_status_completed(cls, batch_status: str) -> bool:
#         return True
#
#     @override
#     @classmethod
#     def is_status_cancelled(cls, batch_status: str) -> bool:
#         return batch_status == "cancelled"
#
#     @override
#     @classmethod
#     def is_status_failed(cls, batch_status: str) -> bool:
#         return batch_status == "failed"
#
#     async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
#         if random.random() < FAIL_RATE and "evaluator" not in self.model_name:
#             raise Exception("Something went wrong in parsing batch results")
#
#         batch_obj = self._get_batch_obj(batch_id)
#         requests = batch_obj["requests"]
#         batch_results = []
#         for req in requests:
#             if random.random() < FAIL_RATE and "evaluator" not in self.model_name:
#                 batch_results.append(
#                     BatchResult(
#                         custom_id=req["custom_id"],
#                         error_message="Dumbmar queried unsuccessfully",
#                     )
#                 )
#             else:
#                 batch_results.append(
#                     BatchResult(
#                         custom_id=req["custom_id"],
#                         llm_output="Dumbmar queried successfully",
#                         live_metadata={
#                             "in_tokens": 0,
#                             "out_tokens": 0,
#                         },
#                     )
#                 )
#         return batch_results
#
#     def get_batch_progress(self, batch_id: str) -> int:
#         batch_obj = self._get_batch_obj(batch_id)
#         return len(batch_obj["requests"])
#
#     def cancel_batch_request(self, batch_id: str):
#         batch_obj = self._get_batch_obj(batch_id)
#         batch_obj["status"] = "cancelled"
#         redis_client.setex(f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj))
#
#     def _get_batch_obj(self, batch_id: str):
#         batch_obj = json.loads(redis_client.get(f"dummy_batch:{batch_id}"))  # type: ignore
#         return batch_obj


class DummyAIModel(LLM):
    FAIL_RATE = 0.1

    def __init__(
        self,
        model_name: str,
        provider: Literal["vals"] = "vals",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        # self.batch: LLMBatchMixin | None = (
        #     DummyAIBatchMixin(self) if self.supports_batch else None
        # )

    def query_body(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": 0,
            "files": len(files),
            "images": len(images),
            "messages": ["Dumbar dummy message"],
        }
        body.update(kwargs)
        return body

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        _ = self.query_body(prompt, images=images, files=files, **kwargs)

        if (
            random.random() < DummyAIModel.FAIL_RATE
            and "evaluator" not in self.model_name
        ):
            raise Exception("Dumbmar couldn't retrieve output.")

        return "Dumbmar retrieved output successfully.", QueryResultMetadata(
            in_tokens=0,
            out_tokens=0,
        )
