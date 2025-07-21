from __future__ import annotations

import json
import random
from typing import Any, Literal, cast

import redis
from redis.client import Redis
from typing_extensions import override

from vals import sdk
from vals.sdk.model_library.base import (
    LLM,
    BatchResult,
    FileInput,
    LLMBatchMixin,
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
)

FAIL_RATE = 0.1
BATCH_EXP = 60 * 10  # 10 minutes
redis_client: Redis = redis.from_url(
    sdk.model_library_settings.REDIS_URL, decode_responses=True
)


class DummyAIBatchMixin(LLMBatchMixin):
    def __init__(self, openai: DummyAIModel):
        self._root: DummyAIModel = openai

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
        return {
            "custom_id": custom_id,
            "method": "",
            "url": "",
            "body": self._root.query_body(prompt, images=images, files=files, **kwargs),
        }

    @override
    async def batch_query(self, batch_name: str, requests: list[dict[str, Any]]) -> str:
        """Sends a batch api query and returns batch_id"""
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            raise Exception("Something went wrong in batch query")

        random_id = "".join(random.choices("0123456789", k=8))
        batch_id = f"dumbar_batch_{random_id}"
        batch_obj = {
            "status": "in_progress",
            "batch_name": batch_name,
            "requests": requests,
        }
        _ = redis_client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )
        return batch_id

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            raise Exception("Something went wrong in parsing batch results")

        batch_obj = self._get_batch_obj(batch_id)
        requests: list[dict[str, Any]] = batch_obj["requests"]
        batch_results: list[BatchResult] = []
        for req in requests:
            custom_id = cast(str, req["custom_id"])
            if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        llm_output="",
                        error_message="Dumbmar queried unsuccessfully",
                    )
                )
            else:
                batch_results.append(
                    BatchResult(
                        custom_id=custom_id,
                        llm_output="Dumbmar queried successfully",
                    )
                )
        return batch_results

    def _get_batch_obj(self, batch_id: str) -> dict[str, Any]:
        batch_obj: dict[str, Any] = json.loads(
            redis_client.get(f"dummy_batch:{batch_id}")  # type: ignore
        )
        return batch_obj

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        batch_obj = self._get_batch_obj(batch_id)
        return len(batch_obj["requests"])

    @override
    async def cancel_batch_request(self, batch_id: str):
        batch_obj = self._get_batch_obj(batch_id)
        batch_obj["status"] = "cancelled"
        _ = redis_client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        batch_obj = self._get_batch_obj(batch_id)
        if random.random() < FAIL_RATE and "evaluator" not in self._root.model_name:
            batch_obj["status"] = "failed"
        elif batch_obj["status"] != "cancelled":
            batch_obj["status"] = random.choice(["completed", "in_progress"])

        _ = redis_client.setex(
            f"dummy_batch:{batch_id}", BATCH_EXP, json.dumps(batch_obj)
        )
        return batch_obj["status"]

    @override
    @classmethod
    def is_status_completed(cls, batch_status: str) -> bool:
        return True

    @override
    @classmethod
    def is_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == "cancelled"

    @override
    @classmethod
    def is_status_failed(cls, batch_status: str) -> bool:
        return batch_status == "failed"


class DummyAIModel(LLM):
    @override
    def get_client(self) -> object:
        raise NotImplementedError("DummyAI does not support client")

    def __init__(
        self,
        model_name: str,
        provider: Literal["vals"] = "vals",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)
        self.batch: LLMBatchMixin | None = (
            DummyAIBatchMixin(self) if self.supports_batch else None
        )

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
            "messages": [
                f"Dumbmar dummy message: {prompt} --- {len(images)} images, {len(files)} files"
            ],
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
        body = self.query_body(prompt, images=images, files=files, **kwargs)

        if random.random() < FAIL_RATE and "evaluator" not in self.model_name:
            raise Exception("Dumbmar couldn't retrieve output.")

        return body["messages"][0], QueryResultMetadata(
            in_tokens=0,
            out_tokens=0,
        )
