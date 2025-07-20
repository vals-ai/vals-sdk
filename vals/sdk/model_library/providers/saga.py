import json
from typing import Literal

from httpx._client import AsyncClient
from typing_extensions import override

from vals.sdk.model_library.base import LLM, FileInput, LLMConfig, QueryResult


class SagaModel(LLM):
    _client: AsyncClient | None = None

    @override
    def get_client(self) -> AsyncClient:
        # use the same client for all instances
        if not SagaModel._client:
            SagaModel._client = AsyncClient()
        return SagaModel._client

    def __init__(
        self,
        model_name: str,
        provider: Literal["concide"] = "concide",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__("concide", model_name, config=config)

    @override
    async def _query_impl(
        self,
        prompt: str,
        *,
        images: list[FileInput],
        files: list[FileInput],
        **kwargs: object,
    ) -> QueryResult:
        HOST = "https://dev.sagalegal.io/api"

        sessionUrl = f"{HOST}/chat-sessions"
        chatUrl = f"{HOST}/chat"

        headers = {
            "Authorization": f"Api-Key {'settings'}",
            "Content-Type": "application/json",
        }

        client = self.get_client()
        response = await client.get(sessionUrl, headers=headers)

        if response.status_code < 400:
            sessionId: str | None = response.json()[0].get("id", None)
            if not sessionId:
                raise Exception("No sessionId returned")
            self.logger.info("Created chat session: %s", sessionId)
        else:
            self.logger.error("Error %s: %s", response.status_code, response.text)
            raise Exception(response.text)

        data = {
            "lastMessage": prompt,
            "model": self.model_name,
            "sessionId": sessionId,
        }

        async with client.stream(
            "POST", chatUrl, headers=headers, json=data
        ) as response:
            result = ""
            if response.status_code == 200:
                async for chunk in response.aiter_lines():
                    if chunk:
                        if chunk.strip() == "[DONE]":
                            break
                        if chunk.startswith("data:"):
                            try:
                                json_data = json.loads(chunk[5:].strip())
                                result += json_data.get("response", "")
                            except json.JSONDecodeError:
                                continue
                self.logger.info("Result: %s", result)
            else:
                self.logger.error(
                    "Request failed with status code %s", response.status_code
                )

            return result, None
