import asyncio
import json
from typing import Literal

from pydantic import BaseModel, PrivateAttr

from vals.graphql_client import Client
from vals.sdk.util import dot_animation, get_ariadne_client, upload_file


class CustomMetric(BaseModel):
    """Represents a custom metric in the vals platform."""

    # server only
    _id: str | None = None
    _archived: bool | None = None
    _file_id: str | None = None

    # user can provide
    project_id: str
    name: str
    python_file_path: str | Literal["vals"]
    description: str | None = None

    _client: Client = PrivateAttr(default_factory=get_ariadne_client)

    def __init__(
        self,
        name: str,
        python_file_path: str | Literal["vals"],
        project_id: str = "default-project",
        description: str | None = None,
    ):
        """
        Create a new custom metric locally. Not uploaded to the server yet.
        """
        super().__init__(
            project_id=project_id,
            name=name,
            python_file_path=python_file_path,
            description=description,
        )

    @property
    def id(self) -> str:
        if self._id is None:
            raise Exception("Custom metric does not exist.")
        return self._id

    @classmethod
    async def from_id(cls, id: str) -> "CustomMetric":
        """
        Get a custom metric from the server by ID
        """
        client = get_ariadne_client()
        result = await client.get_custom_metric(id=id)

        custom_metric = cls(
            name=result.custom_metric.name,
            description=result.custom_metric.description,
            python_file_path="vals",
        )
        custom_metric._id = result.custom_metric.id
        custom_metric.project_id = result.custom_metric.project.slug
        custom_metric._archived = result.custom_metric.archived
        custom_metric._file_id = result.custom_metric.file_id

        return custom_metric

    async def pull(self) -> None:
        """
        Refresh this CustomMetric instance with the latest data from the server.
        """
        if not self._id:
            raise Exception("Custom metric does not exist.")

        updated = await self.__class__.from_id(self._id)
        self.__dict__.update(updated.__dict__)

    async def create(self) -> str:
        """
        Create this custom metric on the server.
        Returns the ID of the custom metric.
        """
        if self._id:
            raise Exception("Custom metric already exists.")

        self._file_id = await upload_file(self.python_file_path, temporary=True)

        result = await self._client.upsert_custom_metric(
            id=None,
            project_id=self.project_id,
            name=self.name,
            description=self.description,
            file_id=self._file_id,
            update_past=False,
        )
        metric = result.upsert_custom_metric
        if not metric:
            raise Exception("Unable to create custom metric.")

        self._id = str(metric.metric.id)
        return self._id

    async def update(self, update_past: bool = False) -> None:
        """
        Update this custom metric on the server
        """
        if not self._id or not self._file_id:
            raise Exception("Custom metric does not exist.")
        if self._archived:
            raise Exception("Custom metric is archived.")

        if self.python_file_path != "vals":
            self.file_id = await upload_file(self.python_file_path)

        result = await self._client.upsert_custom_metric(
            id=self._id,
            project_id=self.project_id,
            name=self.name,
            description=self.description,
            file_id=self._file_id,
            update_past=update_past,
        )
        if not result.upsert_custom_metric:
            raise Exception("Unable to update custom metric.")

    async def archive(self) -> None:
        """
        Archive this custom metric on the server
        """
        if not self._id:
            raise Exception("Custom metric does not exist.")
        if self._archived:
            raise Exception("Custom metric is already archived.")

        result = await self._client.set_archived_status_custom_metrics([self._id], True)
        if not result.set_archived_status_custom_metrics:
            raise Exception("Unable to archive custom metric.")
        self._archived = True

    async def restore(self) -> None:
        """
        Restore this custom metric on the server
        """
        if not self._id:
            raise Exception("Custom metric does not exist.")
        if not self._archived:
            raise Exception("Custom metric is not archived.")

        result = await self._client.set_archived_status_custom_metrics(
            [self._id], False
        )
        if not result.set_archived_status_custom_metrics:
            raise Exception("Unable to restore custom metric.")
        self._archived = False

    async def run(self, run_id: str) -> str:
        """
        Run this custom metric on a run.
        Starts the custom metric task and polls until completion.
        """
        if not self._id or not self._file_id:
            raise Exception("This custom metric does not exist.")

        result = await self._client.start_custom_metric_task(
            run_id=run_id,
            file_id=self._file_id,
        )
        if not result.start_custom_metric_task:
            raise Exception("Unable to run custom metric.")

        message_id = result.start_custom_metric_task.message_id
        if not message_id:
            raise Exception("Unable to poll custom metric.")

        stop_event = asyncio.Event()
        animation_task = asyncio.create_task(dot_animation(stop_event))
        try:
            while True:
                result = await self._client.poll_custom_metric_task(message_id)
                if not result.poll_custom_metric_task:
                    raise Exception("Unable to poll custom metric.")

                response = result.poll_custom_metric_task
                match response.status:
                    case "pending":
                        await asyncio.sleep(3)
                        continue
                    case "success":
                        try:
                            if not response.result:
                                raise Exception("No result found.")
                            response_dict = json.loads(response.result)
                            value = response_dict[self._file_id][run_id]
                            return f"{float(value):.2f}"
                        except Exception as e:
                            raise Exception(f"Failed to parse result: {e}")
                    case "fail":
                        raise Exception(
                            f"Error running custom metric: {response.error}"
                        )
                    case _:
                        raise Exception(f"Unknown status: {response.status}")
        finally:
            stop_event.set()
            await animation_task
