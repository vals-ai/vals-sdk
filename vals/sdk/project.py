"""Project management functionality for vals SDK."""

from typing import List
from pydantic import BaseModel
from vals.sdk.util import get_ariadne_client


class Project(BaseModel):
    """Represents a project in the vals platform."""

    id: str
    name: str
    slug: str
    is_default: bool

    @classmethod
    async def list_projects(
        cls, limit: int = 50, offset: int = 0, search: str = ""
    ) -> List["Project"]:
        """
        List all projects in the organization.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip
            search: Search query to filter projects
        Returns:
            List of Project objects
        """
        client = get_ariadne_client()
        result = await client.list_projects(limit=limit, offset=offset, search=search)

        projects = []
        for proj in result.projects_with_count.projects:
            projects.append(
                cls(
                    id=proj.id,
                    name=proj.name,
                    slug=proj.slug,
                    is_default=proj.is_default,
                )
            )
        return projects
