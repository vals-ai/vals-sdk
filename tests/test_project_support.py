"""Tests for project support functionality in the vals SDK."""

from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vals.sdk.project import Project
from vals.sdk.suite import Suite
from vals.sdk.run import Run


class TestProjectClass:
    """Test the Project class functionality."""

    @pytest.mark.asyncio
    async def test_list_projects(self):
        """Test listing projects."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.projects_with_count.projects = [
            SimpleNamespace(
                id="proj1", name="Project 1", slug="project-1", is_default=False
            ),
            SimpleNamespace(
                id="proj2", name="Default", slug="default", is_default=True
            ),
        ]
        mock_client.list_projects.return_value = mock_response

        with patch("vals.sdk.project.get_ariadne_client", return_value=mock_client):
            projects = await Project.list_projects(limit=10, offset=0)

        assert len(projects) == 2
        assert projects[0].id == "proj1"
        assert projects[0].name == "Project 1"
        assert projects[0].slug == "project-1"
        assert not projects[0].is_default

        assert projects[1].id == "proj2"
        assert projects[1].name == "Default"
        assert projects[1].slug == "default"
        assert projects[1].is_default

        mock_client.list_projects.assert_called_once_with(limit=10, offset=0, search="")


class TestSuiteProjectSupport:
    """Test project support in the Suite class."""

    def test_suite_with_project_id(self):
        """Test creating a Suite instance with project_id."""
        suite = Suite(
            title="Test Suite",
            description="A test suite",
            project_id="test-project-123",
        )

        assert suite.project_id == "test-project-123"
        assert suite.title == "Test Suite"
        assert suite.description == "A test suite"

    def test_suite_without_project_id(self):
        """Test creating a Suite instance without project_id."""
        suite = Suite(title="Test Suite", description="A test suite")

        assert suite.project_id == "default-project"
        assert suite.title == "Test Suite"

    @pytest.mark.asyncio
    async def test_list_suites_with_project_id(self):
        """Test listing suites with project_id filter."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.test_suites_with_count.test_suites = []
        mock_client.get_test_suites_with_count.return_value = mock_response

        with patch("vals.sdk.suite.get_ariadne_client", return_value=mock_client):
            await Suite.list_suites(
                limit=10, offset=0, search="", project_id="test-proj"
            )

        mock_client.get_test_suites_with_count.assert_called_once_with(
            limit=10, offset=0, search="", project_id="test-proj"
        )

    @pytest.mark.asyncio
    async def test_create_suite_with_project_id(self):
        """Test creating a suite with project_id."""
        suite = Suite(
            title="Test Suite",
            description="A test suite",
            project_id="test-project-123",
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.update_test_suite.test_suite.id = "suite-123"
        mock_client.create_or_update_test_suite.return_value = mock_response
        mock_client.update_global_checks.return_value = MagicMock()

        suite._client = mock_client

        await suite.create()

        mock_client.create_or_update_test_suite.assert_called_once_with(
            "0", "Test Suite", "A test suite", "test-project-123"
        )


class TestRunProjectSupport:
    """Test project support in the Run class."""

    @pytest.mark.asyncio
    async def test_list_runs_with_project_id(self):
        """Test listing runs with project_id filter."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.runs_with_count.run_results = []
        mock_client.list_runs.return_value = mock_response

        with patch("vals.sdk.run.get_ariadne_client", return_value=mock_client):
            await Run.list_runs(
                limit=10,
                offset=0,
                suite_id=None,
                show_archived=False,
                search="",
                project_id="test-proj",
            )

        mock_client.list_runs.assert_called_once_with(
            limit=10,
            offset=0,
            suite_id=None,
            archived=False,
            search="",
            project_id="test-proj",
        )


class TestCLIProjectSupport:
    """Test CLI command support for projects."""

    def test_cli_imports_project_group(self):
        """Test that the project group is imported in CLI main."""
        from vals.cli.main import cli

        # Check that project_group is registered
        command_names = [cmd.name for cmd in cli.commands.values()]
        assert "project" in command_names

    @pytest.mark.asyncio
    async def test_project_list_command(self):
        """Test the project list CLI command."""
        from vals.cli.project import list_command_async

        mock_projects = [
            Project(id="1", name="Project 1", slug="project-1", is_default=False),
            Project(id="2", name="Default", slug="default", is_default=True),
        ]

        with patch(
            "vals.sdk.project.Project.list_projects", return_value=mock_projects
        ):
            # This would normally print to console, just testing it doesn't error
            await list_command_async(limit=10, offset=1)


class TestEndToEndProjectSupport:
    """End-to-end tests for project support."""

    @pytest.mark.asyncio
    async def test_suite_create_list_with_project(self):
        """Test creating and listing suites with project support."""
        # This is a mock end-to-end test showing the flow
        project_id = "test-project-123"

        # Create a suite with project_id
        suite = Suite(
            title="E2E Test Suite", description="End-to-end test", project_id=project_id
        )

        assert suite.project_id == project_id

        # Mock listing suites filtered by project
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.test_suites_with_count.test_suites = [
            MagicMock(
                id="suite-1",
                title="E2E Test Suite",
                description="End-to-end test",
                created="2024-01-01",
                creator="test-user",
                last_modified_by="test-user",
                last_modified_at="2024-01-01",
                folder=None,
            )
        ]
        mock_client.get_test_suites_with_count.return_value = mock_response

        with patch("vals.sdk.suite.get_ariadne_client", return_value=mock_client):
            suites = await Suite.list_suites(project_id=project_id)

        assert len(suites) == 1
        assert suites[0].title == "E2E Test Suite"

        # Verify the client was called with the correct project_id
        mock_client.get_test_suites_with_count.assert_called_with(
            limit=50, offset=0, search="", project_id=project_id
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
