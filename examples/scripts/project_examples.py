import asyncio

from vals import Project, Run, Suite
from vals.sdk.types import Check, Test


async def list_all_projects():
    """List all projects in the organization."""
    print("\n=== Listing All Projects ===")
    projects = await Project.list_projects()

    for project in projects:
        default_marker = " (DEFAULT)" if project.is_default else ""
        print(f"- {project.name} [{project.slug}] - ID: {project.id}{default_marker}")

    return projects


async def create_suite_in_project(project_id: str):
    """Create a test suite in a specific project."""
    print(f"\n=== Creating Suite in Project {project_id} ===")

    suite = Suite(
        title="Project Example Suite",
        description="A suite created to demonstrate project support",
        project_id=project_id,
        tests=[
            Test(
                input_under_test="What is 2+2?",
                checks=[Check(operator="includes", criteria="4")],
            )
        ],
    )

    await suite.create()
    print(f"Created suite with ID: {suite.id}")

    return suite


async def list_suites_in_project(project_id: str):
    """List all suites in a specific project."""
    print(f"\n=== Listing Suites in Project {project_id} ===")

    suites = await Suite.list_suites(project_id=project_id, limit=5)

    if not suites:
        print("No suites found in this project")
    else:
        for suite in suites:
            print(f"- {suite.title} (ID: {suite.id})")

    return suites


async def list_runs_in_project(project_id: str):
    """List all runs in a specific project."""
    print(f"\n=== Listing Runs in Project {project_id} ===")

    runs = await Run.list_runs(project_id=project_id, limit=5)

    if not runs:
        print("No runs found in this project")
    else:
        for run in runs:
            status = run.status.value if run.status else "unknown"
            print(f"- Run {run.name} (ID: {run.id}) - Status: {status}")

    return runs


async def main():
    """Main function to run all examples."""
    print("Vals SDK Project Support Examples")
    print("=================================")

    # List all projects
    projects = await list_all_projects()

    if len(projects) > 1:
        # Use the first non-default project for demonstration
        project_to_use = next((p for p in projects if not p.is_default), projects[0])

        # Create a suite in the project
        await create_suite_in_project(project_to_use.id)

        # List suites in the project
        await list_suites_in_project(project_to_use.id)

        # List runs in the project
        await list_runs_in_project(project_to_use.id)
    else:
        print(
            "\nNote: Only one project found. Create additional projects to see project filtering in action."
        )

    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
