# Vals AI CLI and SDK Tool

## Install

```
pip install valsai
```

## Overview

This Python package contains two items - a CLI tool to use Vals from the command line, and an SDK to
use Vals from Python code.

All code is contained within `vals/`, separated by CLI and SDK. Note that many of the CLI functions
are just thin wrappers around the SDK.

## Examples

We've provided a set of examples in `examples/`. They are organized as follows:

- `suites/` - These are example JSON files, of the sort that you may use for `vals suite create <example.json>`.
- `scripts/` - These are example Python scripts that leverage the sdk.

## Authentication

Make an account at [https://platform.vals.ai](https://platform.vals.ai) and confirm your email.

Then, go to [https://platform.vals.ai/auth](https://platform.vals.ai/auth) and create an API key.

If you are using the command line, you should set the following:

```
export VALS_API_KEY=<YOUR API KEY>
```

If you're using the EU instance of vals, you should also set `export VALS_REGION=europe`. Include these in your .zshrc / .bashrc to make them permanent.

See the documentation for passing in your API key directly via Python code for SDK usage.

## CLI Usage

The CLI is run as follows:

```
$ vals
```

Use the `--help` flag at the top and subcommand level for guidance.

Commands must be run from the pip environment the cli was installed in. Commands are split up into subcommands. Currently there are three main subcommands:

- `vals suite --help`: relating to creating / updating tests and suites
- `vals run --help`: relating to creating and querying runs and run results
- `vals project --help`: relating to listing and managing projects

### Project Support

The SDK and CLI now support working with multiple projects within your organization. You can:

1. **Set a default project ID** via environment variable:
   ```bash
   export VALS_PROJECT_ID="your-project-id"
   ```

2. **Specify project ID** when creating or listing resources:
   ```bash
   # List suites in a specific project
   vals suite list --project-id "project-123"
   
   # Create a suite in a specific project  
   vals suite create suite.json --project-id "project-123"
   
   # List runs in a specific project
   vals run list --project-id "project-123"
   ```

3. **List available projects**:
   ```bash
   # List all projects in your organization
   vals project list
   
   # Show the default project
   vals project default
   ```

The priority order for project selection is:
1. Explicit `--project-id` flag
2. `VALS_PROJECT_ID` environment variable
3. Organization's default project

Full documentation of the CLI usage can be found in our documentation at [https://www.platform.vals.ai/docs/index.html#/cli](https://www.platform.vals.ai/docs/index.html#/cli)

## SDK Usage

All of the functionality that is in the CLI can also be accessed via Python functions,
as well as features only available in the SDK.

### Project Support in SDK

The SDK provides full support for working with projects:

```python
from vals import Suite, Run, Project
import asyncio

async def main():
    # List all projects in your organization
    projects = await Project.list_projects()
    for project in projects:
        print(f"{project.name} ({project.id}) - Default: {project.is_default}")
    
    # Get the default project
    default_project = await Project.get_default_project()
    print(f"Default project: {default_project.name}")
    
    # Create a suite in a specific project
    suite = Suite(
        title="My Test Suite",
        description="Test suite with project support",
        project_id="project-123"  # Optional - defaults to organization's default project
    )
    await suite.create()
    
    # List suites in a specific project
    suites = await Suite.list_suites(project_id="project-123")
    
    # List runs in a specific project
    runs = await Run.list_runs(project_id="project-123")

asyncio.run(main())
```

You can also use the `VALS_PROJECT_ID` environment variable to set a default project for all SDK operations:

```python
import os
os.environ['VALS_PROJECT_ID'] = "project-123"

# Now all operations will use this project by default
suite = Suite(title="My Suite", description="Uses project from env var")
await suite.create()  # Creates in project-123
```

See usage documentation in our docs: [https://www.platform.vals.ai/docs/index.html#/sdk](https://www.platform.vals.ai/docs/index.html#/sdk)

# Development

## Local installation

Run the following command to install it locally. The -e flag is optional, but recommended, as it allows you to make changes to the code without reinstalling. The dev flag is required for dev-only depedencies.

```bash
pip install -e ".[dev]"
```

## Running Codegen

Add additional .graphql files to `vals/graphql`, then run the following command:

```
ariadne-codegen --config codegen-config.toml
```

NOTE: This will _overwrite_ anything in the `vals/graphql_client` directory.
