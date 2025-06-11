# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation

```bash
# Install locally for development (recommended with -e flag for live editing)
pip install -e .
pip install -e ".[dev]"
```

### GraphQL Code Generation

```bash
# Regenerate GraphQL client code (overwrites vals/graphql_client/)
ariadne-codegen --config codegen-config.toml
```

### Testing

```bash
# Run tests (uses pytest)
python -m pytest tests/
# Run specific test file
python -m pytest tests/test_project_support.py
```

## Architecture Overview

This is a Python package providing both a CLI tool and SDK for the Vals AI platform. The codebase is split into two main components:

### 1. SDK (`vals/sdk/`)

- **Core Models**: `Suite`, `Run`, `Project` classes for managing test suites, runs, and projects
- **Authentication**: `auth.py` handles API key authentication and credentials
- **GraphQL Integration**: Generated client code in `vals/graphql_client/` (auto-generated, do not edit directly)
- **Project Support**: Multi-project functionality with project ID resolution in `util.py`

### 2. CLI (`vals/cli/`)

- **Entry Point**: `main.py` defines the main CLI with subcommands
- **Subcommands**: `suite.py`, `run.py`, `project.py` provide CLI interfaces
- **Error Handling**: Custom exception handling wrapper in `main.py`

### GraphQL Schema Management

- GraphQL queries/mutations are stored in `vals/graphql/` organized by resource type
- The `ariadne-codegen` tool generates Python client code automatically
- Configuration is in `codegen-config.toml`

### Project ID Resolution Priority

1. Explicit `--project-id` flag in CLI or `project_id` parameter in SDK
2. `VALS_PROJECT_ID` environment variable  
3. Organization's default project (automatically resolved)

## Key Patterns

### Async/Await

The SDK is built with async/await patterns. All SDK methods that interact with the API are async.

### Authentication

Set `VALS_API_KEY` environment variable. For EU region, also set `VALS_REGION=europe`.

### Error Handling

CLI uses custom exception handling wrapper. SDK methods raise appropriate exceptions that are caught and formatted by the CLI.
