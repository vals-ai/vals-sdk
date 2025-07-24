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

Make an account at [platform.vals.ai](https://platform.vals.ai) and confirm your email.

Then, go to [platform.vals.ai/project/default-project/settings/api-keys](https://platform.vals.ai/project/default-project/settings/api-keys) and create an API key. If using a different project, navigate to that project's settings instead.

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

Full documentation of the CLI usage can be found in our documentation at [docs.vals.ai/cli_sdk/cli](https://docs.vals.ai/cli_sdk/cli)

## SDK Usage

All of the functionality that is in the CLI can also be accessed via Python functions,
as well as features only available in the SDK.

See usage documentation in our docs: [docs.vals.ai/cli_sdk/sdk](https://docs.vals.ai/cli_sdk/sdk)

# Development

## Local installation

The sdk uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
A Makefile is provided to help with development.

To install dependencies, run:

```bash
make install
```
Run the following command to install it locally. The -e flag is optional, but recommended, as it allows you to make changes to the code without reinstalling. The dev flag is required for dev-only depedencies.

```bash
pip install -e ".[dev]"
```

If using the sdk in a project, to install the sdk in editable mode (apply local sdk changes without reinstalling), in your project directory run:

```bash
uv pip install -e /path/to/vals-sdk
```
or

```bash
pip install -e /path/to/vals-sdk
```

## Additional Makefile commmands
```
make install        Install dependencies for development
make test           Run tests
make style          Lint & Format
make typecheck      Typecheck
make codegen        Generate GraphQL client
```

## Running Codegen

Add additional .graphql files to `vals/graphql`, then run the following command:

```
make codegen
```
or
```
ariadne-codegen --config codegen-config.toml
```

NOTE: This will _overwrite_ anything in the `vals/graphql_client` directory.
