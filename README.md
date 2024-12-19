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

Then, go to [https://platform.vals.ai/auth](https://platform.vals/ai/auth) and create an API key.

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

Commands must be run from the pip environment the cli was installed in. Commands are split up into subcommand. Currently there are two subcommands:

- `vals suite --help`: relating to creating / updating tests and suites
- `vals run --help`: relating to creating and querying runs and run results.

Full documentation of the CLI usage can be found in our documentation at [https://www.platform.vals.ai/docs/index.html#/cli](https://www.platform.vals.ai/docs/index.html#/cli)

## SDK Usage

All of the functionality that is in the CLI can also be accessed via Python functions,
as well as features only available in the SDK.

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
