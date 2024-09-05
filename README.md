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

- `example_suites/` - These are example JSON files, of the sort that you may use for `vals suite create <example.json>`.
- `example_scripts/` - These are example Python scripts that leverage the sdk.
- `example_run_configs/` - When you start a run, you can optionally provide more parameters (more details below). These are examples of parameter files.

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

- `vals suite`: relating to creating / updating tests and suites
- `vals run`: relating to creating and querying runs and run results.

#### Create Test Suite

Create a test suite from command line

```
vals suite create --interactive
```

Create a test suite from JSON file

```
vals suite create ./example_suite.json
```

The `vals suite create` commands will produce a link to the created test suite.

#### Run

The test suite id embedded at the end of the URL is used in run requests.

Start a new run:

```
vals run start
```

Or specify the suite id (the last part of the)

```
vals run start -s <suite id>
```

Full details of the CLI usage can be found in our documentation at [https://www.platform.vals.ai/docs/index.html#/cli](https://www.platform.vals.ai/docs/index.html#/cli)

## SDK Usage

All of the functionality that is in the CLI can also be accessed via Python import.
For example - to create a suite within Python, you could do the following:

```python

from vals.sdk.suite import create_suite

create_suite(
  {
    "title": "Tax Questions",
    "description": "Questions relating to tax incentives",
    "tests": [
      {
        "input_under_test": "What is QSBS?",
        "checks": {
          {
            "operator": "includes",
            "criteria": "C Corporation"
          }
        }
      }
    ]
  }
)
```

## Passing Functions to SDK

The SDK also includes features the CLI does not. This includes the
ability to test _any_ arbitrary Python function, rather than just the base model.
Therefore, even if you have very complicated internal logic for parsing, chunking, prompt-chaining, etc.
it still can be evaluated with the Vals platform.

First, define a function with the following signature:

```python
def my_model(input_under_test: str) -> str:
  llm_output = "..."
  return llm_output
```

Then, create a suite in the platform and run it like the following:

```python
run_id = run_evaluations(
    # Replace with the link suite link
    "https://platform.vals.ai/view?test_suite_id=xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx",
    test_function,
)
```

There is a full, working example in `examples/examples_scripts/sdk_example.py`, and additional documentation in our docs: [https://www.platform.vals.ai/docs/index.html#/sdk](https://www.platform.vals.ai/docs/index.html#/sdk)
