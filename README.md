# Vals AI CLI Tool

## Install

```
pip install valsai
```

### _Internal: Install during development_

```
cd legal-evaluator/cli
pip install -e .
```

## Authentication

Make an account at [https://platform.vals.ai](https://platform.vals.ai) and confirm your email.

Log in locally with:

```commandline
vals login
```

This will create an auth token for you locally which is used in future commands. You will be prompted to re-login after the token expires (30 days).

## Overall Usage

The CLI is run as follows:

```
$ vals
```

Use the `--help` flag at the top and subcommand level for guidance.

Commands must be run from the pip environment the cli was installed in. Commands are split up into subcommand. Currently there are two subcommands:

- `vals suite`: relating to creating / updating tests and suites
- `vals run`: relating to creating and querying runs and run results.

## Create Test Suite

Create a test suite from command line

```
vals suite create --interactive
```

Create a test suite from JSON file

```
vals suite create ./example_suite.json
```

The `vals suite create` commands will produce a link to the created test suite.

## Run

The test suite id embedded at the end of the URL is used in run requests.

Start a new run:

```
vals run start [suite_id]
```

## Frozen LLM Outputs Workflow

In some cases, it is useful to be able to shortcut the model API call to use a hard-coded output for evaluation. This can be used to evaluate the evaluator.

To provide outputs, modify the JSON test suite file to include the string attribute `fixed_output`. See the example file `example_with_fixed_output.json`.

Then, create the test suite as usual.

```
vals suite create ./example_with_fixed_output.json
```

By default, running this test suite (via the website or CLI command) will still use the API to generate outputs. The `--use_fixed_output` flag must be provided.

Example:

```
vals suite run [suite_id] --use_fixed_output
```

## Example JSON files:

`example_suite.json`

```json
{
  "title": "Test title",
  "description": "Test description",
  "tests": [
    {
      "input_under_test": "Waht is QSBS",
      "checks": [
        {
          "operator": "includes",
          "criteria": "C Corporation"
        },
        {
          "operator": "excludes",
          "criteria": "S Corporation"
        }
      ]
    }
  ]
}
```

`example_suite_with_output.json`

```json
{
  "title": "Test title",
  "description": "Test description",
  "tests": [
    {
      "input_under_test": "What is the meaning of life?",
      "checks": [
        {
          "operator": "includes",
          "criteria": "42"
        },
        {
          "operator": "excludes",
          "criteria": "myth of sisyphus"
        }
      ],
      "fixed_output": "The meaning of life is 42"
    }
  ]
}
```
