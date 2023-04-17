# Gym-preCICE Contribution Guidelines

The following forms of contributions are welcome:

- Bug reports
- Pull requests for bug fixes
- Documentation improvements
- Features

Please note that if you have any contributions related to `environments`, they must be submitted to our [gymprecice-tutorials](https://github.com/gymprecice/gymprecice-tutorials) epository.


# Development

This section contains technical guidelines and tips intended for contributors.

## Type checking

The project uses `pyright` to check types.
To type check locally, install `pyright` per official [instructions](https://github.com/microsoft/pyright#command-line).
It's configuration lives within `pyproject.toml`. It includes list of included and excluded files currently supporting type checks.
To run `pyright` for the project, run the pre-commit process (`pre-commit run --all-files`) or `pyright --project=pyproject.toml`


## Git hooks

The CI will run several checks on the new code pushed to the Gymnasium repository. These checks can also be run locally without waiting for the CI by following the steps below:

1. [install `pre-commit`](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit.
The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`.
**Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.

Additionally, for pull requests, the project runs a number of tests for the whole project using [pytest](https://docs.pytest.org/en/latest/getting-started.html#install-pytest).
These tests can be run locally with `pytest` in the root folder.

## Docstrings

Pydocstyle has been added to the pre-commit process such that all new functions follow the [google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
All new functions require either a short docstring, a single line explaining the purpose of a function
or a multiline docstring that documents each argument and the return type (if there is one) of the function.
In addition, new file and class require top docstrings that should outline the purpose of the file/class.
For classes, code block examples can be provided in the top docstring and not the constructor arguments.

To check your docstrings are correct, run `pre-commit run --all-files` or `pydocstyle --source --explain --convention=google`.
If all docstrings that fail, the source and reason for the failure is provided.
