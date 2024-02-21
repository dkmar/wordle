# wordle

[![PyPI](https://img.shields.io/pypi/v/wordle.svg)](https://pypi.org/project/wordle/)
[![Changelog](https://img.shields.io/github/v/release/dkmar/wordle?include_prereleases&label=changelog)](https://github.com/dkmar/wordle/releases)
[![Tests](https://github.com/dkmar/wordle/workflows/Test/badge.svg)](https://github.com/dkmar/wordle/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dkmar/wordle/blob/master/LICENSE)

Utility for solving and exploring wordle puzzles.

## Installation

Install this tool using `pip`:

    pip install wordle-solver

## Usage

For help, run:

    wordle --help

You can also use:

    python -m wordle --help

## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:

    cd wordle
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
