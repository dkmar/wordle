# wordle

[![PyPI](https://img.shields.io/pypi/v/wordle.svg)](https://pypi.org/project/wordle/)
[![Changelog](https://img.shields.io/github/v/release/dkmar/wordle?include_prereleases&label=changelog)](https://github.com/dkmar/wordle/releases)
[![Tests](https://github.com/dkmar/wordle/workflows/Test/badge.svg)](https://github.com/dkmar/wordle/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dkmar/wordle/blob/master/LICENSE)

Utility for solving and exploring wordle puzzles.

## Usage

Play interactively:

    wordle play [-h]
<img width="787" alt="image" src="https://github.com/dkmar/wordle/assets/31838716/f0100956-052b-47f9-aed1-729bf7136a90">

For help, run:

    wordle --help

## Performance:
<img width="516" alt="image" src="https://github.com/dkmar/wordle/assets/31838716/8e260d66-8312-45f7-940a-fe39cb023344">

(per this [online leaderboard](https://freshman.dev/wordle/leaderboard?mode=hard) for hard-mode using the original 2315-word answer set)
    
## Installation

Install this tool using `pip`:

    pip install wordle-solver

## Development

To contribute to this tool, first checkout the code. Then create a new virtual environment:

    cd wordle
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
