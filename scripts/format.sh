#!/bin/bash

# Runs black and isort formatting, then flake8 and mypy linting recursively on given directory.

path=${1:-"src tests"}
line_length=100

black $path -l $line_length;
isort $path --profile black;
flake8 $path --max-line-length $line_length --ignore E203;
mypy $path --ignore-missing-imports --warn-unused-ignores --no-site-packages;
