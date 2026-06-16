#!/usr/bin/env bash
# This script formats the code of this package

echo "Formatting import statements..."
ruff check --fix --config=../pyproject.toml ..

# echo "Formating docstrings..."
# docformatter --in-place --black --recursive ..

echo "Formatting source code..."
ruff format --config=../pyproject.toml ..