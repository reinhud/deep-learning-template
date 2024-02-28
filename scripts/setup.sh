#!/bin/sh

# Install pre-commit hooks into .git
echo "Installing pre-commit hooks..."
poetry run pre-commit install

# Print information about torch installation
echo "Printing startup info..."
poetry run python src/utils/visualizations/print_torch_info.py

# Running tests to verify everything is working
# echo "Running tests..."
# poetry run pytest --cov=src --disable-pytest-warnings tests/
