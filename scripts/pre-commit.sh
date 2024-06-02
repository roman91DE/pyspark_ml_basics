#!/bin/bash

# This script is symlinked to .git/hooks/pre-commit

echo "Running pre-commit hook"

python_bin=$(which python3)
echo "Using Python binary at $python_bin"

# Get the list of Python files
files=$(find ./src -name "*.py")

# Check if any Python files are found
if [ -z "$files" ]; then
    echo "No Python files found to check."
    exit 1
fi

for FILE in $files; do
    echo "Checking File $FILE"

    # Run isort and check for errors
    "$python_bin" -m isort "$FILE"
    if [ $? -ne 0 ]; then
        echo "isort failed on $FILE"
        exit 1
    fi

    # Run black and check for errors
    "$python_bin" -m black "$FILE"
    if [ $? -ne 0 ]; then
        echo "black failed on $FILE"
        exit 1
    fi

    # Run mypy and check for errors
    "$python_bin" -m mypy "$FILE"
    if [ $? -ne 0 ]; then
        echo "mypy failed on $FILE"
        exit 1
    fi
done
