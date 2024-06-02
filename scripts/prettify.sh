#!/bin/bash

# This file is symlinked to .git/hooks/pre-commit

python_bin=$(which python3)

for FILE in ./src/*.py; do
    echo "Checking File $FILE"
    "$python_bin" -m isort "$FILE"
    "$python_bin" -m black "$FILE"
    "$python_bin" -m mypy "$FILE"
done
