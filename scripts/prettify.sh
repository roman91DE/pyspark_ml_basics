#!/bin/bash

# This file is symlinked to .git/hooks/pre-commit

python_bin=$(which python3)

for FILE in ./src/*.py; do
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
