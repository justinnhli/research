#!/bin/bash

# determine appropriate executable
if [ -e "$HOME/.venv/research/bin/py.test" ]; then
    pytest="$HOME/.venv/research/bin/py.test" 
elif command -v py.test >/dev/null 2>&1; then
    pytest="py.test"
else
    echo 'Cannot find py.test; quitting...'
    exit 1
fi

# change to current directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# go up once to the project root directory
cd ..
# run pytest
"$pytest" --verbose --cov=research/ --cov-config .coveragerc tests
