#!/bin/bash

# determine appropriate executable
if [ -e "$HOME/.venv/research/bin/py.test" ]; then
    pydocstyle="$HOME/.venv/research/bin/pydocstyle" 
elif command -v py.test >/dev/null 2>&1; then
    pydocstyle="pydocstyle"
else
    echo 'Cannot find pydocstyle; quitting...'
    exit 1
fi

# change to current directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# go up once to the project root directory
cd ..
# run pydocstyle
"$pydocstyle" *.py research/
