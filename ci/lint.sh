#!/bin/bash

# determine appropriate executable
if [ -e "$HOME/.venv/research/bin/py.test" ]; then
    pylint="$HOME/.venv/research/bin/pylint" 
elif command -v py.test >/dev/null 2>&1; then
    pylint="pylint"
else
    echo 'Cannot find pylint; quitting...'
    exit 1
fi

# change to current directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# go up once to the project root directory
cd ..
# run pylint
"$pylint" --min-similarity-lines=8 *.py research/ tests/*.py
