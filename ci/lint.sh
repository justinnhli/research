#!/bin/bash

if [ -e "$HOME/.venv/research/bin/py.test" ]; then
    pylint="$HOME/.venv/research/bin/pylint" 
elif command -v py.test >/dev/null 2>&1; then
    pylint="pylint"
else
    echo 'Cannot find pylint; quitting...'
    exit 1
fi

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
"$pylint" *.py research/ tests/*.py
