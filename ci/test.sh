#!/bin/sh

if [ -e "$HOME/.venv/research/bin/py.test" ]; then
    pytest="$HOME/.venv/research/bin/py.test" 
elif command -v py.test >/dev/null 2>&1; then
    pytest="py.test"
else
    echo 'Cannot find py.test; quitting...'
    exit 1
fi

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
"$pytest" --verbose --cov=research/ tests
exit $?
