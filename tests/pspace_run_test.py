"""Tests for pspace_run.py."""

import sys
from io import StringIO
from os.path import dirname, realpath
from pathlib import Path

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from permspace import PermutationSpace

import research.pspace_run as pspace_run

PSPACE = PermutationSpace(
    ['arabic', 'letter_lower', 'roman_lower'],
    arabic=range(1, 4),
    letter_lower=list('abc'),
    roman_lower=['i', 'ii', 'iii'],
)

def run_experiment(params): # pylint: disable = unused-argument
    """Empty experiment function.

    Arguments:
        params (permspace.Namespace): The parameters.
    """


def same_namespace(namespace, string):
    """Check if the string is of the namespace.

    Arguments:
        namespace (permspace.Namespace): The namespace.
        string (str): The string to check.

    Returns:
        bool: True if the string is of the namespace.
    """
    return string.endswith(str(namespace))


def test_pspace(capsys, monkeypatch):
    """Test pspace_run.

    Arguments:
        capsys (pytest.CaptureFixture): Pytest object to capture standard output.
        monkeypatch (pytest.MonkeyPatch): Pytest object to fake standard input.
    """
    curr_file = Path(__file__).resolve()
    # dry run
    pspace_run.main([
        '--dry-run',
        str(curr_file),
        f'{curr_file.stem}.PSPACE',
        f'{curr_file.stem}.run_experiment',
    ])
    captured_output = capsys.readouterr().out
    for params, line in zip(PSPACE, captured_output.splitlines()):
        line = line.strip()
        assert same_namespace(params, line)
    # serial
    pspace_run.main(
        [],
        curr_file,
        f'{curr_file.stem}.PSPACE',
        f'{curr_file.stem}.run_experiment',
    )
    captured_output = capsys.readouterr().out
    for params, line in zip(PSPACE, captured_output.splitlines()):
        line = line.strip()
        assert same_namespace(params, line)
    # parallel
    monkeypatch.setattr('sys.stdin', StringIO('n'))
    pspace_run.main(
        ['--dispatch', 'true', '--num-cores', '2'],
        curr_file,
        f'{curr_file.stem}.PSPACE',
        f'{curr_file.stem}.run_experiment',
    )
    captured_output = capsys.readouterr().out
    for param, values in PSPACE.parameters.items():
        assert f'{param} ({len(values)})' in captured_output
    assert '0 filters; total size: 27' in captured_output
    assert 'total invocations: 2' in captured_output
