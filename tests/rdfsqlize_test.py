#!/usr/bin/env python3
"""Tests for knowledge_base.py."""

import sys
from os import remove
from os.path import dirname, realpath, join as join_path

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable=wrong-import-position
from research.knowledge_base import KnowledgeFile
from research.rdfsqlize import sqlize


def test_rdfsqlize():
    """Test rdfsqlize and knowledge bases."""
    nt_file = join_path(DIRECTORY, 'states.nt')

    output_file = sqlize(nt_file, 'states')

    kb = KnowledgeFile(output_file, kb_name='states') # pylint: disable=invalid-name

    results = []
    sparql = 'SELECT ?state WHERE {?state a dbo:State}'
    for bindings in kb.query_sparql(sparql):
        results.append(bindings['state'].split('/')[-1])

    results = [state.replace('_', ' ') for state in results]
    answers = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
        'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
        'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
        'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
        'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
    ]
    if sorted(results) != answers:
        remove(output_file)
        assert False, 'Failed to query for all states'

    results = []
    sparql = 'SELECT ?state where {?state <http://dbpedia.org/property/AdmittanceDate> "1889-11-02"}'
    for bindings in kb.query_sparql(sparql):
        results.append(bindings['state'].split('/')[-1])

    results = [state.replace('_', ' ') for state in results]
    if not sorted(results) == ['North Dakota', 'South Dakota']:
        remove(output_file)
        assert False, 'Failed to query for the Dakotas'

    remove(output_file)
