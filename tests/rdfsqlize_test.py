#!/usr/bin/env python3

import sys
from os import remove
from os.path import dirname, realpath, join as join_path

DIRECTORY = dirname(realpath(__file__))

sys.path.append(join_path(DIRECTORY, '..', '..'))

from research.kb import KnowledgeFile # pylint: disable=wrong-import-position
from research.rdfsqlize import sqlize # pylint: disable=wrong-import-position

def test_rdfsqlize():

    nt_file = join_path(DIRECTORY, 'states.nt')

    output_file = sqlize(nt_file, 'states')

    results = []
    kb = KnowledgeFile(output_file, kb_name='states') # pylint: disable=invalid-name
    for state in kb.query_sparql('SELECT ?state where {?state a dbo:State}'):
        results.append(str(state[0]).split('/')[-1])
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

    assert sorted(results) == answers

    remove(output_file)
