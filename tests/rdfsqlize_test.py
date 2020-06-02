"""Tests for knowledge_base.py."""

from os.path import dirname, realpath, join as join_path
from shutil import copy
from tempfile import TemporaryDirectory

from research import KnowledgeFile
from research import sqlize


def test_rdfsqlize():
    """Test rdfsqlize and knowledge bases."""
    with TemporaryDirectory() as temp_dir:
        print(temp_dir)

        nt_file = copy(join_path(dirname(realpath(__file__)), 'states.nt'), temp_dir)

        output_file = sqlize(nt_file, 'states')

        kb = KnowledgeFile(output_file, kb_name='states') # pylint: disable = invalid-name

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
        assert sorted(results) == answers, 'Failed to query for all states'

        results = []
        sparql = 'SELECT ?state where {?state <http://dbpedia.org/property/AdmittanceDate> "1889-11-02"}'
        for bindings in kb.query_sparql(sparql):
            results.append(bindings['state'].split('/')[-1])

        results = [state.replace('_', ' ') for state in results]
        assert sorted(results) == ['North Dakota', 'South Dakota'], 'Failed to query for the Dakotas'
