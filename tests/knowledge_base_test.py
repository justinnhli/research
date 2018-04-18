#!/usr/bin/env python3

import sys
from os import remove
from os.path import dirname, realpath, join as join_path

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))


from research.knowledge_base import SparqlEndpoint # pylint: disable=wrong-import-position

def test_sparql_endpoint():
    dbpedia = SparqlEndpoint('https://dbpedia.org/sparql')
    query = '''
        SELECT DISTINCT ?state WHERE {
            ?state dbo:country dbr:United_States .
            ?state dct:subject dbc:States_of_the_United_States .
        } LIMIT 100
    '''
    result = [binding['state'] for binding in dbpedia.query_sparql(query)]
    assert sorted(result) == [
        'http://dbpedia.org/resource/Alabama',
        'http://dbpedia.org/resource/Alaska',
        'http://dbpedia.org/resource/Arizona',
        'http://dbpedia.org/resource/Arkansas',
        'http://dbpedia.org/resource/California',
        'http://dbpedia.org/resource/Colorado',
        'http://dbpedia.org/resource/Connecticut',
        'http://dbpedia.org/resource/Delaware',
        'http://dbpedia.org/resource/Florida',
        'http://dbpedia.org/resource/Georgia_(U.S._state)',
        'http://dbpedia.org/resource/Hawaii',
        'http://dbpedia.org/resource/Idaho',
        'http://dbpedia.org/resource/Illinois',
        'http://dbpedia.org/resource/Indiana',
        'http://dbpedia.org/resource/Iowa',
        'http://dbpedia.org/resource/Kansas',
        'http://dbpedia.org/resource/Kentucky',
        'http://dbpedia.org/resource/Louisiana',
        'http://dbpedia.org/resource/Maine',
        'http://dbpedia.org/resource/Maryland',
        'http://dbpedia.org/resource/Massachusetts',
        'http://dbpedia.org/resource/Michigan',
        'http://dbpedia.org/resource/Minnesota',
        'http://dbpedia.org/resource/Mississippi',
        'http://dbpedia.org/resource/Missouri',
        'http://dbpedia.org/resource/Montana',
        'http://dbpedia.org/resource/Nebraska',
        'http://dbpedia.org/resource/Nevada',
        'http://dbpedia.org/resource/New_Hampshire',
        'http://dbpedia.org/resource/New_Jersey',
        'http://dbpedia.org/resource/New_Mexico',
        'http://dbpedia.org/resource/New_York',
        'http://dbpedia.org/resource/North_Carolina',
        'http://dbpedia.org/resource/North_Dakota',
        'http://dbpedia.org/resource/Ohio',
        'http://dbpedia.org/resource/Oklahoma',
        'http://dbpedia.org/resource/Oregon',
        'http://dbpedia.org/resource/Pennsylvania',
        'http://dbpedia.org/resource/Rhode_Island',
        'http://dbpedia.org/resource/South_Carolina',
        'http://dbpedia.org/resource/South_Dakota',
        'http://dbpedia.org/resource/Tennessee',
        'http://dbpedia.org/resource/Texas',
        'http://dbpedia.org/resource/Utah',
        'http://dbpedia.org/resource/Vermont',
        'http://dbpedia.org/resource/Virginia',
        'http://dbpedia.org/resource/Washington_(state)',
        'http://dbpedia.org/resource/West_Virginia',
        'http://dbpedia.org/resource/Wisconsin',
        'http://dbpedia.org/resource/Wyoming',
    ]


test_sparql_endpoint()
