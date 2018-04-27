#!/usr/bin/env python3

import sys
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

from research.knowledge_base import URI, SparqlEndpoint # pylint: disable=wrong-import-position


def test_sparql_endpoint():
    dbo_country = URI(URI.NAMESPACES['dbo'] + 'country')
    dbr_us = URI(URI.NAMESPACES['dbr'] + 'United_States')
    dct_subject = URI('http://purl.org/dc/terms/subject')
    dbc_us = URI('http://dbpedia.org/resource/Category:States_of_the_United_States')
    query = '''
        SELECT DISTINCT ?state WHERE {{
            ?state {dbo_country} {dbr_us} .
            ?state {dct_subject} {dbc_us} .
        }} LIMIT 100
    '''.format(
        dbo_country=('<' + str(dbo_country) + '>'),
        dbr_us=dbr_us.short_str,
        dct_subject=('<' + str(dct_subject) + '>'),
        dbc_us=dbc_us.short_str,
    ).strip()
    assert query == '''
        SELECT DISTINCT ?state WHERE {
            ?state <http://dbpedia.org/ontology/country> dbr:United_States .
            ?state <http://purl.org/dc/terms/subject> dbc:States_of_the_United_States .
        } LIMIT 100
    '''.strip()

    dbpedia = SparqlEndpoint('https://dbpedia.org/sparql')
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
