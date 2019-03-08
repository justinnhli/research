#!/usr/bin/env python3
"""Tests for knowledge_base.py."""

import sys
from ast import literal_eval
from os.path import dirname, realpath

import pytest

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.knowledge_base import Value, SparqlEndpoint


def test_value():
    """Test the Value class."""
    uri = 'http://dbpedia.org/resource/California'
    values = [
        Value.from_uri(uri),
        Value.from_namespace_fragment('dbr', 'California'),
    ]
    for val in values:
        assert val.is_uri
        assert not val.is_literal
        assert val.uri == uri
        assert val.namespace == 'dbr'
        assert val.prefix == 'http://dbpedia.org/resource/'
        assert val.fragment == 'California'
        assert str(val) == val.rdf_format == f'<{uri}>'
        with pytest.raises(ValueError):
            print(val.literal_value)
        with pytest.raises(ValueError):
            print(val.lang)
        with pytest.raises(ValueError):
            print(val.datatype)
    literal = '"xyz"@en^^<http://example.org/ns/userDatatype>'
    val = Value.from_literal(literal)
    assert val.literal_value == 'xyz'
    assert val.lang == 'en'
    assert val.datatype.rdf_format == '<http://example.org/ns/userDatatype>'
    assert val.rdf_format == literal
    literals = [
        ('false', 'boolean'),
        ('1', 'integer'),
        ('3.14', 'double'),
        ('"hello"', None),
    ]
    for literal, literal_type in literals:
        if literal in ('true', 'false'):
            python_literal = literal_eval(literal.title())
        else:
            python_literal = literal_eval(literal)
        print(literal, literal.title(), python_literal)
        for val in [Value.from_literal(literal), Value.from_python_literal(python_literal)]:
            assert val.literal_value == python_literal
            assert val.lang is None
            if literal_type is None:
                assert val.datatype is None
            else:
                assert val.datatype.rdf_format == f'<http://www.w3.org/2001/XMLSchema#{literal_type}>'
            assert val.rdf_format == literal


def test_sparql_endpoint():
    """Test URIs and SPARQL endpoints with dbpedia."""
    dbo_country = Value.from_uri(Value.NAMESPACES['dbo'] + 'country')
    dbr_us = Value.from_uri(Value.NAMESPACES['dbr'] + 'United_States')
    dct_subject = Value.from_uri('http://purl.org/dc/terms/subject')
    dbc_us = Value.from_uri('http://dbpedia.org/resource/Category:States_of_the_United_States')
    query = '''
        SELECT DISTINCT ?state WHERE {{
            ?state {dbo_country} {dbr_us} .
            ?state {dct_subject} {dbc_us} .
        }} LIMIT 100
    '''.format(
        dbo_country=str(dbo_country),
        dbr_us=dbr_us.namespace_fragment,
        dct_subject=str(dct_subject),
        dbc_us=dbc_us.namespace_fragment,
    ).strip()
    assert query == '''
        SELECT DISTINCT ?state WHERE {
            ?state <http://dbpedia.org/ontology/country> dbr:United_States .
            ?state <http://purl.org/dc/terms/subject> dbc:States_of_the_United_States .
        } LIMIT 100
    '''.strip()

    dbpedia = SparqlEndpoint('https://dbpedia.org/sparql')
    result = [binding['state'].uri for binding in dbpedia.query_sparql(query)]
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
