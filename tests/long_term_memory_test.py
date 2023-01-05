#!/usr/bin/env python3
"""Tests for RL memory code."""

from research import SparqlEndpoint
from research import NaiveDictLTM, NetworkXLTM, SparqlLTM
from research import AttrVal


def _test_ltm(ltm):
    """Test an LTM."""
    ltm.store('cat', is_a='mammal', has='fur', name='cat')
    ltm.store('bear', is_a='mammal', has='fur', name='bear')
    ltm.store('whale', is_a='mammal', lives_in='water')
    ltm.store('whale', name='whale') # this activates whale
    ltm.store('fish', is_a='animal', lives_in='water', has='scale')
    ltm.store('fish', has='gill')
    ltm.store('mammal', has='vertebra', is_a='animal')
    # retrieval
    result = ltm.retrieve('whale')
    assert sorted(result) == [('is_a', 'mammal'), ('lives_in', 'water'), ('name', 'whale')]
    # multi-valued retrieval
    result = ltm.retrieve('fish')
    assert sorted(result) == [('has', 'gill'), ('has', 'scale'), ('is_a', 'animal'), ('lives_in', 'water')]
    # failed query
    result = ltm.query(set([('has', 'vertebra'), ('lives_in', 'water')]))
    assert result is None
    # unique query
    result = ltm.query(set([('has', 'vertebra')]))
    assert sorted(result) == [AttrVal('has', 'vertebra'), AttrVal('is_a', 'animal')]
    # query traversal
    ltm.store('cat')
    # at this point, whale has been activated twice (from the store and the retrieve)
    # while cat has been activated once (from the store)
    # so a search for mammals will give, in order: whale, cat, bear
    result = ltm.query(set([('is_a', 'mammal')]))
    assert AttrVal('name', 'whale') in result
    assert ltm.has_next_result
    result = ltm.next_result()
    assert AttrVal('name', 'cat') in result
    assert ltm.has_next_result
    result = ltm.next_result()
    assert AttrVal('name', 'bear') in result
    assert not ltm.has_next_result
    assert ltm.has_prev_result
    result = ltm.prev_result()
    assert ltm.has_prev_result
    result = ltm.prev_result()
    assert AttrVal('name', 'whale') in result
    assert not ltm.has_prev_result


def test_naivedictltm():
    """Test the dict LTM."""

    def activation_fn(ltm, mem_id, time):
        # pylint: disable = unused-argument
        ltm.activations[mem_id] += 1

    _test_ltm(NaiveDictLTM(activation_fn=activation_fn))


def test_networkxltm():
    """Test the NetworkX LTM."""

    def activation_fn(ltm, mem_id, time):
        # pylint: disable = unused-argument
        ltm.graph.nodes[mem_id]['activation'] += 1

    _test_ltm(NetworkXLTM(activation_fn=activation_fn))


def test_sparqlltm():
    """Test the SPARQL endpoint LTM."""
    release_date_attr = '<http://dbpedia.org/property/released>'
    release_date_value = '"1979-11-30"^^<http://www.w3.org/2001/XMLSchema#date>'
    # connect to DBpedia
    dbpedia = SparqlEndpoint('https://dbpedia.org/sparql')
    # test retrieve
    ltm = SparqlLTM(dbpedia)
    result = ltm.retrieve('<http://dbpedia.org/resource/The_Wall>')
    assert AttrVal(release_date_attr, release_date_value) in result, result
    # test query
    result = ltm.query(set([
        ('<http://dbpedia.org/ontology/releaseDate>', '"1979-11-30"^^xsd:date'),
        ('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', '<http://dbpedia.org/ontology/Album>'),
    ]))
    assert AttrVal(release_date_attr, release_date_value) in result, result
