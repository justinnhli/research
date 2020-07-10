#!/usr/bin/env python3
"""Tests for RL memory code."""

from research import SparqlEndpoint
from research import State, Action, Environment
from research import MemoryArchitectureMetaEnvironment
from research import NaiveDictLTM, NetworkXLTM, SparqlLTM
from research import AttrVal


def test_memory_architecture():
    """Test the memory architecture meta-environment."""

    class TestEnv(Environment):
        """A simple environment with a single string state."""

        def __init__(self, size, index=0):
            """Initialize the TestEnv.

            Arguments:
                size (int): The length of one side of the square.
                index (int): The initial int.
            """
            super().__init__()
            self.size = size
            self.init_index = index
            self.index = self.init_index

        def get_state(self): # noqa: D102
            return State(index=self.index)

        def get_observation(self): # noqa: D102
            return State(index=self.index)

        def get_actions(self): # noqa: D102
            if self.index == -1:
                return []
            else:
                return [Action(str(i)) for i in range(-1, size * size)]

        def reset(self): # noqa: D102
            self.start_new_episode()

        def start_new_episode(self): # noqa: D102
            self.index = self.init_index

        def react(self, action): # noqa: D102
            assert action in self.get_actions()
            if action.name != 'no-op':
                self.index = int(action.name)
            if self.end_of_episode():
                return 100
            else:
                return -1

        def visualize(self): # noqa: D102
            pass

    size = 5
    env = MemoryArchitectureMetaEnvironment(
        env=TestEnv(
            size=size,
            index=0,
        ),
        ltm=NaiveDictLTM(),
    )
    env.start_new_episode()
    for i in range(size * size):
        env.add_to_ltm(index=i, row=(i // size), col=(i % size))
    # test observation
    assert env.get_observation() == State(
        perceptual_index=0,
    ), env.get_observation()
    # test actions
    assert (
        set(env.get_actions()) == set([
            *(Action(str(i)) for i in range(-1, size * size)),
            Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=0),
        ])
    ), set(env.get_actions())
    # test pass-through reaction
    reward = env.react(Action('9'))
    assert env.get_observation() == State(
        perceptual_index=9,
    ), env.get_observation()
    assert reward == -1, reward
    # query test
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=9))
    assert env.get_observation() == State(
        perceptual_index=9,
        query_index=9,
        retrieval_index=9,
        retrieval_row=1,
        retrieval_col=4,
    ), env.get_observation()
    # query with no results
    env.react(Action('copy', src_buf='retrieval', src_attr='row', dst_buf='query', dst_attr='row', dst_val=1))
    env.react(Action('0'))
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=0))
    env.react(Action('delete', buf='query', attr='index', val=9))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_index=0,
        query_row=1,
    ), env.get_observation()
    # delete and query test
    env.react(Action('delete', buf='query', attr='index', val=0))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # next result test
    env.react(Action('next-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=6,
        retrieval_row=1,
        retrieval_col=1,
    ), env.get_observation()
    # delete test
    env.react(Action('prev-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # complete the environment
    reward = env.react(Action('-1'))
    assert env.end_of_episode()
    assert reward == 100, reward


def test_networkxltm():
    """Test the NetworkX LTM."""

    def activation_fn(graph, mem_id):
        graph.nodes[mem_id]['activation'] += 1

    ltm = NetworkXLTM(activation_fn=activation_fn)
    ltm.store('cat', is_a='mammal', has='fur', name='cat')
    ltm.store('bear', is_a='mammal', has='fur', name='bear')
    ltm.store('whale', is_a='mammal', lives_in='water')
    ltm.store('whale', name='whale') # this activates whale
    ltm.store('fish', is_a='animal', lives_in='water')
    ltm.store('mammal', has='vertebra', is_a='animal')
    # retrieval
    result = ltm.retrieve('whale')
    assert sorted(result) == [('is_a', 'mammal'), ('lives_in', 'water'), ('name', 'whale')]
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


def test_sparqlltm():
    """Test the SPARQL endpoint LTM."""
    release_date_attr = '<http://dbpedia.org/ontology/releaseDate>'
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