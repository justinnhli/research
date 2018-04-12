#!/usr/bin/env python3
"""Semantic relation/word embedding code.

The goal here is to see the degree to which relations in a semantic
network also have vectors that are tightly clustered.
"""

import sys
from os.path import dirname, realpath

# make sure research library code is available
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from research.knowledge_base import SparqlEndpoint
from research.word_embedding import load_model


def get_ave_sigma(model, word_pairs):
    """Compute the average sigma between word pairs.

    Arguments:
        model (GenSimModel): The word vector model.
        word_pairs (list[str]): A list of pairs (parent, child) from the knowledge base

    Returns:
        KeyedVector: The average difference between the pairs of words.
    """
    sigma = 0
    for word1, word2 in word_pairs:
        sigma += model.word_vec(word1) - model.word_vec(word2)
    return (1 / len(word_pairs)) * sigma


def get_vectors(model, words):
    """Get the vectors associated with a list of words.

    Arguments:
        model (GenSim): The model to use for word embeddings.
        words (list[str]): The list of words.

    Returns:
        list[GenSimVector]: A list of high-dimensional vectors.
    """
    return [model.word_vec(word) for word in words]


def get_relation_pairs(relation, limit=100):
    """Get names of concepts in a relation.

    Arguments:
        relation (str): A relation.
        limit (int): The number of results to return. Defaults to 20.

    Returns:
        list[str]: A list of pairs (parent, child) from the knowledge base
    """
    dbpedia = SparqlEndpoint('http://dbpedia.org/sparql')
    sparql = '''
    SELECT DISTINCT ?parent_name, ?child_name, ?parent, ?child WHERE {{
        ?parent {} ?child .
        ?parent dbp:name ?parent_name .
        ?child dbp:name ?child_name .
    }} LIMIT {}
    '''.format(relation, limit)
    results = []
    for bindings in dbpedia.query_sparql(sparql):
        results.append([bindings['parent_name'], bindings['child_name']])
    return results


def main():
    """Get vectors associated with parent-child pairs in a KB"""
    relation = 'dbp:capital'
    pairs = get_relation_pairs(relation)
    vector_model = load_model('data/models/GoogleNews-vectors-negative300.bin')
    for parent, child in pairs:
        print('pair: {}'.format([parent, child]))
        try:
            parent_vector = get_vectors(vector_model, [parent])[0]
            child_vector = get_vectors(vector_model, [child])[0]
            print('parent vector:', parent_vector)
            print('child vector:', parent_vector)
        except KeyError:
            print('cannot find vector for at least one word; skipping')
        finally:
            print()


if __name__ == '__main__':
    main()
