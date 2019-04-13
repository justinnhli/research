"""Tests for data_structures.py."""

import sys
from os.path import dirname, realpath
from itertools import permutations

import pytest

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.data_structures import UnionFind, TreeMultiMap


def test_unionfind():
    """Test UnionFind."""
    assert not UnionFind()
    union_find = UnionFind(range(4))
    assert all(i in union_find for i in range(len(union_find)))
    assert set(union_find) == set(range(4))
    for i in range(4, 8):
        union_find.add(i)
    assert all(union_find[i] == i for i in range(len(union_find)))
    for i in range(1, 8, 2):
        union_find.union(1, i)
    assert all(union_find.same(5, i) for i in range(1, 8, 2))
    assert set(union_find[i] for i in range(0, 8, 2)) == set(range(0, 8, 2))

def _test_tmm_order(order):
    perm_size = len(order)
    tmm = TreeMultiMap()
    assert not tmm
    for size, number in enumerate(order, start=1):
        assert number not in tmm
        tmm.add(number, number)
        assert number in tmm
        assert len(tmm) == size
    assert list(tmm) == list(tmm.keys()) == list(tmm.values()) == list(range(perm_size))
    assert list(tmm.items()) == [(num, num) for num in range(perm_size)]
    for num_removed, number in enumerate(order, start=1):
        tmm.remove(number, number)
        assert tmm.size == perm_size - num_removed
        assert list(tmm.items()) == sorted((num, num) for num in order[num_removed:])

def test_treemultimap():
    """Test TreeMultiMap."""
    tmm = TreeMultiMap()
    assert 0 not in tmm
    assert tmm.get_first(0) is None
    assert tmm.get_last(0) is None
    assert list(tmm) == list(tmm.keys()) == list(tmm.values()) == list(tmm.items()) == []
    perm_size = 5
    for order in permutations(range(perm_size)):
        _test_tmm_order(order)
    test_cases = [
        (0, 1, 2, 7, 8, 9, 3, 4, 5, 6),
    ]
    for order in test_cases:
        _test_tmm_order(order)
    tmm = TreeMultiMap(multi_level=TreeMultiMap.UNIQUE_VALUE)
    for key in range(1, 11):
        for value in range(key):
            tmm.add(key, value)
    for key in range(1, 11):
        assert tmm.get_first(key) == 0
        assert tmm.get_last(key) == key - 1
        assert list(tmm.yield_all(key)) == list(range(key))
    tmm.clear()
    assert tmm.size == 0
    assert list(tmm) == []
    tmm = TreeMultiMap.from_dict({i: i for i in range(100)})
    assert list(tmm.items()) == [(i, i) for i in range(100)]
    tmm = TreeMultiMap()
    tmm.add(42, 42)
    with pytest.raises(ValueError):
        tmm.add(42, 42)
    with pytest.raises(ValueError):
        tmm.remove(42, 43)
