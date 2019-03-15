"""Tests for data_structures.py."""

import sys
from os.path import dirname, realpath

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

def test_treemultimap():
    """Test TreeMultiMap."""
    tmm = TreeMultiMap()
    assert 0 not in tmm
    assert tmm.get_first(0) is None
    assert tmm.get_last(0) is None
    assert list(tmm) == list(tmm.keys()) == list(tmm.values()) == list(tmm.items()) == []
    orders = [
        [3, 2, 1], # root left left
        [3, 1, 2], # root left right
        [1, 3, 2], # root right left
        [1, 2, 3], # root right right
    ]
    for order in orders:
        tmm = TreeMultiMap()
        assert not tmm
        for size, number in enumerate(order, start=1):
            assert number not in tmm
            tmm.add(number, number)
            assert number in tmm
            assert len(tmm) == size
        assert list(tmm) == list(tmm.keys()) == list(tmm.values()) == [1, 2, 3]
        assert list(tmm.items()) == [(1, 1), (2, 2), (3, 3)]
    tmm = TreeMultiMap()
    for key in range(1, 11):
        for value in range(key):
            tmm.add(key, value)
    for key in range(1, 11):
        assert tmm.get_first(key) == 0
        assert tmm.get_last(key) == key - 1
        assert list(tmm[key]) == list(range(key))
    tmm = TreeMultiMap.from_dict({i: i for i in range(100)})
    assert list(tmm.items()) == [(i, i) for i in range(100)]
    with pytest.raises(ValueError):
        tmm = TreeMultiMap()
        tmm.add(42, 42)
        tmm.add(42, 42)
