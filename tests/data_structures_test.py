"""Tests for data_structures.py."""

import sys
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable=wrong-import-position
from research.data_structures import UnionFind


def test_unionfind():
    """Test UnionFind."""
    assert not UnionFind()
    union_find = UnionFind([0, 1, 2, 3])
    assert all(i in union_find for i in range(len(union_find)))
    for i in range(4, 8):
        union_find.add(i)
    assert all(union_find[i] == i for i in range(len(union_find)))
    for i in range(1, 8, 2):
        union_find.union(1, i)
    assert all(union_find.same(5, i) for i in range(1, 8, 2))
    assert set(union_find[i] for i in range(0, 8, 2)) == set(range(0, 8, 2))
