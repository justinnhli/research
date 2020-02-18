"""Tests for data_structures.py."""

import sys
from os.path import dirname, realpath
from itertools import permutations

import pytest

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.data_structures import UnionFind, AVLTree


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


def test_avltree():
    """Test AVLTree."""
    tree = AVLTree()
    assert 0 not in tree
    assert list(tree) == list(tree.keys()) == list(tree.values()) == list(tree.items()) == []
    size = 7
    # set check
    for permutation in permutations(range(size)):
        tree = AVLTree()
        for element in permutation:
            tree.add(element)
        assert len(tree) == size
        assert list(e for e in tree) == list(range(size)), f'{list(e for e in tree)} != {list(range(size))}'
        for n in range(size):
            assert n in tree
        for n in range(size):
            tree.discard(n)
            assert len(tree) == size - n - 1
            assert list(e for e in tree) == list(range(n+1,size)), f'{list(e for e in tree)} != {list(range(size))}'
    src_set = set(range(101))
    assert AVLTree.from_set(src_set).to_set() == src_set
    # map check
    for permutation in permutations(range(size)):
        tree = AVLTree()
        for key in permutation:
            tree[key] = key * key
        assert len(tree) == size
        assert list(e for e in tree) == list(range(size)), f'{list(e for e in tree)} != {list(range(size))}'
        assert list(tree.items()) == list((n, n * n) for n in range(size))
        for n in range(size):
            assert n in tree
            assert tree[n] == n * n
        for n in range(size):
            del tree[n]
            assert len(tree) == size - n - 1
    src_dict = {n: n * n for n in range(101)}
    assert AVLTree.from_dict(src_dict).to_dict() == src_dict
    # defaultdict check
    tree = AVLTree(factory=(lambda: AVLTree()))
    for i in range(10):
        for j in range(i, i + 5):
            tree[i].add(j)
    for i in range(10):
        assert tree[i].to_set() == set(range(i, i + 5))
    tree.clear()
    assert len(tree) == 0
    assert list(tree) == []
