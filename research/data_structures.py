"""Utility data structures."""

from typing import Any, Optional, Iterable, Iterator, Generator, Mapping, Hashable, AbstractSet, Tuple, ValuesView


class UnionFind:
    """UnionFind for discrete sets."""

    def __init__(self, nodes=None):
        # type: (Optional[Iterable[Hashable]]) -> None
        """Initialize the UnionFind.

        Arguments:
            nodes (Hashable): The object to be hashed.
        """
        if nodes is None:
            nodes = []
        self.parents = {node: node for node in nodes}

    def __len__(self):
        # type: () -> int
        return len(self.parents)

    def __contains__(self, node):
        # type: (Hashable) -> bool
        return node in self.parents

    def __getitem__(self, node):
        # type: (Hashable) -> Hashable
        path = []
        while self.parents[node] != node:
            path.append(node)
            node = self.parents[node]
        path.append(node)
        rep = path[-1]
        for step in path:
            self.parents[step] = rep
        return rep

    def __iter__(self):
        # type: () -> Iterator[Hashable]
        return iter(self.parents)

    def __bool__(self):
        # type: () -> bool
        return bool(self.parents)

    def union(self, node1, node2):
        # type: (Hashable, Hashable) -> None
        """Join two discrete sets.

        Arguments:
            node1 (Hashable): A member of one set to be joined.
            node2 (Hashable): A member of the other set to be joined.
        """
        self.add(node1)
        rep1 = self[node1]
        self.add(node2)
        rep2 = self[node2]
        self.parents[rep2] = rep1

    def same(self, node1, node2):
        # type: (Hashable, Hashable) -> bool
        """Check if two members are in the same set.

        Arguments:
            node1 (Hashable): A member of one set.
            node2 (Hashable): A member of the other set.

        Returns:
            bool: True if the members are in the same set.
        """
        return self[node1] == self[node2]

    def add(self, node, parent=None):
        # type: (Hashable, Optional[Hashable]) -> bool
        """Add a node.

        Arguments:
            node (Hashable): The member to add.
            parent (Hashable): A node from the same set, if any.

        Returns:
            bool: True if a new member has been added.
        """
        if node in self.parents:
            return False
        if parent is None:
            parent = node
        self.parents[node] = parent
        return True


class AVLTree:

    class Node:

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.left = None
            self.right = None
            self.height = 1
            self.balance = 0

        def update_metadata(self):
            left_height = (self.left.height if self.left else 0)
            right_height = (self.right.height if self.right else 0)
            self.height = max(left_height, right_height) + 1
            self.balance = right_height - left_height

    def __init__(self, factory=None):
        self.factory = factory
        self.size = 0
        self.root = None

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return self._get_node(key) is not None

    def __iter__(self):
        for node in self._nodes():
            yield node.key

    def __setitem__(self, key, value):
        self._put(key, value)

    def __getitem__(self, key):
        node = self._get_node(key)
        if node is not None:
            return node.value
        elif self.factory is None:
            return None
        else:
            result = self.factory()
            self[key] = result
            return result

    def __delitem__(self, key):
        self._del(key)

    def _put(self, key, value, node=None):
        self.root = self._put_helper(self.root, key, value)

    def _put_helper(self, node, key, value):
        if node is None:
            self.size += 1
            return self.Node(key, value)
        elif key == node.key:
            node.value = value
            return node
        elif key < node.key:
            node.left = self._put_helper(node.left, key, value)
        else:
            node.right = self._put_helper(node.right, key, value)
        return self._balance(node)

    def _get_node(self, key):

        def _get_node_helper(node, key):
            if node is None:
                return None
            elif key < node.key:
                return _get_node_helper(node.left, key)
            elif node.key < key:
                return _get_node_helper(node.right, key)
            else:
                return node

        return _get_node_helper(self.root, key)

    def _del(self, key):
        self.root, value = self._del_helper(self.root, key)
        return value

    def _del_helper(self, node, key):
        value = None
        if node is None:
            raise KeyError(key)
        elif key < node.key:
            node.left, value = self._del_helper(node.left, key)
        elif node.key < key:
            node.right, value = self._del_helper(node.right, key)
        else:
            if node.left is None and node.right is None:
                self.size -= 1
                return None, node.value
            replacement = node
            if node.left is not None:
                replacement = node.left
                while replacement.right is not None:
                    replacement = replacement.right
                node.left, value = self._del_helper(node.left, replacement.key)
            elif node.right is not None:
                replacement = node.right
                while replacement.left is not None:
                    replacement = replacement.left
                node.right, value = self._del_helper(node.right, replacement.key)
            node.key = replacement.key
            node.value = replacement.value
        return self._balance(node), value

    def _nodes(self):

        def _nodes_helper(node):
            if node is None:
                return
            yield from _nodes_helper(node.left)
            yield node
            yield from _nodes_helper(node.right)

        yield from _nodes_helper(self.root)

    def clear(self):
        self.size = 0
        self.root = None

    def add(self, element):
        self._put(element, None)

    def remove(self, element):
        self._del(element)

    def discard(self, element):
        try:
            self._del(element)
        except KeyError:
            pass

    def get(self, key, default=None):
        node = self._get_node(key)
        if node is None:
            return default
        else:
            return node.value

    def pop(self, key, default=None):
        try:
            value = self._del(key)
            return value
        except KeyError:
            return default

    def keys(self):
        for node in self._nodes():
            yield node.key

    def values(self):
        for node in self._nodes():
            yield node.value

    def items(self):
        for node in self._nodes():
            yield node.key, node.value

    def to_set(self):
        return set(self)

    def to_dict(self):
        return dict(self.items())

    @staticmethod
    def _balance(node):
        node.update_metadata()
        if node.balance < -1:
            if node.left.balance == 1:
                node.left = AVLTree._rotate_ccw(node.left)
            return AVLTree._rotate_cw(node)
        elif 1 < node.balance:
            if node.right.balance == -1:
                node.right = AVLTree._rotate_cw(node.right)
            return AVLTree._rotate_ccw(node)
        else:
            return node

    @staticmethod
    def _rotate_cw(node):
        left = node.left
        node.left = left.right
        left.right = node
        node.update_metadata()
        left.update_metadata()
        return left

    @staticmethod
    def _rotate_ccw(node):
        right = node.right
        node.right = right.left
        right.left = node
        node.update_metadata()
        right.update_metadata()
        return right

    @staticmethod
    def from_set(src_set):
        tree = AVLTree()
        for element in src_set:
            tree.add(element)
        return tree

    @staticmethod
    def from_dict(src_dict):
        tree = AVLTree()
        for key, value in src_dict.items():
            tree[key] = value
        return tree
