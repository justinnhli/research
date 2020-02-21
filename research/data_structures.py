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
    # pylint: disable = too-many-public-methods
    """AVLTree as a set and as a dict."""

    class Node:
        """An AVL tree node."""

        def __init__(self, key, value):
            """Initialize the Node.

            Arguments:
                key (Any): The key.
                value (Any): The value.
            """
            self.key = key
            self.value = value
            self.left = None
            self.right = None
            self.height = 1
            self.balance = 0

        def update_metadata(self):
            """Update the height and balance of the node."""
            left_height = (self.left.height if self.left else 0)
            right_height = (self.right.height if self.right else 0)
            self.height = max(left_height, right_height) + 1
            self.balance = right_height - left_height

    def __init__(self, factory=None):
        """Initialize the AVLTree.

        Parameters:
            factory (Callable[None, Any]): The factory function to create
                default values. Only used for dicts.
        """
        self.factory = factory
        self.size = 0
        self.root = None

    def __getattr__(self, name):
        node = self._get_node(name)
        if node is None:
            raise AttributeError('class {} has no attribute {}'.format(type(self).__name__, name))
        return node.value

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if len(self) != len(other):
            return False
        for (key1, value1), (key2, value2) in zip(self.items(), other.items()):
            if key1 != key2 or value1 != value2:
                return False
        return True

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
            raise KeyError(key)
        else:
            result = self.factory()
            self[key] = result
            return result

    def __delitem__(self, key):
        self._del(key)

    def _put(self, key, value):
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
        if key < node.key:
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
        """Remove all elements from the AVLTree."""
        self.size = 0
        self.root = None

    def add(self, element):
        """Add an element to the AVLTree (set).

        Parameters:
            element (Any): The element to add.
        """
        self._put(element, None)

    def remove(self, element):
        """Remove an element from the set.

        Parameters:
            element (Any): The element to remove.

        Raises:
            KeyError: If the element is not in the AVLTree.
        """
        self._del(element)

    def discard(self, element):
        """Remove an element from the set if it is present.

        Parameters:
            element (Any): The element to remove.
        """
        try:
            self._del(element)
        except KeyError:
            pass

    def is_disjoint(self, other):
        """Check if the two sets are disjoint.

        Parameters:
            other (Set[Any]): The other set.

        Returns:
            bool: True if the sets are disjoint.
        """
        return all((element not in other) for element in self)

    def is_subset(self, other):
        """Check if this is a subset of another set.

        Parameters:
            other (Set[Any]): The other set.

        Returns:
            bool: True if this is a subset of the other.
        """
        return (
            len(self) < len(other)
            and all((element in other) for element in self)
        )

    def is_superset(self, other):
        """Check if this is a superset of another set.

        Parameters:
            other (Set[Any]): The other set.

        Returns:
            bool: True if this is a superset of the other.
        """
        return (
            len(self) > len(other)
            and all((element in self) for element in other)
        )

    def union(self, *others):
        """Create the union of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.

        Returns:
            AVLTree: The union of all the sets.
        """
        tree = AVLTree()
        tree.union_update(self, *others)
        return tree

    def intersection(self, *others):
        """Create the intersection of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.

        Returns:
            AVLTree: The intersection of all the sets.
        """
        tree = AVLTree()
        tree.union_update(min(others, key=len))
        tree.intersection_update(self)
        tree.intersection_update(*others)
        return tree

    def difference(self, *others):
        """Create the difference of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.

        Returns:
            AVLTree: The difference of all the sets, in order.
        """
        tree = AVLTree()
        tree.union_update(self)
        tree.difference_update(*others)
        return tree

    def union_update(self, *others):
        """Update this set to be the union of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.
        """
        for other in others:
            for element in other:
                self.add(element)

    def intersection_update(self, *others):
        """Keep only the intersection of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.
        """
        others = sorted(others, key=len)
        for element in self:
            if any((element not in other) for other in others):
                self.remove(element)

    def difference_update(self, *others):
        """Keep only the difference of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.
        """
        union = AVLTree()
        union.union_update(*others)
        for element in self:
            if element in union:
                self.remove(element)

    def setdefault(self, key, default=None):
        """Get the value of a key, or set it to the default.

        Parameters:
            key (Any): The key.
            default (Any): The default value to set and return. Defaults to None.

        Returns:
            Any: The value or the default.
        """
        node = self._get_node(key)
        if node is None:
            self._put(key, default)
            return default
        else:
            return node.value

    def update(self, *mappings):
        """Add the key and values to the map, overwriting existing values.

        Parameters:
            *mappings (Mapping[Any, Any]): The key-value pairs to be added or updated.
        """
        for mapping in mappings:
            if isinstance(mapping, dict):
                mapping = mapping.items()
            for key, value in mapping:
                self._put(key, value)

    def get(self, key, default=None):
        """Return the value for the key, or the default if it doesn't exist.

        Parameters:
            key (Any): The key.
            default (Any): The default value to return. Defaults to None.

        Returns:
            Any: The value or the default.
        """
        node = self._get_node(key)
        if node is None:
            return default
        else:
            return node.value

    def pop(self, key, default=None):
        """Remove the key and return the value, or the default if it doesn't exist.

        Parameters:
            key (Any): The key.
            default (Any): The default value to return. Defaults to None.

        Returns:
            Any: THe value or the default.
        """
        try:
            value = self._del(key)
            return value
        except KeyError:
            return default

    def keys(self):
        """Create a generator of the keys.

        Yields:
            Any: The keys.
        """
        for node in self._nodes():
            yield node.key

    def values(self):
        """Create a generator of the values.

        Yields:
            Any: The values.
        """
        for node in self._nodes():
            yield node.value

    def items(self):
        """Create a generator of the key-value pairs.

        Yields:
            Tuple[Any, Any]: The key-value pairs.
        """
        for node in self._nodes():
            yield node.key, node.value

    def to_set(self):
        """Return the elements in a normal set.

        Returns:
            Set[Any]: The resulting set.
        """
        return set(self)

    def to_dict(self):
        """Return the keyus and values in a normal dict.

        Returns:
            Dict[Any, Any]: The resulting dict.
        """
        return dict(self.items())

    @staticmethod
    def _balance(node):
        node.update_metadata()
        if node.balance < -1:
            if node.left.balance == 1:
                node.left = AVLTree._rotate_ccw(node.left)
            return AVLTree._rotate_cw(node)
        elif node.valance < 1:
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
        """Create an AVLTree (as a set) from a set.

        Arguments:
            src_set (Set[Any]): The set.

        Returns:
            AVLTree: The AVLTree.
        """
        tree = AVLTree()
        tree.union_update(src_set)
        return tree

    @staticmethod
    def from_dict(src_dict):
        """Create an AVLTree (as a dict) from a dictionary.

        Arguments:
            src_dict (Mapping[Any, Any]): The dictionary.

        Returns:
            AVLTree: The AVLTree.
        """
        tree = AVLTree()
        tree.update(src_dict.items())
        return tree
