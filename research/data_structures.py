"""Utility data structures."""

from typing import Any, Optional, Union
from typing import Iterable, Iterator, Generator, Mapping, Hashable, Callable, Collection, AbstractSet
from typing import Tuple, Set, Dict, ValuesView


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


class AVLTree(Mapping[Any, Any]):
    # pylint: disable = too-many-public-methods
    """AVLTree as a set and as a dict."""

    class Node:
        """An AVL tree node."""

        def __init__(self, key, value):
            # type: (Any, Any) -> None
            """Initialize the Node.

            Arguments:
                key (Any): The key.
                value (Any): The value.
            """
            self.key = key
            self.value = value
            self.left = None # type: Optional[AVLTree.Node]
            self.right = None # type: Optional[AVLTree.Node]
            self.height = 1
            self.balance = 0

        def update_metadata(self):
            # type: () -> None
            """Update the height and balance of the node."""
            left_height = (self.left.height if self.left else 0)
            right_height = (self.right.height if self.right else 0)
            self.height = max(left_height, right_height) + 1
            self.balance = right_height - left_height

    def __init__(self, factory=None):
        # type: (Callable[[], Any]) -> None
        """Initialize the AVLTree.

        Parameters:
            factory (Callable[None, Any]): The factory function to create
                default values. Only used for dicts.
        """
        self.factory = factory
        self.size = 0
        self.root = None # type: Optional[AVLTree.Node]
        self._hash = None

    def __bool__(self):
        # type: () -> bool
        return self.size != 0

    def __eq__(self, other):
        # type: (Any) -> bool
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        if len(self) != len(other):
            return False
        return all(
            pair1 == pair2 for pair1, pair2
            in zip(self.items(), other.items())
        )

    def __lt__(self, other):
        # type: (Any) -> bool
        for pair1, pair2 in zip(self.items(), other.items()):
            if pair1 < pair2:
                return True
            if pair1 > pair2:
                return False
        return len(self) < len(other)

    def __len__(self):
        # type: () -> int
        return self.size

    def __contains__(self, key):
        # type: (Any) -> bool
        return self._get_node(key) is not None

    def __iter__(self):
        # type: () -> Generator[Any, None, None]
        for node in self._nodes():
            yield node.key

    def __setitem__(self, key, value):
        # type: (Any, Any) -> None
        self._put(key, value)

    def __getitem__(self, key):
        # type: (Any) -> Any
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
        # type: (Any) -> None
        self._del(key)

    @property
    def contents_hash(self):
        """Get a hash of the contents.

        Note: this hash will change if the contents of this AVLTree changes.

        Returns:
            int: A hash of the contents.
        """
        if self._hash is None:
            self._hash = hash(tuple(self.items()))
        return self._hash

    def _put(self, key, value):
        # type: (Any, Any) -> None

        def _put_helper(self, node, key, value):
            # type: (AVLTree.Node, Any, Any) -> AVLTree.Node
            if node is None:
                self.size += 1
                return self.Node(key, value)
            elif key == node.key:
                node.value = value
                return node
            elif key < node.key:
                node.left = _put_helper(self, node.left, key, value)
            else:
                node.right = _put_helper(self, node.right, key, value)
            return self._balance(node)

        self.root = _put_helper(self, self.root, key, value)
        self._hash = None

    def _get_node(self, key):
        # type: (Any) -> AVLTree.Node

        def _get_node_helper(node, key):
            # type: (AVLTree.Node, Any) -> AVLTree.Node
            # FIXME should the first argument be Optional?
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
        # type: (Any) -> Any

        def _del_helper(self, node, key):
            # type: (AVLTree.Node, Any) -> Tuple[AVLTree.Node, Any]
            value = None
            if node is None:
                raise KeyError(key)
            if key < node.key:
                node.left, value = _del_helper(self, node.left, key)
            elif node.key < key:
                node.right, value = _del_helper(self, node.right, key)
            else:
                if node.left is None and node.right is None:
                    self.size -= 1
                    return None, node.value
                replacement = node
                if node.left is not None:
                    replacement = node.left
                    while replacement.right is not None:
                        replacement = replacement.right
                    replacement_key = replacement.key
                    replacement_value = replacement.value
                    node.left, value = _del_helper(self, node.left, replacement.key)
                elif node.right is not None:
                    replacement = node.right
                    while replacement.left is not None:
                        replacement = replacement.left
                    replacement_key = replacement.key
                    replacement_value = replacement.value
                    node.right, value = _del_helper(self, node.right, replacement.key)
                else:
                    raise KeyError('this should not happen')
                node.key = replacement_key
                node.value = replacement_value
            return self._balance(node), value

        self.root, value = _del_helper(self, self.root, key)
        self._hash = None
        return value

    def _nodes(self):
        # type: () -> Generator[AVLTree.Node, None, None]

        def _nodes_helper(node):
            # type: (Optional[AVLTree.Node]) -> Generator[AVLTree.Node, None, None]
            if node is None:
                return
            yield from _nodes_helper(node.left)
            yield node
            yield from _nodes_helper(node.right)

        yield from _nodes_helper(self.root)

    def clear(self):
        # type: () -> None
        """Remove all elements from the AVLTree."""
        self.size = 0
        self.root = None
        self._hash = None

    def add(self, element):
        # type: (Any) -> None
        """Add an element to the AVLTree (set).

        Parameters:
            element (Any): The element to add.
        """
        self._put(element, None)

    def remove(self, element):
        # type: (Any) -> None
        """Remove an element from the set.

        Parameters:
            element (Any): The element to remove.

        Raises:
            KeyError: If the element is not in the AVLTree.
        """
        self._del(element)

    def discard(self, element):
        # type: (Any) -> None
        """Remove an element from the set if it is present.

        Parameters:
            element (Any): The element to remove.
        """
        try:
            self._del(element)
        except KeyError:
            pass

    def is_disjoint(self, other):
        # type: (Iterable[Any]) -> bool
        """Check if the two sets are disjoint.

        Parameters:
            other (Set[Any]): The other set.

        Returns:
            bool: True if the sets are disjoint.
        """
        return all((element not in other) for element in self)

    def is_subset(self, other):
        # type: (Collection[Any]) -> bool
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
        # type: (Collection[Any]) -> bool
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
        # type: (*Iterable[Any]) -> AVLTree
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
        # type: (*Collection[Any]) -> AVLTree
        """Create the intersection of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.

        Returns:
            AVLTree: The intersection of all the sets.
        """
        tree = AVLTree()
        tree.union_update(min(others, key=len)) # type: ignore
        tree.intersection_update(self)
        tree.intersection_update(*others)
        return tree

    def difference(self, *others):
        # type: (*Iterable[Any]) -> AVLTree
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
        # type: (*Iterable[Any]) -> None
        """Update this set to be the union of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.
        """
        for other in others:
            for element in other:
                self.add(element)

    def intersection_update(self, *others):
        # type: (*Iterable[Any]) -> None
        """Keep only the intersection of this and other sets.

        Parameters:
            *others (Set[Any]): The other sets.
        """
        sorted_others = sorted(others, key=len) # type: ignore
        for element in self:
            if any((element not in other) for other in sorted_others):
                self.remove(element)

    def difference_update(self, *others):
        # type: (*Iterable[Any]) -> None
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
        # type: (Any, Optional[Any]) -> Any
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
        # type: (*Union[AbstractSet[Tuple[Any, Any]], Mapping[Any, Any]]) -> None
        """Add the key and values to the map, overwriting existing values.

        Parameters:
            *mappings (Mapping[Any, Any]): The key-value pairs to be added or updated.
        """
        for mapping in mappings:
            if isinstance(mapping, Mapping):
                for key, value in mapping.items():
                    self._put(key, value)
            else:
                for key, value in mapping:
                    self._put(key, value)

    def get(self, key, default=None):
        # type: (Any, Any) -> Any
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
        # type: (Any, Any) -> Any
        """Remove the key and return the value, or the default if it doesn't exist.

        Parameters:
            key (Any): The key.
            default (Any): The default value to return. Defaults to None.

        Returns:
            Any: The value or the default.
        """
        try:
            value = self._del(key)
            return value
        except KeyError:
            return default

    def keys(self):
        # type: () -> AbstractSet[Any]
        """Create a generator of the keys.

        Yields:
            Any: The keys.
        """
        for node in self._nodes():
            yield node.key

    def values(self):
        # type: () -> ValuesView[Any]
        """Create a generator of the values.

        Yields:
            Any: The values.
        """
        for node in self._nodes():
            yield node.value

    def items(self):
        # type: () -> AbstractSet[Tuple[Any, Any]]
        """Create a generator of the key-value pairs.

        Yields:
            Tuple[Any, Any]: The key-value pairs.
        """
        for node in self._nodes():
            yield node.key, node.value

    def to_set(self):
        # type: () -> Set[Any]
        """Return the elements in a normal set.

        Returns:
            Set[Any]: The resulting set.
        """
        return set(self)

    def to_dict(self):
        # type: () -> Dict[Any, Any]
        """Return the keys and values in a normal dict.

        Returns:
            Dict[Any, Any]: The resulting dict.
        """
        return dict(self.items())

    @staticmethod
    def _balance(node):
        # type: (AVLTree.Node) -> Node
        node.update_metadata()
        if node.balance < -1:
            if node.left.balance == 1:
                node.left = AVLTree._rotate_ccw(node.left)
            return AVLTree._rotate_cw(node)
        elif node.balance > 1:
            if node.right.balance == -1:
                node.right = AVLTree._rotate_cw(node.right)
            return AVLTree._rotate_ccw(node)
        else:
            return node

    @staticmethod
    def _rotate_cw(node):
        # type: (AVLTree.Node) -> Node
        left = node.left
        node.left = left.right
        left.right = node
        node.update_metadata()
        left.update_metadata()
        return left

    @staticmethod
    def _rotate_ccw(node):
        # type: (AVLTree.Node) -> Node
        right = node.right
        node.right = right.left
        right.left = node
        node.update_metadata()
        right.update_metadata()
        return right

    @staticmethod
    def from_set(src_set):
        # type: (Set[Any]) -> AVLTree
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
        # type: (Mapping[Any, Any]) -> AVLTree
        """Create an AVLTree (as a dict) from a dictionary.

        Arguments:
            src_dict (Mapping[Any, Any]): The dictionary.

        Returns:
            AVLTree: The AVLTree.
        """
        tree = AVLTree()
        tree.update(src_dict.items())
        return tree
