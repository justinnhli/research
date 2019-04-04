"""Utility data structures."""


class UnionFind:
    """UnionFind for discrete sets."""

    def __init__(self, nodes=None):
        """Initialize the UnionFind.

        Arguments:
            nodes (Hashable): The object to be hashed.
        """
        if nodes is None:
            nodes = []
        self.parents = {node: node for node in nodes}

    def __len__(self):
        return len(self.parents)

    def __contains__(self, node):
        return node in self.parents

    def __getitem__(self, node):
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
        return iter(self.parents)

    def __bool__(self):
        return bool(self.parents)

    def union(self, node1, node2):
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
        """Check if two members are in the same set.

        Arguments:
            node1 (Hashable): A member of one set.
            node2 (Hashable): A member of the other set.

        Returns:
            bool: True if the members are in the same set.
        """
        return self[node1] == self[node2]

    def add(self, node, parent=None):
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


class TreeMultiMap:
    """A tree-based multi-map."""

    class Node:
        """A tree node."""

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

        def __lt__(self, other):
            return (self.key, self.value) < (other.key, other.value)

        def __contains__(self, key):
            return self.get_first(key) is not None

        def __iter__(self):
            if self.left:
                yield from self.left # pylint: disable = not-an-iterable
            yield self
            if self.right:
                yield from self.right # pylint: disable = not-an-iterable

        def __str__(self):
            return f'Node({self.key}, {self.value})'

        @property
        def childless(self):
            """Whether a node has children.

            Returns:
                bool: True if the node has no children, False otherwise.

            """
            return self.left is None and self.right is None

        def get_first(self, key):
            """Find the first node in the subtree with a given key.

            Arguments:
                key (Any): The key to find.

            Returns:
                Node: The first node with the given key, or None.
            """
            if self.left and key < self.key:
                return self.left.get_first(key)
            elif self.right and self.key < key:
                return self.right.get_first(key)
            elif self.key == key:
                if self.left:
                    result = self.left.get_first(key)
                    if result:
                        return result
                    else:
                        return self
                else:
                    return self
            return None

        def get_last(self, key):
            """Find the last node in the subtree with a given key.

            Arguments:
                key (Any): The key to find.

            Returns:
                Node: The last node with the given key, or None.
            """
            if self.left and key < self.key:
                return self.left.get_last(key)
            if self.right and self.key < key:
                return self.right.get_last(key)
            if self.key == key:
                if self.right:
                    result = self.right.get_last(key)
                    if result:
                        return result
                    else:
                        return self
                else:
                    return self
            return None

        def yield_all(self, key):
            """Iterate through all nodes with a given key.

            Arguments:
                key (Any): The key to find.

            Yields:
                Node: The nodes with the given key.
            """
            if self.left and key < self.key:
                yield from self.left.yield_all(key)
            elif self.right and self.key < key:
                yield from self.right.yield_all(key)
            elif self.key == key:
                if self.left:
                    yield from self.left.yield_all(key)
                yield self
                if self.right:
                    yield from self.right.yield_all(key)

        def update_height_balance(self):
            """Update the height and balance of this node."""
            if self.left:
                left_height = self.left.height
            else:
                left_height = 0
            if self.right:
                right_height = self.right.height
            else:
                right_height = 0
            self.balance = right_height - left_height
            self.height = max(left_height, right_height) + 1

    def __init__(self, **kwargs):
        """Initialize the TreeMultiMap.

        Arguments:
            **kwargs: Arbitrary keyword arguments.
        """
        self.root = None
        self.size = 0
        for key, value in kwargs.items():
            self.add(key, value)

    def __len__(self):
        return self.size

    def __contains__(self, key):
        if self.root is None:
            return False
        return key in self.root

    def __iter__(self):
        yield from self.keys()

    def __getitem__(self, key):
        if self.root is None:
            return None
        return next(self.root.yield_all(key)).value

    def _balance(self, node):
        node.update_height_balance()
        if node.balance < -1:
            return self._balance_left(node)
        elif node.balance > 1:
            return self._balance_right(node)
        else:
            return node

    def _balance_left(self, node):
        if node.left.balance == -1:
            return self._rotate_right(node)
        elif node.left.balance == 0:
            if node.left.childless:
                return node
            else:
                return self._rotate_right(node)
        elif node.left.balance == 1:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        assert False, 'This should never happen'
        return None

    def _balance_right(self, node):
        if node.right.balance == 1:
            return self._rotate_left(node)
        elif node.right.balance == 0:
            if node.right.childless:
                return node
            else:
                return self._rotate_left(node)
        elif node.right.balance == -1:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        assert False, 'This should never happen'
        return None

    def add(self, key, value):
        """Associate the value with the key.

        Arguments:
            key (Any): The key.
            value (Any): The value.
        """
        self.root = self._add(key, value, self.root)
        self.size += 1

    def _add(self, key, value, node):
        if node is None:
            return TreeMultiMap.Node(key, value)
        elif (key, value) < (node.key, node.value):
            node.left = self._add(key, value, node.left)
        elif (key, value) > (node.key, node.value):
            node.right = self._add(key, value, node.right)
        else:
            raise ValueError('key-value already exists in map')
        return self._balance(node)

    def get(self, key, default=None):
        """Get a key or return a default value.

        Arguments:
            key (Any): The key to get.
            default (Any): The value if the key does not exist. Defaults out None.

        Returns:
            Any: The value associated with the key.
        """
        if key in self:
            return self.get_first(key)
        else:
            return default

    def get_first(self, key):
        """Find the first value with a given key.

        Arguments:
            key (Any): The key to find.

        Returns:
            Node: The first value with the given key, or None.
        """
        if self.root is None:
            return None
        node = self.root.get_first(key)
        if node is None:
            return None
        else:
            return node.value

    def get_last(self, key):
        """Find the last value with a given key.

        Arguments:
            key (Any): The key to find.

        Returns:
            Node: The last value with the given key, or None.
        """
        if self.root is None:
            return None
        node = self.root.get_last(key)
        if node is None:
            return None
        else:
            return node.value

    def keys(self):
        """Iterate through all keys.

        Yields:
            Any: The keys.
        """
        if self.root is None:
            return
        for node in self.root:
            yield node.key

    def values(self):
        """Iterate through all values.

        Yields:
            Any: The values.
        """
        if self.root is None:
            return
        for node in self.root:
            yield node.value

    def items(self):
        """Iterate through all key-value pairs.

        Yields:
            Tuple[Any, Any]: The keys and values.
        """
        if self.root is None:
            return
        for node in self.root:
            yield (node.key, node.value)

    @staticmethod
    def _rotate_left(node):
        r"""Perform a left rotation.

        If the node is B, go from:
              B
             / \
            A   D
               /
              C
        to:
                D
               /
              B
             / \
            A   C
        and return D.
        """
        # definitions
        node_b = node
        node_d = node_b.right
        node_c = node_d.left
        # rotate
        node_b.right = node_c
        node_d.left = node_b
        node_b.update_height_balance()
        node_d.update_height_balance()
        return node_d

    @staticmethod
    def _rotate_right(node):
        r"""Perform a right rotation.

        If the node is C, go from:
              C
             / \
            A   D
             \
              B
        to:
            A
             \
              C
             / \
            B   D
        and return A.
        """
        # definitions
        node_c = node
        node_a = node_c.left
        node_b = node_a.right
        # rotate
        node_c.left = node_b
        node_a.right = node_c
        node_c.update_height_balance()
        node_a.update_height_balance()
        return node_a

    @staticmethod
    def from_dict(src_dict):
        """Create a TreeMultiMap from a dictionary.

        Arguments:
            src_dict (Mapping[Any, Any]): The dictionary.

        Returns:
            TreeMultiMap: The TreeMultiMap.
        """
        tmm = TreeMultiMap()
        for key, value in src_dict.items():
            tmm.add(key, value)
        return tmm
