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

    class Node:

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.parent = None
            self.left = None
            self.right = None
            self.height = 1
            self.balance = 0

        def __lt__(self, other):
            return (self.key, self.value) < (other.key, other.value)

        def __iter__(self):
            if self.left:
                yield from self.left
            yield self.key
            if self.right:
                yield from self.right

        def __str__(self):
            return f'Node({self.key}, {self.value})'

        def find_first(self, key):
            if self.left and key < self.key:
                return self.left.find_first(key)
            elif self.right and self.key < key:
                return self.right.find_first(key)
            elif self.key == key:
                if self.left:
                    result = self.left.find_first(key)
                    if result:
                        return result
                    else:
                        return self
                else:
                    return self
            return None

        def find_last(self, key):
            if self.left and key < self.key:
                return self.left.find_last(key)
            if self.right and self.key < key:
                return self.right.find_last(key)
            if self.key == key:
                if self.right:
                    result = self.right.find_last(key)
                    if result:
                        return result
                    else:
                        return self
                else:
                    return self
            return None

        def contains(self, key):
            return self.find_first(key) is not None

        def yield_all(self, key):
            if self.left and key < self.key:
                yield from self.left.yield_all(key)
            elif self.right and self.key < key:
                yield from self.right.yield_all(key)
            elif self.key == key:
                if self.left:
                    yield from self.left.yield_all(key)
                yield self.value
                if self.right:
                    yield from self.right.yield_all(key)

        def keys(self):
            if self.left:
                yield from self.left.keys()
            yield self.key
            if self.right:
                yield from self.right.keys()

        def values(self):
            if self.left:
                yield from self.left.values()
            yield self.value
            if self.right:
                yield from self.right.values()

        def items(self):
            if self.left:
                yield from self.left.items()
            yield (self.key, self.value)
            if self.right:
                yield from self.right.items()

        def update_metadata(self):
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

    def __init__(self):
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, key, value):
        new_node = TreeMultiMap.Node(key, value)
        if self.root is None:
            self.root = new_node
            self.root.update_metadata()
        else:
            self._add(self.root, new_node)
        self.size += 1

    def _add(self, node, new_node):
        # insert
        if node is None:
            return new_node
        elif new_node < node:
            node.left = self._add(node.left, new_node)
            node.left.parent = node
        elif node < new_node:
            node.right = self._add(node.right, new_node)
            node.right.parent = node
        else:
            raise ValueError('key-value already exists in map')
        # balance
        return self._balance(node)

    def _balance(self, node):
        node.update_metadata()
        if node.balance < -1:
            if node.left.balance == -1:
                return self._rotate_right(node)
            elif node.left.balance == 1:
                self._rotate_left(node.left)
                return self._rotate_right(node)
            assert False, 'This should never happen'
            return None
        elif node.balance > 1:
            if node.right.balance == 1:
                return self._rotate_left(node)
            elif node.right.balance == -1:
                self._rotate_right(node.right)
                return self._rotate_left(node)
            assert False, 'This should never happen'
            return None
        else:
            return node

    def _rotate_left(self, node):
        r"""Perform a left rotation.

        If the node is C (left/right child of A), go from:
              A
              |
              C
             / \
            B   E
               / \
              D   F
        to:
              A
              |
              E
             / \
            C   F
           / \
          B   D
        and return E.
        """
        # definitions
        node_c = node
        node_a = node_c.parent
        node_e = node_c.right
        node_d = node_e.left
        # rotate
        node_c.right = node_d
        if node_d:
            node_d.parent = node_c
        node_e.left = node_c
        node_c.parent = node_e
        node_e.parent = node_a
        if node_a:
            if node_a.left == node_c:
                node_a.left = node_e
            else:
                node_a.right = node_e
        else:
            self.root = node_e
        # update metadata
        node_c.update_metadata()
        node_e.update_metadata()
        if node_a:
            node_a.update_metadata()
        return node_e

    def _rotate_right(self, node):
        r"""Perform a  rotation.

        If the node is E (left/right child of A), go from:
              A
              |
              E
             / \
            C   F
           / \
          B   D
        to:
              A
              |
              C
             / \
            B   E
               / \
              D   F
        and return C.
        """
        # definitions
        node_e = node
        node_a = node_e.parent
        node_c = node_e.left
        node_d = node_c.right
        # rotate
        node_e.left = node_d
        if node_d:
            node_d.parent = node_e
        node_c.right = node_e
        node_e.parent = node_c
        node_c.parent = node_a
        if node_a:
            if node_a.left == node_e:
                node_a.left = node_c
            else:
                node_a.right = node_c
        else:
            self.root = node_c
        # update metadata
        node_e.update_metadata()
        node_c.update_metadata()
        if node_a:
            node_a.update_metadata()
        return node_c

    def __contains__(self, key):
        if self.root is None:
            return False
        return self.root.contains(key)

    def __iter__(self):
        if self.root is None:
            return
        yield from self.root

    def __getitem__(self, key):
        if self.root is None:
            return
        yield from self.root.yield_all(key)

    def get_first(self, key):
        if self.root is None:
            return None
        node = self.root.find_first(key)
        if node is None:
            return None
        else:
            return node.value

    def get_last(self, key):
        if self.root is None:
            return None
        node = self.root.find_last(key)
        if node is None:
            return None
        else:
            return node.value

    def keys(self):
        if self.root is None:
            return
        yield from self.root.keys()

    def values(self):
        if self.root is None:
            return
        yield from self.root.values()

    def items(self):
        if self.root is None:
            return
        yield from self.root.items()

    def visualize(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node is None:
            return
        print(depth * '  ' + f'{node.key}: {node.value}')
        if node.left:
            print((depth + 1) * '  ' + 'left:')
            self.visualize(node.left, depth + 1)
        if node.right:
            print((depth + 1) * '  ' + 'right:')
            self.visualize(node.right, depth + 1)

    @staticmethod
    def from_dict(src_dict):
        tmm = TreeMultiMap()
        for key, value in src_dict.items():
            tmm.add(key, value)
        return tmm
