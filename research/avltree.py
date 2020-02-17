class AVLTree:

    NONE = 0
    SET = 1
    DICT = 2

    class Node:

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.height = 1
            self.balance = 0
            self.left = None
            self.right = None

        def update_metadata(self):
            left_height = (self.left.height if self.left else 0)
            right_height = (self.right.height if self.right else 0)
            self.height = max(left_height, right_height) + 1
            self.balance = right_height - left_height

    def __init__(self, adt=None, factory=None):
        if adt is None:
            adt = self.NONE
        elif adt not in (self.SET, self.DICT):
            raise ValueError(f'adt must be either AVLTree.SET or AVLTree.DICT')
        self.adt = adt
        self.factory = factory
        if factory is not None:
            self._check_is_map()
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
        self._check_is_map()
        self._put(key, value)

    def __getitem__(self, key):
        self._check_is_map()
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
        self._check_is_map()
        self._del(key)

    def _check_is_set(self):
        if self.adt == self.SET:
            return
        elif self.adt == self.NONE:
            self.adt = self.SET
        else:
            raise RuntimeError('AVLTree is being used as a map, but a set-only function was called')

    def _check_is_map(self):
        if self.adt == self.DICT:
            return
        elif self.adt == self.NONE:
            self.adt = self.DICT
        else:
            raise RuntimeError('AVLTree is being used as a set, but a dict-only function was called')

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
        node.update_metadata()
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
        self.root = self._del_helper(self.root, key)

    def _del_helper(self, node, key):
        if node is None:
            raise KeyError(key)
        elif key < node.key:
            node.left = self._del_helper(node.left, key)
        elif node.key < key:
            node.right = self._del_helper(node.right, key)
        else:
            if node.left is None and node.right is None:
                self.size -= 1
                return None
            replacement = node
            if node.left is not None:
                replacement = node.left
                while replacement.right is not None:
                    replacement = replacement.right
                node.left = self._del_helper(node.left, replacement.key)
            elif node.right is not None:
                replacement = node.right
                while replacement.left is not None:
                    replacement = replacement.left
                node.right = self._del_helper(node.right, replacement.key)
            node.key = replacement.key
            node.value = replacement.value
        node.update_metadata()
        return self._balance(node)

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
        self._check_is_set()
        self._put(element, None)

    def remove(self, element):
        self._check_is_set()
        self._del(element)

    def discard(self, element):
        self._check_is_set()
        try:
            self._del(element)
        except KeyError:
            pass

    def get(self, key, default=None):
        self._check_is_map()
        node = self._get_node(key)
        if node is None:
            return default
        else:
            return node.value

    def keys(self):
        self._check_is_map()
        for node in self._nodes():
            yield node.key

    def values(self):
        self._check_is_map()
        for node in self._nodes():
            yield node.value

    def items(self):
        self._check_is_map()
        for node in self._nodes():
            yield node.key, node.value

    def to_set(self):
        self._check_is_set()
        return set(self)

    def to_dict(self):
        self._check_is_map()
        return dict(self.items())

    @staticmethod
    def _balance(node):
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
        tree = AVLTree(adt=AVLTree.SET)
        for element in src_set:
            tree.add(element)
        return tree

    @staticmethod
    def from_dict(src_dict):
        tree = AVLTree(adt=AVLTree.DICT)
        for key, value in src_dict.items():
            tree[key] = value
        return tree
