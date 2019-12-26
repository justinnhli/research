class AVLTree:

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

    def __init__(self, is_set=False):
        self.is_set = is_set
        self.size = 0
        self.root = None

    def __len__(self):
        return self.size

    def __setitem__(self, key, value):
        self.root = self._put(self.root, key, value)

    def __contains__(self, key):
        return self._get_node(self.root, key) is not None

    def __getitem__(self, key):
        node = self._get_node(self.root, key)
        if node is None:
            return None
        else:
            return node.value

    def __iter__(self):
        yield from (node.key for node in self._nodes(self.root))

    def clear(self):
        self.size = 0
        self.root = None

    def _nodes(self, node):
        if node is None:
            return
        yield from self._nodes(node.left)
        yield node
        yield from self._nodes(node.right)

    def _put(self, node, key, value):
        if node is None:
            return AVLTree.Node(key, value)
        elif key == node.key:
            node.value = value
            return node
        elif key < node.key:
            node.left = self._put(node.left, key, value)
        else:
            node.right = self._put(node.right, key, value)
        node.update_metadata()
        return self._balance(node)

    def _remove(self, node, key):
        if node is None:
            raise ValueError(f'key {key} not in AVLTree')
        elif key == node.key:
            if node.balance < 0:
                pass
            else:
                pass
        elif key < node.key:
            return self._remove(node.left, key)
        else:
            return self._remove(node.right, key)
        node.update_metadata()
        return self._balance()

    def _balance(self, node):
        if node.balance < -1:
            if node.left.balance == 1:
                node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        elif node.balance > 1:
            if node.right.balance == -1:
                node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        else:
            return node

    def _get_node(self, node, key):
        if node is None:
            return None
        elif key == node.key:
            return node
        elif key < node.key:
            return self._get_node(key, node.left)
        else:
            return self._get_node(key, node.right)

    def audit(self, node):
        if node is None:
            return True
        has_errors = False
        has_errors |= (
            (node.left is None or node.left.key < node.key)
            and (node.right is None or node.key < node.right.key)
        )
        has_errors |= (node.height == max(
                (node.left.height if node.left else 0),
                (node.right.height if node.right else 0),
            ) + 1
        )
        has_errors |= (-1 <= node.balance <= 1)
        return not has_errors

    @staticmethod
    def rotate_left(root):
        right = root.right
        root.right = right.left
        right.left = root
        root.update_metadata()
        right.update_metadata()
        return right

    @staticmethod
    def rotate_right(root):
        left = root.left
        root.left = left.right
        left.right = root
        root.update_metadata()
        left.update_metadata()
        return left


def main():
    tree = AVLTree()
    tree[3] = None
    tree[1] = None
    tree[2] = None


if __name__ == '__main__':
    main()
