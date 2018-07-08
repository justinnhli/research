import sys
from collections import namedtuple
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_environments import State, Action, Environment
from research.randommixin import RandomMixin
from research.data_structures import UnionFind


class Location:
    """A container class for a 2D coordinate in a grid."""

    def __init__(self, index, size):
        self.index = index
        self.size = size
        self.up_wall = True
        self.down_wall = True
        self.left_wall = True
        self.right_wall = True

    @property
    def row(self):
        """Get the column of this location."""
        return self.index // self.size

    @property
    def col(self):
        """Get the column of this location."""
        return self.index % self.size


class RandomMaze(Environment, RandomMixin):

    def __init__(self, size=5, randomize=False, representation='symbol', *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        self.size = size
        self.randomize = randomize
        self.representation = representation
        # variables
        self.start = None
        self.goal = None
        self.location = None
        self.password = None
        self.locations = []
        self.goal_map = {}
        self.password_map = {}
        self.reset()

    @property
    def row(self):
        return self.to_coords(self.location)[0]

    @property
    def col(self):
        return self.to_coords(self.location)[1]

    @property
    def num_locations(self):
        return self.size * self.size

    def get_state(self): # noqa: D102
        return self.get_observation()

    def get_observation(self): # noqa: D102
        if self.representation == 'coords':
            coords = self.to_coords(self.location)
            return State(row=coords[0], col=coords[1])
        elif self.representation == 'symbol':
            return State(location=self.location)
        assert False
        return None

    def get_actions(self): # noqa: D102
        if self.location == self.goal:
            return []
        return [
            Action('up'),
            Action('down'),
            Action('left'),
            Action('right'),
        ]

    def react(self, action): # noqa: D102
        #assert action.name in ['up', 'down', 'left', 'right', 'no-op']
        if action.name == 'up' and self.row > 0:
            self.location -= self.size
        elif action.name == 'down' and self.row < self.size - 1:
            self.location += self.size
        elif action.name == 'left' and self.col > 0:
            self.location -= 1
        elif action.name == 'right' and self.col < self.size - 1:
            self.location += 1
        if self.location == self.goal:
            return 100
        else:
            return -1

    def reset(self): # noqa: D102
        self.rng.seed(self.random_seed)
        self.start_new_episode()

    def start_new_episode(self): # noqa: D102
        if not self.randomize:
            self.rng.seed(self.random_seed)
        self.start = self.rng.randrange(self.num_locations)
        self.location = self.start
        self.goal = self._random_location([self.start])
        self.password = self._random_location([self.start, self.goal])
        self._generate_maze()
        self.goal_map = self._solve_for(self.goal)
        self.password_map = self._solve_for(self.password)

    def to_coords(self, index):
        return index // self.size, index % self.size

    def to_index(self, row, col):
        return row * self.size + col

    def _random_location(self, taboo=None):
        if taboo is None:
            taboo = set()
        else:
            taboo = set(taboo)
        result = self.rng.randrange(self.num_locations)
        while result in taboo:
            result = self.rng.randrange(self.num_locations)
        return result

    def _generate_maze(self):
        """Generate the maze using randomized Kruskal's algorithm."""
        # initialize maze with walls everywhere
        self.locations = []
        for i in range(self.num_locations):
            self.locations.append(Location(i, self.size))
        # create the set of all walls
        walls = set()
        for outer in range(self.size):
            # horizontals
            walls |= set(
                (self.to_index(outer, inner), self.to_index(outer, inner + 1))
                for inner in range(self.size - 1)
            )
            # verticals
            walls |= set(
                (self.to_index(inner, outer), self.to_index(inner + 1, outer))
                for inner in range(self.size - 1)
            )
        # remove walls until maze is connected
        union_find = UnionFind(list(range(self.num_locations)))
        num_components = self.num_locations
        while num_components > 1:
            loc1, loc2 = walls.pop()
            if union_find.same(loc1, loc2):
                continue
            if loc2 - loc1 == 1:
                # horizontal
                self.locations[loc1].right_wall = False
                self.locations[loc2].left_wall = False
            else:
                # vertical
                self.locations[loc1].down_wall = False
                self.locations[loc2].up_wall = False
            union_find.union(loc1, loc2)
            num_components -= 1

    def _solve_for(self, index):
        """Solve the maze with breadth first search from a location.

        Arguments:
            index (int): The index of the location to solve for.

        Returns:
            Dict[int, str]: A dictionary of directions to take at each location.
        """
        opposite = dict(zip(
            ['up', 'down', 'left', 'right'],
            ['down', 'up', 'right', 'left'],
        ))
        path = {i: None for i in range(self.num_locations)}
        queue = [(self.goal, None)]
        while queue:
            index, source = queue.pop(0)
            loc = self.locations[index]
            for direction in ['up', 'down', 'left', 'right']:
                if source != direction and not getattr(loc, direction + '_wall'):
                    next_index = getattr(loc, direction + '_index')
                    if next_index is not None and path[next_index] is None:
                        next_action = opposite[direction]
                        path[next_index] = next_action
                        queue.append((next_index, next_action))
        return path

    def visualize(self): # noqa: D102
        arrows = {
            None: '@',
            'up': '^',
            'down': 'v',
            'left': '<',
            'right': '>',
        }
        lines = []
        lines.append('   ' + ' '.join(str(c % 10) for c in range(self.size)))
        for r in range(self.size):
            row_indexes = [self.to_index(r, c) for c in range(self.size)]
            # wall above each row
            lines.append('  #' + '#'.join(('#' if self.locations[i].up_wall else ' ') for i in row_indexes) + '#')
            # the row itself
            lines.append(
                str(r % 10) + ' #' + ''.join(
                    (arrows[self.goal_map[i]] + ('#' if self.locations[i].right_wall else ' ')) for i in row_indexes
                )
            )
        lines.append('  ' + (2 * self.size + 1) * '#')
        return '\n'.join(lines)


BufferProperties = namedtuple(
    'BufferProperties',
    [
        'copyable',
        'overwritable',
        'appendable',
        'deletable',
        'defaults',
    ],
)


def memory_architecture(cls):
    """Decorate an Environment to become a memory architecture.

    Arguments:
        cls (class): The Environment superclass.

    Returns:
        class: A subclass with a memory architecture.
    """
    assert issubclass(cls, Environment)

    class MemoryArchitectureMetaEnvironment(cls):
        """A subclass to add a long-term memory to an Environment."""

        # pylint: disable = missing-docstring

        BUFFERS = {
            'perceptual': BufferProperties(
                copyable=True,
                overwritable=False,
                appendable=False,
                deletable=False,
                defaults={},
            ),
            'query': BufferProperties(
                copyable=False,
                overwritable=False,
                appendable=True,
                deletable=True,
                defaults={},
            ),
            'retrieval': BufferProperties(
                copyable=True,
                overwritable=False,
                appendable=False,
                deletable=False,
                defaults={},
            ),
            'action': BufferProperties(
                copyable=False,
                overwritable=True,
                appendable=True,
                deletable=False,
                defaults={
                    'name': 'no-op',
                },
            ),
        }

        def __init__(self, explicit_actions=False, load_goal_path=False, map_representation='symbol', *args, **kwargs): # noqa: D102
            # pylint: disable = keyword-arg-before-vararg
            # parameters
            self.explicit_actions = explicit_actions
            self.load_goal_path = load_goal_path
            self.map_representation = map_representation
            # variables
            self.ltm = set()
            self.buffers = {}
            self.clear_buffers()
            super().__init__(*args, **kwargs)

        @property
        def slots(self):
            for buf, attrs in sorted(self.buffers.items()):
                for attr, val in attrs.items():
                    yield buf, attr, val

        def clear_buffers(self):
            self.buffers = {}
            for buf, props in self.BUFFERS.items():
                self.buffers[buf] = {}
                for key, value in props.defaults.items():
                    self.buffers[buf][key] = value

        def to_dict(self):
            return {buf + '_' + attr: val for buf, attr, val in self.slots}

        def get_state(self): # noqa: D102
            return State(**self.to_dict())

        def get_observation(self): # noqa: D102
            return State(**self.to_dict())

        def _generate_output_actions(self):
            actions = []
            for action in super().get_actions():
                actions.append(Action('act', super_name=action.name))
            return actions

        def _generate_copy_actions(self):
            actions = []
            for src_buf, src_props in self.BUFFERS.items():
                if not src_props.copyable:
                    continue
                for src_attr in self.buffers[src_buf]:
                    for dst_buf, dst_prop in self.BUFFERS.items():
                        if dst_prop.appendable:
                            if src_attr not in self.buffers[dst_buf]:
                                actions.append(Action(
                                    'copy',
                                    src_buf=src_buf,
                                    src_attr=src_attr,
                                    dst_buf=dst_buf,
                                    dst_attr=src_attr,
                                ))
                        if dst_prop.overwritable:
                            for dst_attr in self.buffers[dst_buf]:
                                actions.append(Action(
                                    'copy',
                                    src_buf=src_buf,
                                    src_attr=src_attr,
                                    dst_buf=dst_buf,
                                    dst_attr=dst_attr,
                                ))
            return actions

        def _generate_delete_actions(self):
            actions = []
            for buf, prop in self.BUFFERS.items():
                if not prop.deletable:
                    continue
                for attr in self.buffers[buf]:
                    actions.append(Action(
                        'delete',
                        buf=buf,
                        attr=attr,
                    ))
            return actions

        def get_actions(self): # noqa: D102
            actions = super().get_actions()
            if actions == []:
                return actions
            actions = []
            if self.explicit_actions:
                actions.extend(self._generate_output_actions())
            actions.extend(self._generate_copy_actions())
            actions.extend(self._generate_delete_actions())
            return actions

        def reset(self): # noqa: D102
            super().reset()
            self.clear_buffers()

        def start_new_episode(self): # noqa: D102
            super().start_new_episode()
            self.clear_buffers()
            if self.load_goal_path:
                self.ltm = set()
                if self.map_representation == 'symbol':
                    LocDir = namedtuple('LocationDirection', ['location', 'direction'])
                    for location, direction in self.goal_map.items():
                        self.ltm.add(LocDir(location, direction))
                else:
                    LocDir = namedtuple('LocationDirection', ['row', 'col', 'direction'])
                    for location, direction in self.goal_map.items():
                        self.ltm.add(LocDir(
                            location // self.size,
                            location % self.size,
                            direction,
                        ))
            self._sync_input_buffers()

        def react(self, action): # noqa: D102
            assert 0 <= self.row < self.size, str('index is {} ({}, {})'.format(self.location, self.row, self.col))
            assert 0 <= self.col < self.size, str('index is {} ({}, {})'.format(self.location, self.row, self.col))
            # handle internal actions
            query_changed = False
            if action.name == 'act':
                self.buffers['action']['name'] = action.super_name
            elif action.name == 'copy':
                val = self.buffers[action.src_buf][action.src_attr]
                self.buffers[action.dst_buf][action.dst_attr] = val
                if action.dst_buf == 'query':
                    query_changed = True
            elif action.name == 'delete':
                del self.buffers[action.buf][action.attr]
                if action.buf == 'query':
                    query_changed = True
            # update memory buffers
            if query_changed:
                self._query_ltm()
            else:
                self._clear_ltm_buffers()
            # interface with underlying environment
            real_action = Action(**self.buffers['action'])
            reward = super().react(real_action)
            self._sync_input_buffers()
            return reward

        def _query_ltm(self):
            if self.buffers['query']:
                candidates = []
                for candidate in self.ltm:
                    candidate_dict = candidate._asdict()
                    match = all(
                        attr in candidate_dict and candidate_dict[attr] == val
                        for attr, val in self.buffers['query'].items()
                    )
                    if match:
                        candidates.append(candidate_dict)
                if candidates:
                    self.buffers['retrieval'] = self.rng.choice(candidates)

        def _clear_ltm_buffers(self):
            self.buffers['query'] = {}
            self.buffers['retrieval'] = {}

        def _sync_input_buffers(self):
            # update input buffers
            self.buffers['perceptual'] = {}
            for attr, val in sorted(super().get_observation().as_dict().items()):
                self.buffers['perceptual'][attr] = val
            # clear output buffer
            self.buffers['action'] = {}
            self.buffers['action']['name'] = 'no-op'

    return MemoryArchitectureMetaEnvironment
