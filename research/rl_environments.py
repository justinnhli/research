"""Reinforcement learning environments."""

from .randommixin import RandomMixin
from .data_structures import TreeMultiMap


class Environment:
    """A reinforcement learning environment."""

    def get_state(self):
        """Get the current state.

        This is different from get_observation() in that the state includes
        otherwise hidden information that the agent may not have access to. As
        a rule, the state should be sufficient to completely reconstruct the
        environment. (Although this may not always be necessary, with a random
        seed for example.)

        Returns:
            State: The state, or None if at the end of an episode
        """
        raise NotImplementedError()

    def get_observation(self):
        """Get the current observation.

        See note on get_state() for the difference between the methods.

        Returns:
            State: The observation as a State, or None if the episode has ended
        """
        return self.get_state()

    def get_actions(self):
        """Get the available actions.

        Returns:
            [Action]: A list of available actions.
        """
        raise NotImplementedError()

    def end_of_episode(self):
        """Determine if the episode has ended.

        Returns:
            bool: True if the episode has ended.
        """
        return self.get_actions() == []

    def reset(self):
        """Reset the environment entirely.

        The result of calling this method should have the same effect as
        creating a new environment from scratch. Use start_new_episode() to reset the
        environment for a new episode.
        """
        raise NotImplementedError()

    def start_new_episode(self):
        """Reset the environment for a new episode.

        See note on reset() for the difference between the methods.
        """
        raise NotImplementedError()

    def react(self, action):
        """Update the environment to an agent action.

        Assumes the argument is one of the returned actions from get_actions().

        Arguments:
            action (Action): The action the agent took

        Returns:
            float: The reward for the agent for the previous action.
        """
        raise NotImplementedError()

    def visualize(self):
        """Create visualization of the environment.

        Returns:
            str: A visualization of the current state.
        """
        raise NotImplementedError()


class Action(TreeMultiMap):
    """An action in a reinforcement learning environment."""

    def __init__(self, name, **kwargs):
        """Initialize an Action object.

        Arguments:
            name (str): The name of the Action
            **kwargs: Arbitrary key-value pairs
        """
        super().__init__(multi_level=TreeMultiMap.UNIQUE_KEY, **kwargs)
        self.name = name

    def __hash__(self):
        return hash(tuple([self.name, *self]))

    def __lt__(self, other):
        if self.name < other.name:
            return True
        elif self.name > other.name:
            return False
        else:
            return super().__lt__(other)

    def __eq__(self, other):
        # pylint: disable = protected-access
        if not isinstance(other, Action):
            return False
        return self.name == other.name and super().__eq__(other)

    def __str__(self):
        return 'Action("{}", {})'.format(
            self.name,
            ', '.join('{}={}'.format(k, v) for k, v in self.items()),
        )


class State(TreeMultiMap):
    """A state or observation in a reinforcement learning environment."""

    def __init__(self, **kwargs):
        """Initialize a State object.

        Arguments:
            **kwargs: Arbitrary key-value pairs
        """
        super().__init__(multi_level=TreeMultiMap.UNIQUE_KEY, **kwargs)


class GridWorld(Environment):
    """A simple, obstacle-free GridWorld environment."""

    def __init__(self, width, height, start, goal, *args, **kwargs):
        """Initialize a GridWorld.

        Arguments:
            width (int): The width of the grid.
            height (int): The height of the grid.
            start (Tuple[int, int]): The starting location. Origin is top left.
            goal (Tuple[int, int]): The goal location. Origin is top left.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.start = list(start)
        self.goal = list(goal)
        self.row = start[0]
        self.col = start[1]

    def get_state(self): # noqa: D102
        return State(row=self.row, col=self.col)

    def get_actions(self): # noqa: D102
        if [self.row, self.col] == self.goal:
            return []
        actions = []
        if self.row > 0:
            actions.append(Action('up'))
        if self.row < self.height - 1:
            actions.append(Action('down'))
        if self.col > 0:
            actions.append(Action('left'))
        if self.col < self.width - 1:
            actions.append(Action('right'))
        return actions

    def reset(self): # noqa: D102
        self.start_new_episode()

    def start_new_episode(self): # noqa: D102
        self.row = self.start[0]
        self.col = self.start[1]

    def react(self, action=None): # noqa: D102
        assert action in self.get_actions()
        if action.name == 'up':
            self.row = max(0, self.row - 1)
        elif action.name == 'down':
            self.row = min(self.height - 1, self.row + 1)
        elif action.name == 'left':
            self.col = max(0, self.col - 1)
        elif action.name == 'right':
            self.col = min(self.width - 1, self.col + 1)
        if [self.row, self.col] == self.goal:
            return 1
        else:
            return -1

    def visualize(self): # noqa: D102
        raise NotImplementedError


def augment_state(state, memories, prefix):
    """Add memory items to states and observations.

    Note that we need to remove existing 'memory_' attributes because
    super().get_state() could call the overridden get_observation().

    Arguments:
        state (State): The state to augment.
        memories (List[Any]): The memories to augment with.
        prefix (str): The prefix to use for memories.

    Returns:
        State: The state with memory items.
    """
    for key in list(state.keys()):
        if key.startswith(prefix):
            del state[key]
    memories = {(prefix + '{}'.format(i)): value for i, value in enumerate(memories)}
    return State(**memories, **state)


def gating_memory(cls):
    """Decorate an Environment to include a gating memory.

    This decorator function takes a class (and some parameters) and, on the
    fly, creates a subclass with additional "memory" elements. Specifically, it
    augments the observation with additional attributes, which the agent can
    use as part of its state. The subclass also provides an additional "gate"
    action which the agent can use to change the contents of memory.

    Since the gating of memory is an "internal" action, a different reward may
    be given for the "gate" action.

    Arguments:
        cls (class): The Environment superclass.

    Returns:
        class: A subclass with a gating memory.
    """
    assert issubclass(cls, Environment)

    class GatingMemoryMetaEnvironment(cls):
        """A subclass to add a gating memory to an Environment."""

        # pylint: disable = missing-docstring

        ATTR_PREFIX = 'memory_'

        def __init__(self, num_memory_slots=1, reward=0, *args, **kwargs):
            """Initialize a GatingMemoryMetaEnvironment.

            Arguments:
                num_memory_slots (int): The amount of memory. Defaults to 1.
                reward (float): The reward for an internal action. Defaults to 0.
                *args: Arbitrary positional arguments.
                **kwargs: Arbitrary keyword arguments.
            """
            # pylint: disable = keyword-arg-before-vararg
            super().__init__(*args, **kwargs)
            self.reward = reward
            self.memories = num_memory_slots * [None]

        def get_state(self):
            state = super().get_state()
            if state is None:
                return None
            return augment_state(state, self.memories, self.ATTR_PREFIX)

        def get_observation(self):
            observation = super().get_observation()
            if observation is None:
                return None
            return augment_state(observation, self.memories, self.ATTR_PREFIX)

        def get_actions(self):
            actions = super().get_actions()
            if actions == []:
                return actions
            observations = super().get_observation()
            if observations is None:
                return actions
            for slot_num in range(len(self.memories)):
                for attr in observations:
                    if attr.startswith(self.ATTR_PREFIX):
                        continue
                    actions.append(Action('gate', slot=slot_num, attribute=attr))
            return actions

        def reset(self):
            super().reset()
            self.memories = len(self.memories) * [None]

        def start_new_episode(self):
            super().start_new_episode()
            self.memories = len(self.memories) * [None]

        def react(self, action):
            if action.name == 'gate':
                self.memories[action.slot] = getattr(super().get_observation(), action.attribute)
                return self.reward
            else:
                return super().react(action)

    return GatingMemoryMetaEnvironment


def fixed_long_term_memory(cls):
    """Decorate an Environment to include a long-term memory of fixed size.

    Arguments:
        cls (class): The Environment superclass.

    Returns:
        class: A subclass with a fixed-sized long-term memory.
    """
    assert issubclass(cls, Environment)

    class LongTermMemoryMetaEnvironment(cls):
        """A subclass to add a long-term memory to an Environment."""

        # pylint: disable = missing-docstring

        WM_PREFIX = 'wm_' # pylint: disable = invalid-name
        LTM_PREFIX = 'ltm_' # pylint: disable = invalid-name

        def __init__(self, num_wm_slots=1, num_ltm_slots=1, reward=0, *args, **kwargs):
            """Initialize a LongTermMemoryMetaEnvironment.

            Arguments:
                num_wm_slots (int): The amount of working memory. Defaults to 1.
                num_ltm_slots (int): The amount of long-term memory. Defaults to 1.
                reward (float): The reward for an internal action. Defaults to 0.
                *args: Arbitrary positional arguments.
                **kwargs: Arbitrary keyword arguments.
            """
            # pylint: disable = keyword-arg-before-vararg
            super().__init__(*args, **kwargs)
            self.reward = reward
            self.wm = num_wm_slots * [None] # pylint: disable = invalid-name
            self.ltm = num_ltm_slots * [None]

        def get_state(self):
            state = super().get_state()
            if state is None:
                return None
            state = augment_state(state, self.wm, self.WM_PREFIX)
            state = augment_state(state, self.ltm, self.LTM_PREFIX)
            return state

        def get_observation(self):
            observation = super().get_observation()
            if observation is None:
                return None
            return augment_state(observation, self.wm, self.WM_PREFIX)

        def get_actions(self):
            actions = super().get_actions()
            if actions == []:
                return actions
            observations = super().get_observation()
            if observations is None:
                return actions
            for slot_num in range(len(self.ltm)):
                for attr in observations:
                    if attr.startswith(self.WM_PREFIX):
                        continue
                    actions.append(Action('store', slot=slot_num, attribute=attr))
            for wm_slot_num in range(len(self.wm)):
                for ltm_slot_num in range(len(self.ltm)):
                    actions.append(Action('retrieve', wm_slot=wm_slot_num, ltm_slot=ltm_slot_num))
            return actions

        def reset(self):
            super().reset()
            self.wm = len(self.wm) * [None]
            self.ltm = len(self.ltm) * [None]

        def start_new_episode(self):
            super().start_new_episode()
            self.wm = len(self.wm) * [None]
            self.ltm = len(self.ltm) * [None]

        def react(self, action):
            if action.name == 'store':
                self.ltm[action.slot] = getattr(super().get_observation(), action.attribute)
                return self.reward
            elif action.name == 'retrieve':
                self.wm[action.wm_slot] = self.ltm[action.ltm_slot]
                return self.reward
            else:
                return super().react(action)

    return LongTermMemoryMetaEnvironment


class SimpleTMaze(Environment, RandomMixin):
    """A T-maze environment, with hints on which direction to go."""

    def __init__(self, length, hint_pos, goal_x=0, *args, **kwargs):
        """Initialize the TMaze.

        Arguments:
            length (int): The length of the hallway before the choice point.
            hint_pos (int): The location of the hint.
            goal_x (int): The location of the goal. Must be -1 or 1. If left
                to default of 0, goal_x is chosen at random.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        # pylint: disable = keyword-arg-before-vararg
        assert 0 <= hint_pos < length
        super().__init__(*args, **kwargs)
        self.length = length
        self.hint_pos = hint_pos
        self.x = 0
        self.y = 0
        self.init_goal_x = goal_x
        self.goal_x = 0 # dummy value

    def get_state(self): # noqa: D102
        observation = self.get_observation()
        return State(goal_x=self.goal_x, **observation)

    def get_observation(self): # noqa: D102
        if self.y == self.hint_pos:
            return State(x=self.x, y=self.y, symbol=self.goal_x)
        return State(x=self.x, y=self.y, symbol=0)

    def get_actions(self): # noqa: D102
        actions = []
        if self.x == 0:
            if self.y < self.length:
                actions.append(Action('up'))
            elif self.y == self.length:
                actions.append(Action('left'))
                actions.append(Action('right'))
        return actions

    def reset(self): # noqa: D102
        self.start_new_episode()

    def start_new_episode(self): # noqa: D102
        self.x = 0
        self.y = 0
        if self.init_goal_x == 0:
            self.goal_x = self.rng.choice([-1, 1])
        else:
            self.goal_x = self.init_goal_x

    def react(self, action): # noqa: D102
        assert action in self.get_actions()
        if action.name == 'up':
            self.y += 1
        elif action.name == 'right':
            self.x += 1
        elif action.name == 'left':
            self.x -= 1
        if self.y == self.length:
            if self.x == self.goal_x:
                return 10
            elif self.x == -self.goal_x:
                return -10
            else:
                return -1
        elif action.name == 'gate':
            return -0.050
        else:
            return -1

    def visualize(self): # noqa: D102
        lines = []
        for _ in range(self.length + 1):
            lines.append([' ', '_', ' '])
        lines[0][1 + self.goal_x] = '$'
        lines[0][1 - self.goal_x] = '#'
        lines[self.length - self.y][1] = '*'
        lines[self.length - self.hint_pos][1] = '!'
        return '\n'.join(''.join(line) for line in lines)
