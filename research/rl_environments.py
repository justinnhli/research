"""Reinforcement learning environments."""

from typing import Any, Iterable, Tuple, List

from .randommixin import RandomMixin
from .data_structures import AVLTree


class Environment:
    """A reinforcement learning environment."""

    def get_state(self):
        # type: () -> State
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
        # type: () -> State
        """Get the current observation.

        See note on get_state() for the difference between the methods.

        Returns:
            State: The observation as a State, or None if the episode has ended
        """
        return self.get_state()

    def get_actions(self):
        # type: () -> List[Action]
        """Get the available actions.

        Returns:
            [Action]: A list of available actions.
        """
        raise NotImplementedError()

    def end_of_episode(self):
        # type: () -> bool
        """Determine if the episode has ended.

        Returns:
            bool: True if the episode has ended.
        """
        return self.get_actions() == []

    def reset(self):
        # type: () -> None
        """Reset the environment entirely.

        The result of calling this method should have the same effect as
        creating a new environment from scratch. Use start_new_episode() to reset the
        environment for a new episode.
        """
        raise NotImplementedError()

    def start_new_episode(self):
        # type: () -> None
        """Reset the environment for a new episode.

        See note on reset() for the difference between the methods.
        """
        raise NotImplementedError()

    def react(self, action):
        # type: (Action) -> float
        """Update the environment to an agent action.

        Assumes the argument is one of the returned actions from get_actions().

        Arguments:
            action (Action): The action the agent took

        Returns:
            float: The reward for the agent for the previous action.
        """
        raise NotImplementedError()

    def visualize(self):
        # type: () -> str
        """Create visualization of the environment.

        Returns:
            str: A visualization of the current state.
        """
        raise NotImplementedError()


class Action(AVLTree):
    """An action in a reinforcement learning environment."""

    def __init__(self, name, **kwargs):
        # type: (str, **Any) -> None
        """Initialize an Action object.

        Arguments:
            name (str): The name of the Action
            **kwargs: Arbitrary key-value pairs
        """
        super().__init__()
        self.update(kwargs)
        self['_name_'] = name

    def __hash__(self):
        # type: () -> int
        return self.contents_hash

    def __str__(self):
        # type: () -> str
        return 'Action("{}", {})'.format(
            self.name,
            ', '.join('{}={}'.format(k, v) for k, v in self.items()),
        )

    @property
    def name(self):
        """Get the name of the Action.

        Returns:
            str: The name of the Action.
        """
        return self['_name_']


class State(AVLTree):
    """A state or observation in a reinforcement learning environment."""

    def __init__(self, **kwargs):
        # type: (**Any) -> None
        """Initialize a State object.

        Arguments:
            **kwargs: Arbitrary key-value pairs
        """
        super().__init__()
        self.update(kwargs)

    def __hash__(self):
        return self.contents_hash


class GridWorld(Environment):
    """A simple, obstacle-free GridWorld environment."""

    def __init__(self, width, height, start, goal, *args, **kwargs):
        # type: (int, int, Tuple[int, int], Tuple[int, int], *Any, **Any) -> None
        """Initialize a GridWorld.

        Arguments:
            width (int): The width of the grid.
            height (int): The height of the grid.
            start (Tuple[int, int]): The starting location. Origin is top left.
            goal (Tuple[int, int]): The goal location. Origin is top left.
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs) # type: ignore
        self.width = width
        self.height = height
        self.start = list(start)
        self.goal = list(goal)
        self.row = start[0]
        self.col = start[1]

    def get_state(self): # noqa: D102
        # type: () -> State
        return State(row=self.row, col=self.col)

    def get_actions(self): # noqa: D102
        # type: () -> List[Action]
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
        # type: () -> None
        self.start_new_episode()

    def start_new_episode(self): # noqa: D102
        # type: () -> None
        self.row = self.start[0]
        self.col = self.start[1]

    def react(self, action): # noqa: D102
        # type: (Action) -> float
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
        # type: () -> str
        raise NotImplementedError


class SimpleTMaze(Environment, RandomMixin):
    """A T-maze environment, with hints on which direction to go."""

    def __init__(self, length, hint_pos, goal_x=0, *args, **kwargs):
        # type: (int, int, int, *Any, **Any) -> None
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
        # type: () -> State
        observation = self.get_observation()
        return State(goal_x=self.goal_x, **observation)

    def get_observation(self): # noqa: D102
        # type: () -> State
        if self.y == self.hint_pos:
            return State(x=self.x, y=self.y, symbol=self.goal_x)
        return State(x=self.x, y=self.y, symbol=0)

    def get_actions(self): # noqa: D102
        # type: () -> List[Action]
        actions = []
        if self.x == 0:
            if self.y < self.length:
                actions.append(Action('up'))
            elif self.y == self.length:
                actions.append(Action('left'))
                actions.append(Action('right'))
        return actions

    def reset(self): # noqa: D102
        # type: () -> None
        self.start_new_episode()

    def start_new_episode(self): # noqa: D102
        # type: () -> None
        self.x = 0
        self.y = 0
        if self.init_goal_x == 0:
            self.goal_x = self.rng.choice([-1, 1])
        else:
            self.goal_x = self.init_goal_x

    def react(self, action): # noqa: D102
        # type: (Action) -> float
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
        # type: () -> str
        lines = []
        for _ in range(self.length + 1):
            lines.append([' ', '_', ' '])
        lines[0][1 + self.goal_x] = '$'
        lines[0][1 - self.goal_x] = '#'
        lines[self.length - self.y][1] = '*'
        lines[self.length - self.hint_pos][1] = '!'
        return '\n'.join(''.join(line) for line in lines)
