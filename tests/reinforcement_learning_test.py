#!/usr/bin/env python3
"""Tests for reinforcement_learning.py."""

import sys
from collections import namedtuple
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_core import train_and_evaluate
from research.rl_environments import State, Action, Environment
from research.rl_environments import GridWorld, SimpleTMaze
from research.rl_environments import gating_memory, fixed_long_term_memory, memory_architecture
from research.rl_agents import TabularQLearningAgent
from research.rl_agents import epsilon_greedy
from research.randommixin import RandomMixin

RLTestStep = namedtuple('RLTestStep', ['observation', 'actions', 'action', 'reward'])


def test_gridworld():
    """Test the GridWorld environment."""
    env = GridWorld(
        width=2,
        height=3,
        start=[0, 0],
        goal=[2, 0],
    )
    env.start_new_episode()
    expected_steps = [
        RLTestStep(State(row=0, col=0), [Action('down'), Action('right')], Action('right'), -1),
        RLTestStep(State(row=0, col=1), [Action('down'), Action('left')], Action('right'), -1),
        RLTestStep(State(row=0, col=1), [Action('down'), Action('left')], Action('down'), -1),
        RLTestStep(State(row=1, col=1), [Action('up'), Action('down'), Action('left')], Action('down'), -1),
        RLTestStep(State(row=2, col=1), [Action('up'), Action('left')], Action('down'), -1),
        RLTestStep(State(row=2, col=1), [Action('up'), Action('left')], Action('up'), -1),
        RLTestStep(State(row=1, col=1), [Action('up'), Action('down'), Action('left')], Action('left'), -1),
        RLTestStep(State(row=1, col=0), [Action('up'), Action('down'), Action('right')], Action('down'), 1),
        RLTestStep(State(row=2, col=0), [], None, None),
    ]
    for expected in expected_steps:
        assert env.get_observation() == expected.observation
        assert set(env.get_actions()) == set(expected.actions)
        if expected.action is not None:
            reward = env.react(expected.action)
            assert reward == expected.reward


def test_simpletmaze():
    """Test the SimpleTMaze environment."""
    env = SimpleTMaze(2, 1, -1)
    env.start_new_episode()
    assert env.get_state() == State(x=0, y=0, symbol=0, goal_x=-1)
    expected_steps = [
        RLTestStep(
            State(x=0, y=0, symbol=0),
            [Action('up')],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=1, symbol=-1),
            [Action('up')],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=2, symbol=0),
            [Action('left'), Action('right')],
            Action('left'),
            10,
        ),
        RLTestStep(State(x=-1, y=2, symbol=0), [], None, None),
    ]
    for expected in expected_steps:
        assert env.get_observation() == expected.observation
        assert set(env.get_actions()) == set(expected.actions)
        if expected.action is not None:
            reward = env.react(expected.action)
            assert reward == expected.reward


def test_simpletmaze_gatingmemory():
    """Test the gating memory meta-environment."""
    env = gating_memory(SimpleTMaze)(
        num_memory_slots=1,
        reward=-0.05,
        length=2,
        hint_pos=1,
    )
    env.start_new_episode()
    goal = env.get_state().goal_x
    assert env.get_state() == State(x=0, y=0, symbol=0, goal_x=goal, memory_0=None)
    expected_steps = [
        RLTestStep(
            State(x=0, y=0, symbol=0, memory_0=None),
            [
                Action('up'),
                Action('gate', slot=0, attribute='x'),
                Action('gate', slot=0, attribute='y'),
                Action('gate', slot=0, attribute='symbol'),
            ],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=1, symbol=goal, memory_0=None),
            [
                Action('up'),
                Action('gate', slot=0, attribute='x'),
                Action('gate', slot=0, attribute='y'),
                Action('gate', slot=0, attribute='symbol'),
            ],
            Action('gate', slot=0, attribute='symbol'),
            -0.05,
        ),
        RLTestStep(
            State(x=0, y=1, symbol=goal, memory_0=goal),
            [
                Action('up'),
                Action('gate', slot=0, attribute='x'),
                Action('gate', slot=0, attribute='y'),
                Action('gate', slot=0, attribute='symbol'),
            ],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=2, symbol=0, memory_0=goal),
            [
                Action('left'),
                Action('right'),
                Action('gate', slot=0, attribute='x'),
                Action('gate', slot=0, attribute='y'),
                Action('gate', slot=0, attribute='symbol'),
            ],
            Action('right' if goal == -1 else 'left'),
            -10,
        ),
        RLTestStep(State(x=1 if goal == -1 else -1, y=2, symbol=0, memory_0=goal), [], None, None),
    ]
    for expected in expected_steps:
        assert env.get_observation() == expected.observation
        assert set(env.get_actions()) == set(expected.actions)
        if expected.action is not None:
            reward = env.react(expected.action)
            assert reward == expected.reward


def test_simpletmaze_fixedltm():
    """Test the fixed LTM meta-environment."""
    env = fixed_long_term_memory(SimpleTMaze)(
        num_wm_slots=1,
        num_ltm_slots=1,
        reward=-0.05,
        length=2,
        hint_pos=1,
        goal_x=1,
    )
    env.start_new_episode()
    assert env.get_state() == State(x=0, y=0, symbol=0, goal_x=1, wm_0=None, ltm_0=None)
    expected_steps = [
        RLTestStep(
            State(x=0, y=0, symbol=0, wm_0=None),
            [
                Action('up'),
                Action('store', slot=0, attribute='x'),
                Action('store', slot=0, attribute='y'),
                Action('store', slot=0, attribute='symbol'),
                Action('retrieve', wm_slot=0, ltm_slot=0),
            ],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=1, symbol=1, wm_0=None),
            [
                Action('up'),
                Action('store', slot=0, attribute='x'),
                Action('store', slot=0, attribute='y'),
                Action('store', slot=0, attribute='symbol'),
                Action('retrieve', wm_slot=0, ltm_slot=0),
            ],
            Action('store', slot=0, attribute='symbol'),
            -0.05,
        ),
        RLTestStep(
            State(x=0, y=1, symbol=1, wm_0=None),
            [
                Action('up'),
                Action('store', slot=0, attribute='x'),
                Action('store', slot=0, attribute='y'),
                Action('store', slot=0, attribute='symbol'),
                Action('retrieve', wm_slot=0, ltm_slot=0),
            ],
            Action('up'),
            -1,
        ),
        RLTestStep(
            State(x=0, y=2, symbol=0, wm_0=None),
            [
                Action('left'),
                Action('right'),
                Action('store', slot=0, attribute='x'),
                Action('store', slot=0, attribute='y'),
                Action('store', slot=0, attribute='symbol'),
                Action('retrieve', wm_slot=0, ltm_slot=0),
            ],
            Action('retrieve', wm_slot=0, ltm_slot=0),
            -0.05,
        ),
        RLTestStep(
            State(x=0, y=2, symbol=0, wm_0=1),
            [
                Action('left'),
                Action('right'),
                Action('store', slot=0, attribute='x'),
                Action('store', slot=0, attribute='y'),
                Action('store', slot=0, attribute='symbol'),
                Action('retrieve', wm_slot=0, ltm_slot=0),
            ],
            Action('right'),
            10,
        ),
        RLTestStep(State(x=1, y=2, symbol=0, wm_0=1), [], None, None),
    ]
    for expected in expected_steps:
        assert env.get_observation() == expected.observation
        assert set(env.get_actions()) == set(expected.actions)
        if expected.action is not None:
            reward = env.react(expected.action)
            assert reward == expected.reward


def test_agent():
    """Test the epsilon greedy tabular Q-learning agent."""
    env = GridWorld(
        width=5,
        height=5,
        start=[0, 0],
        goal=[4, 4],
    )
    agent = epsilon_greedy(TabularQLearningAgent)(
        exploration_rate=0.05,
        learning_rate=0.1,
        discount_rate=0.9,
        random_seed=8675309,
    )
    assert agent.random_seed == 8675309
    returns = list(train_and_evaluate(
        env,
        agent,
        num_episodes=500,
        eval_frequency=50,
        eval_num_episodes=50,
    ))
    for row in range(3):
        for col in range(3):
            best_action = agent.get_best_stored_action(State(row=row, col=col))
            assert best_action is None or best_action.name in ['down', 'right']
    agent.print_policy()
    # the optimal policy takes 8 steps for a 5x5 grid
    # -6 comes from 7 steps of -1 reward and 1 step of +1 reward
    assert returns[-1] == -6


def test_memory_architecture():
    """Test the memory architecture meta-environment."""

    class TestEnv(Environment, RandomMixin):
        """A simple environment with a single string state."""

        def __init__(self, string=''):
            """Initialize the TestEnv.

            Arguments:
                string (str): The initial string.
            """
            self.init_string = string
            self.string = self.init_string
            super().__init__()

        def get_state(self): # noqa: D102
            return self.get_observation()

        def get_observation(self): # noqa: D102
            return State(string=self.string)

        def get_actions(self): # noqa: D102
            if self.string == 'done':
                return []
            else:
                return [Action('no-op')]

        def reset(self): # noqa: D102
            self.start_new_episode()

        def start_new_episode(self): # noqa: D102
            self.string = self.init_string

        def react(self, action): # noqa: D102
            if action.name != 'no-op':
                self.string = action.name
            if self.string == 'done':
                return 100
            else:
                return -1

        def visualize(self): # noqa: D102
            pass

    env = memory_architecture(TestEnv)(explicit_actions=False)
    env.start_new_episode()
    size = 5
    for i in range(size * size):
        env.add_to_ltm(index=i, row=(i // size), col=(i % size))
    # test observation
    assert env.get_observation() == State(
        perceptual_string='',
        action_name='no-op',
    ), env.get_observation()
    # test actions
    assert (
        set(env.get_actions()) == set(
            [
                Action("copy", dst_attr='string', dst_buf='query', src_attr='string', src_buf='perceptual'),
                Action("copy", dst_attr='string', dst_buf='action', src_attr='string', src_buf='perceptual'),
                Action("copy", dst_attr='name', dst_buf='action', src_attr='string', src_buf='perceptual'),
            ]
        )
    ), set(env.get_actions())
    # test pass-through reaction
    reward = env.react(Action('write', buf='action', attr='name', val='test-action'))
    assert env.get_observation() == State(
        perceptual_string='test-action',
        action_name='no-op',
    ), env.get_observation()
    assert reward == -1, reward
    # query test
    env.react(Action('write', buf='query', attr='index', val=9))
    assert env.get_observation() == State(
        perceptual_string='test-action',
        action_name='no-op',
        query_index=9,
        retrieval_index=9,
        retrieval_row=1,
        retrieval_col=4,
    ), env.get_observation()
    # query with no results
    env.react(Action('write', buf='query', attr='row', val=4))
    assert env.get_observation() == State(
        perceptual_string='test-action',
        action_name='no-op',
        query_index=9,
        query_row=4,
    ), env.get_observation()
    # query and retrieval should clear on external action
    env.react(Action('write', buf='action', attr='name', val='clear-memory-buffer'))
    assert env.get_observation() == State(
        perceptual_string='clear-memory-buffer',
        action_name='no-op',
    ), env.get_observation()
    # complete the environment
    reward = env.react(Action('write', buf='action', attr='name', val='done'))
    assert env.end_of_episode()
    assert reward == 100, reward
