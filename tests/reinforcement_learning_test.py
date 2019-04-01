#!/usr/bin/env python3
"""Tests for basic reinforcement learning code."""

import sys
from collections import namedtuple
from math import copysign
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_core import train_and_evaluate
from research.rl_environments import State, Action, Environment, RandomMixin
from research.rl_environments import GridWorld, SimpleTMaze
from research.rl_environments import gating_memory, fixed_long_term_memory
from research.rl_agents import TabularQLearningAgent, LinearQLearner
from research.rl_agents import epsilon_greedy

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
        RLTestStep(State(row=0, col=1), [Action('down'), Action('left')], Action('down'), -1),
        RLTestStep(State(row=1, col=1), [Action('up'), Action('down'), Action('left')], Action('down'), -1),
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
    # the optimal policy takes 8 steps for a 5x5 grid
    # -6 comes from 7 steps of -1 reward and 1 step of +1 reward
    assert returns[-1] == -6


def test_linear_agent():
    """Test the linear approximation Q-learning agent."""

    class InfiniteGridWorld(Environment, RandomMixin):
        """An infinite gridworld. Goal is (0, 0)."""

        def __init__(self, max_size, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_size = max_size
            self.row = 0
            self.col = 0

        def get_state(self): # noqa: D102
            return State(row=self.row, col=self.col)

        def get_actions(self): # noqa: D102
            if self.row == self.col == 0:
                return []
            else:
                return [
                    Action('up'),
                    Action('down'),
                    Action('left'),
                    Action('right'),
                    Action('upleft'),
                    Action('upright'),
                    Action('downleft'),
                    Action('downright'),
                ]

        def reset(self): # noqa: D102
            self.start_new_episode()

        def start_new_episode(self): # noqa: D102
            while self.row == self.col == 0:
                self.row = self.rng.randrange(-self.max_size, self.max_size + 1)
                self.col = self.rng.randrange(-self.max_size, self.max_size + 1)

        def react(self, action=None): # noqa: D102
            assert action in self.get_actions()
            if 'up' in action.name:
                self.row -= 1
            if 'down' in action.name:
                self.row += 1
            if 'left' in action.name:
                self.col -= 1
            if 'right' in action.name:
                self.col += 1
            if self.row == self.col == 0:
                return 1
            else:
                return 0

        def visualize(self): # noqa: D102
            raise NotImplementedError

    def feature_extractor(state, action=None): # pylint: disable = unused-argument
        return {
            'row': (0 if state['row'] == 0 else copysign(1, state['row'])),
            'col': (0 if state['col'] == 0 else copysign(1, state['col'])),
        }

    size = 1000
    env = InfiniteGridWorld(max_size=size)
    agent = LinearQLearner(
        learning_rate=0.1,
        discount_rate=0.9,
        feature_extractor=feature_extractor,
    )
    # train the agent
    for _ in range(50):
        env.start_new_episode()
        while not env.end_of_episode():
            observation = env.get_observation()
            name = ''
            if observation['row'] < 0:
                name += 'down'
            elif observation['row'] > 0:
                name += 'up'
            if observation['col'] < 0:
                name += 'right'
            elif observation['col'] > 0:
                name += 'left'
            action = Action(name)
            action = agent.force_act(observation, action)
            reward = env.react(action)
            agent.observe_reward(env.get_observation(), reward)
    # test that the agent can finish within `2 * size` steps
    for _ in range(50):
        env.start_new_episode()
        step = 2 * size
        while step > 0 and not env.end_of_episode():
            observation = env.get_observation()
            action = agent.act(observation, env.get_actions())
            reward = env.react(action)
            step -= 1
        assert env.end_of_episode()
