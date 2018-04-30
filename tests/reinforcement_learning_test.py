#!/usr/bin/env python3
"""Tests for reinforcement_learning.py."""

import sys
from collections import namedtuple
from os.path import dirname, realpath

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable=wrong-import-position
from research.reinforcement_learning import State, Action
from research.reinforcement_learning import GridWorld

RLTestStep = namedtuple('RLTestStep', ['action', 'observation', 'actions'])


def test_gridworld():
    """Test the GridWorld environment."""
    env = GridWorld(
        width=2,
        height=3,
        start=[0, 0],
        goal=[2, 0],
    )
    env.new_episode()
    assert env.get_observation() == State(row=0, col=0)
    assert set(env.get_actions()) == set([Action('down'), Action('right')])
    expected_steps = [
        RLTestStep(Action('right'), State(row=0, col=1), [Action('down'), Action('left')]),
        RLTestStep(Action('right'), State(row=0, col=1), [Action('down'), Action('left')]),
        RLTestStep(Action('down'), State(row=1, col=1), [Action('up'), Action('down'), Action('left')]),
        RLTestStep(Action('down'), State(row=2, col=1), [Action('up'), Action('left')]),
        RLTestStep(Action('down'), State(row=2, col=1), [Action('up'), Action('left')]),
        RLTestStep(Action('up'), State(row=1, col=1), [Action('up'), Action('down'), Action('left')]),
        RLTestStep(Action('left'), State(row=1, col=0), [Action('up'), Action('down'), Action('right')]),
        RLTestStep(Action('down'), None, []),
    ]
    for expected in expected_steps:
        reward = env.react(expected.action)
        assert reward == -1
        assert env.get_observation() == expected.observation
        assert set(env.get_actions()) == set(expected.actions)
    reward = env.react(None)
    assert reward == 1
