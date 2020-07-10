#!/usr/bin/env python3
"""Tests for RL memory code."""

from research import State, Action, Environment
from research import NaiveDictLTM
from research import TabularQLearningAgent, epsilon_greedy
from research import MemoryArchitectureMetaEnvironment
from research import train_and_evaluate


def test_memory_architecture_unit():
    """Do unit tests on the memory architecture meta-environment."""

    class TestEnv(Environment):
        """A simple environment with a single string state."""

        def __init__(self, size, index=0):
            """Initialize the TestEnv.

            Arguments:
                size (int): The length of one side of the square.
                index (int): The initial int.
            """
            super().__init__()
            self.size = size
            self.init_index = index
            self.index = self.init_index

        def get_state(self): # noqa: D102
            return State(index=self.index)

        def get_observation(self): # noqa: D102
            return State(index=self.index)

        def get_actions(self): # noqa: D102
            if self.index == -1:
                return []
            else:
                return [Action(str(i)) for i in range(-1, size * size)]

        def reset(self): # noqa: D102
            self.start_new_episode()

        def start_new_episode(self): # noqa: D102
            self.index = self.init_index

        def react(self, action): # noqa: D102
            assert action in self.get_actions()
            if action.name != 'no-op':
                self.index = int(action.name)
            if self.end_of_episode():
                return 100
            else:
                return -1

        def visualize(self): # noqa: D102
            pass

    size = 5
    env = MemoryArchitectureMetaEnvironment(
        env=TestEnv(
            size=size,
            index=0,
        ),
        ltm=NaiveDictLTM(),
    )
    env.start_new_episode()
    for i in range(size * size):
        env.add_to_ltm(index=i, row=(i // size), col=(i % size))
    # test observation
    assert env.get_observation() == State(
        perceptual_index=0,
    ), env.get_observation()
    # test actions
    assert (
        set(env.get_actions()) == set([
            *(Action(str(i)) for i in range(-1, size * size)),
            Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=0),
        ])
    ), set(env.get_actions())
    # test pass-through reaction
    reward = env.react(Action('9'))
    assert env.get_observation() == State(
        perceptual_index=9,
    ), env.get_observation()
    assert reward == -1, reward
    # query test
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=9))
    assert env.get_observation() == State(
        perceptual_index=9,
        query_index=9,
        retrieval_index=9,
        retrieval_row=1,
        retrieval_col=4,
    ), env.get_observation()
    # query with no results
    env.react(Action('copy', src_buf='retrieval', src_attr='row', dst_buf='query', dst_attr='row', dst_val=1))
    env.react(Action('0'))
    env.react(Action('copy', src_buf='perceptual', src_attr='index', dst_buf='query', dst_attr='index', dst_val=0))
    env.react(Action('delete', buf='query', attr='index', val=9))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_index=0,
        query_row=1,
    ), env.get_observation()
    # delete and query test
    env.react(Action('delete', buf='query', attr='index', val=0))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # next result test
    env.react(Action('next-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=6,
        retrieval_row=1,
        retrieval_col=1,
    ), env.get_observation()
    # delete test
    env.react(Action('prev-result'))
    assert env.get_observation() == State(
        perceptual_index=0,
        query_row=1,
        retrieval_index=5,
        retrieval_row=1,
        retrieval_col=0,
    ), env.get_observation()
    # complete the environment
    reward = env.react(Action('-1'))
    assert env.end_of_episode()
    assert reward == 100, reward


def test_memory_architecture_integration():
    """Do integration tests on the memory architecture meta-environment."""

    def reset_memory(mem_env, _):
        mem_env.ltm.clear()
        mem_env.ltm.store(y=mem_env.env.length, goal=mem_env.env.goal_x)

    tmaze = SimpleTMaze(100, hint_pos=-1, random_seed=8675309)
    mem_env = MemoryArchitectureMetaEnvironment(
        tmaze,
        ltm=NaiveDictLTM(),
        internal_reward=-.1,
        max_internal_actions=1,
        buf_ignore=['scratch'],
    )
    agent = epsilon_greedy(TabularQLearningAgent)(
        exploration_rate=0.05,
        learning_rate=0.1,
        discount_rate=0.9,
        random_seed=8675309,
    )
    returns = train_and_evaluate(
        mem_env,
        agent,
        num_episodes=20000,
        eval_frequency=50,
        eval_num_episodes=5,
        min_return=-200,
        new_episode_hook=reset_memory,
    )
    print()
    for i, total_reward in enumerate(returns, start=1):
        print(i, total_reward)
    print()
    agent.print_value_function()
