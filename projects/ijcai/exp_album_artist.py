#!/usr/bin/env python3

import sys
from datetime import datetime
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))

# pylint: disable = wrong-import-position
from permspace import PermutationSpace

from research.knowledge_base import SparqlEndpoint
from research.pspace_run import parallel_main
from research.rl_core import train_and_evaluate
from research.rl_agents import epsilon_greedy, LinearQLearner
from research.rl_memory import memory_architecture, SparqlKB

from record_store import RecordStore, feature_extractor
from record_store import NAME_FIRST_LETTER


def testing():
    agent = epsilon_greedy(LinearQLearner)(
        # Linear Q Learner
        learning_rate=0.1,
        discount_rate=0.9,
        feature_extractor=feature_extractor,
        # Epsilon Greedy
        exploration_rate=0.05,
        # Random Mixin
        random_seed=8675309,
    )
    env = memory_architecture(RecordStore)(
        # record store
        data_file='data/album_artist',
        num_albums=1000,
        # memory architecture
        max_internal_actions=5,
        knowledge_store=SparqlKB(
            SparqlEndpoint('http://162.233.132.179:8890/sparql'),
            augments=[NAME_FIRST_LETTER],
        ),
        # Random Mixin
        random_seed=8675309,
    )
    for trial in range(1000):
        env.start_new_episode()
        step = 0
        total = 0
        while not env.end_of_episode():
            observation = env.get_observation()
            actions = env.get_actions()
            action = agent.act(observation, actions)
            reward = env.react(action)
            agent.observe_reward(observation, reward, actions=env.get_actions())
            step += 1
            total += reward
            if total < -100:
                break
        print(trial, total)
    env.start_new_episode()
    visited = set()
    for step in range(10):
        print(step)
        observation = env.get_observation()
        print(observation)
        if observation in visited:
            print('\n')
            print('Looped; quitting.\n')
            break
        elif env.end_of_episode():
            break
        print(feature_extractor(observation))
        actions = env.get_actions()
        for action in sorted(actions):
            print(action)
            print('    ', agent.get_value(env.get_observation(), action))
        action = agent.get_best_stored_action(env.get_observation(), actions=actions)
        print(action)
        env.react(action)
        print()


def run_experiment(params):
    agent = epsilon_greedy(LinearQLearner)(
        # Linear Q Learner
        learning_rate=0.1,
        discount_rate=0.9,
        feature_extractor=feature_extractor,
        # Epsilon Greedy
        exploration_rate=0.05,
        # Random Mixin
        random_seed=params.random_seed,
    )
    env = memory_architecture(RecordStore)(
        # record store
        data_file=params.data_file,
        num_albums=params.num_albums,
        # memory architecture
        max_internal_actions=params.max_internal_actions,
        knowledge_store=SparqlKB(
            SparqlEndpoint('http://162.233.132.179:8890/sparql'),
            augments=[NAME_FIRST_LETTER],
        ),
        # Random Mixin
        random_seed=params.random_seed,
    )
    trial_result = train_and_evaluate(
        env,
        agent,
        num_episodes=params.num_episodes,
        eval_frequency=params.eval_frequency,
        min_return=-100,
    )
    episodes = range(0, params.num_episodes, params.eval_frequency)
    filename = '_'.join([
        f'seed{params.random_seed}',
        f'num{params.num_albums}',
        f'max{params.max_internal_actions}',
    ]) + '.csv'
    data_file = Path(DIRECTORY, 'results', Path(params.data_file).name, filename)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    for episode, mean_return in zip(episodes, trial_result):
        with data_file.open('a') as fd:
            fd.write(f'{datetime.now().isoformat("_")} {episode} {mean_return}\n')
    weights_file = data_file.parent.joinpath(filename + '.weights')
    with weights_file.open('w') as fd:
        for action, weights in agent.weights.items():
            fd.write(str(action))
            fd.write('\n')
            for feature, weight in weights.items():
                fd.write(f'    {feature} {weight}')
                fd.write('\n')


PSPACE = PermutationSpace(
    ['random_seed', 'num_albums', 'max_internal_actions'],
    random_seed=[
        0.35746869278354254, 0.7368915891545381, 0.03439267552305503, 0.21913569678035283, 0.0664623502695384,
        0.53305059438797, 0.7405341747379695, 0.29303361447547216, 0.014835598224628765, 0.5731489218909421,
    ],
    num_episodes=10000,
    eval_frequency=100,
    num_albums=range(100, 1050, 100),
    max_internal_actions=range(1, 6),
    data_file='data/album_artist'
)


def main():
    curr_file = Path(__file__).resolve()
    parallel_main(str(curr_file), f'{curr_file.stem}.PSPACE', f'{curr_file.stem}.run_experiment')


if __name__ == '__main__':
    main()
