#!/usr/bin/env python3

import sys
from collections import namedtuple
from datetime import datetime
from math import isnan
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))

# pylint: disable = wrong-import-position
from permspace import PermutationSpace

from research.rl_core import train_and_evaluate
from research.rl_agents import epsilon_greedy, LinearQLearner
from research.rl_environments import State, Action, Environment
from research.rl_memory import memory_architecture, NaiveDictKB
from research.randommixin import RandomMixin

Album = namedtuple('Album', 'title, artist, release_date, genre')


class RecordStore(Environment, RandomMixin):

    def __init__(self, num_albums=100, num_artists=20, num_years=2, num_genres=100, *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        self.num_albums = num_albums
        self.num_artists = num_artists
        self.num_years = num_years
        self.num_genres = num_genres
        # variables
        self.albums = {}
        self.titles = []
        self.album = None
        self.location = None
        self.reset()

    def get_state(self):
        return self.get_observation()

    def get_observation(self):
        return State(title=self.album.title)

    def get_actions(self):
        actions = []
        if int(self.location) == self.album.release_date:
            return actions
        for release_date in range(self.num_years):
            actions.append(Action(str(release_date)))
        return actions

    def react(self, action):
        self.location = action.name
        if int(self.location) == self.album.release_date:
            return 0
        else:
            return -10

    def reset(self):
        for i in range(self.num_albums):
            title = str(i)
            artist = self.rng.randrange(self.num_artists)
            release_date = self.rng.randrange(self.num_years)
            genre = self.rng.randrange(self.num_genres)
            self.albums[title] = Album(title, artist, release_date, genre)
            self.titles.append(title)
        self.titles = sorted(self.titles)

    def start_new_episode(self):
        self.album = self.albums[self.rng.choice(self.titles)]
        self.location = '-1'

    def visualize(self):
        raise NotImplementedError()


def feature_extractor(state):
    features = set()
    features.add('_bias')
    for attribute, value in state.as_dict().items():
        features.add((attribute, value))
        features.add(attribute)
    return features


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
        num_albums=100,
        num_artists=100 // 3,
        num_years=2,
        num_genres=10,
        # memory architecture
        knowledge_store=NaiveDictKB(),
        # Random Mixin
        random_seed=8675309,
    )
    for album in env.albums.values():
        env.add_to_ltm(**album._asdict())
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
        num_albums=params.num_albums,
        num_artists=params.num_artists,
        num_years=params.num_years,
        num_genres=params.num_genres,
        # memory architecture
        max_internal_actions=params.max_internal_actions,
        knowledge_store=NaiveDictKB(),
        # Random Mixin
        random_seed=params.random_seed,
    )
    for album in env.albums.values():
        env.add_to_ltm(**album._asdict())
    trial_result = train_and_evaluate(
        env,
        agent,
        num_episodes=params.num_episodes,
        eval_frequency=params.eval_frequency,
        min_return=-100,
    )
    filename = '-'.join([
        f'seed{params.random_seed}',
        f'ratio{params.albums_per_artist}',
        f'albums{params.num_albums}',
        f'years{params.num_years}',
        f'genres{params.num_genres}',
    ])
    episodes = range(0, params.num_episodes, params.eval_frequency)
    results_path = Path(DIRECTORY, 'data', 'exp1')
    results_path.mkdir(parents=True, exist_ok=True)
    for episode, mean_return in zip(episodes, trial_result):
        with results_path.joinpath(filename).open('a') as fd:
            fd.write(f'{datetime.now().isoformat("_")} {episode} {mean_return}\n')
        if (episode + params.eval_frequency) % 1000 == 0:
            has_nan = False
            with open(f'{DIRECTORY}/data/exp1/{filename}', 'a') as fd:
                fd.write(30 * '-' + '\n')
                visited = set()
                env.start_new_episode()
                for step in range(10):
                    fd.write(f'{step}\n')
                    observation = env.get_observation()
                    fd.write(f'{observation}\n')
                    if observation in visited:
                        fd.write('\n')
                        fd.write('Looped; quitting.\n')
                        break
                    elif env.end_of_episode():
                        break
                    visited.add(observation)
                    fd.write(f'{feature_extractor(observation)}\n')
                    actions = env.get_actions()
                    for action in sorted(actions):
                        fd.write(f'{action}\n')
                        fd.write(f'    {agent.get_value(env.get_observation(), action)}\n')
                        for feature, weight in agent.weights[action].items():
                            if isnan(weight):
                                has_nan = True
                            fd.write(f'        {feature}: {weight}\n')
                    action = agent.get_best_stored_action(env.get_observation(), actions=env.get_actions())
                    fd.write(f'{action}\n')
                    env.react(action)
                    fd.write('\n')
                fd.write(30 * '-' + '\n')
            if has_nan:
                return


def main():
    pspace = PermutationSpace(
        ['num_albums', 'albums_per_artist', 'num_genres', 'num_years', 'random_seed'],
        random_seed=[
            0.35746869278354254, 0.7368915891545381, 0.03439267552305503, 0.21913569678035283, 0.0664623502695384,
            #0.53305059438797, 0.7405341747379695, 0.29303361447547216, 0.014835598224628765, 0.5731489218909421,
        ],
        num_episodes=10000,
        eval_frequency=100,
        num_albums=[100, 1000],
        albums_per_artist=[3, 5, 10],
        num_artists=(lambda num_albums, albums_per_artist: num_albums // albums_per_artist),
        num_years=[2, 10, 30],
        num_genres=[10, 30],
        max_internal_actions=5,
    )
    size = len(pspace)
    for i, params in enumerate(pspace, start=1):
        print(f'{datetime.now().isoformat()} {i}/{size} running {params}')
        run_experiment(params)


if __name__ == '__main__':
    main()
