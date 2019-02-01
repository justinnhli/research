#!/usr/bin/env python3

import sys
from os.path import realpath, dirname
from collections import namedtuple
from uuid import uuid4 as uuid

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_agents import epsilon_greedy, LinearQLearner
from research.rl_environments import State, Action, Environment, memory_architecture
from research.randommixin import RandomMixin

Album = namedtuple('Album', 'title, artist, year, genre')

class RecordStore(Environment, RandomMixin):
    NUM_ALBUMS = 100
    NUM_YEARS = 2
    NUM_GENRES = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.albums = {}
        self.titles = []
        self.reset()
        self.album = None
        self.location = None

    def get_state(self):
        return self.get_observation()

    def get_observation(self):
        return State(title=self.album.title)

    def get_actions(self):
        actions = []
        if int(self.location) == self.album.year:
            return actions
        for year in range(self.NUM_YEARS):
            actions.append(Action(str(year)))
        return actions

    def react(self, action):
        self.location = action.name
        if int(self.location) == self.album.year:
            return 0
        else:
            return -10

    def reset(self):
        for i in range(self.NUM_ALBUMS):
            title = str(i)
            artist = str(i)
            #artist = '-' # FIXME
            year = self.rng.randrange(self.NUM_YEARS)
            genre = self.rng.randrange(self.NUM_GENRES)
            #genre = '-' # FIXME
            self.albums[title] = Album(title, artist, year, genre)
            self.titles.append(title)
        self.titles = sorted(self.titles)

    def start_new_episode(self):
        self.album = self.albums[self.rng.choice(self.titles)]
        self.location = '-1'


def feature_extractor(state):
    features = set()
    features.add('_bias')
    for attribute in state:
        if attribute.startswith('scratch'):
            features.add((attribute, state[attribute]))
        else:
            features.add(attribute)
    return features


def main():
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
            agent.observe_reward(observation, reward, actions=actions)
            step += 1
            total += reward
            if total < -100:
                break
        print(trial, total)
    env.start_new_episode()
    for step in range(10):
        print(step)
        print(env.get_observation())
        print(feature_extractor(env.get_observation()))
        for action in sorted(env.get_actions()):
            print(action)
            print('    ', agent.get_value(env.get_observation(), action))
        action = agent.get_best_stored_action(env.get_observation(), env.get_actions())
        print(action)
        env.react(action)
        print()


if __name__ == '__main__':
    main()
