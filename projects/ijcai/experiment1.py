#!/usr/bin/env python3

import sys
from collections import namedtuple
from os.path import realpath, dirname
from uuid import uuid4 as uuid

DIRECTORY = dirname(realpath(__file__))
sys.path.insert(0, dirname(DIRECTORY))

# pylint: disable = wrong-import-position
from research.rl_agents import epsilon_greedy, LinearQLearner
from research.rl_environments import State, Action, Environment
from research.rl_memory import memory_architecture, NaiveDictKB
from research.randommixin import RandomMixin

Album = namedtuple('Album', 'title, artist, year, genre')

class RecordStore(Environment, RandomMixin):

    def __init__(self, num_albums=100, num_artists=20, num_years=2, num_genres=100, *args, **kwargs):
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
        if int(self.location) == self.album.year:
            return actions
        for year in range(self.num_years):
            actions.append(Action(str(year)))
        return actions

    def react(self, action):
        self.location = action.name
        if int(self.location) == self.album.year:
            return 0
        else:
            return -10

    def reset(self):
        for i in range(self.num_albums):
            title = str(i)
            artist = self.rng.randrange(self.num_artists)
            year = self.rng.randrange(self.num_years)
            genre = self.rng.randrange(self.num_genres)
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
