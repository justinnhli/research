#!/usr/bin/env python3

import re
import sys
from collections import namedtuple
from datetime import datetime
from math import isnan
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))

# pylint: disable = wrong-import-position
from permspace import PermutationSpace

from research.knowledge_base import SparqlEndpoint
from research.rl_core import train_and_evaluate
from research.rl_agents import epsilon_greedy, LinearQLearner
from research.rl_environments import State, Action, Environment
from research.rl_memory import memory_architecture, SparqlKB
from research.randommixin import RandomMixin

Album = namedtuple('Album', 'title, release_year')


class RecordStore(Environment, RandomMixin):

    def __init__(self, num_albums=1000, *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        self.num_albums = num_albums
        # variables
        self.albums = {}
        self.titles = []
        self.album = None
        self.release_years = set()
        self.location = None
        self.reset()

    def get_state(self):
        return self.get_observation()

    def get_observation(self):
        return State.from_dict({
            '<http://www.w3.org/2000/01/rdf-schema#label>': self.album.title,
        })

    def get_actions(self):
        if self.location == self.album.release_year:
            return []
        actions = []
        for release_year in self.release_years:
            actions.append(Action(release_year))
        return actions

    def react(self, action):
        self.location = action.name
        if self.location == self.album.release_year:
            return 0
        else:
            return -10

    def reset(self):
        select_statement = f'''
            SELECT DISTINCT ?title ?release_date WHERE {{
                ?album <http://wikidata.dbpedia.org/ontology/type> <http://wikidata.dbpedia.org/resource/Q1242743> ;
                       <http://xmlns.com/foaf/0.1/name> ?title ;
                       <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
                FILTER ( lang(?title) = "en" )
            }} LIMIT {self.num_albums}
        '''
        endpoint = SparqlEndpoint('http://162.233.132.179:8890/sparql')
        self.albums = {}
        for result in endpoint.query_sparql(select_statement):
            title = result['title'].rdf_format
            release_date = result['release_date'].rdf_format
            release_year = date_to_year(release_date)
            self.albums[title] = Album(title, release_year)
            self.release_years.add(release_year)
        self.titles = sorted(self.albums.keys())

    def start_new_episode(self):
        self.album = self.albums[self.rng.choice(self.titles)]
        self.location = 'start'

    def visualize(self):
        raise NotImplementedError()


def date_to_year(date):
    return re.sub('^"([0-9]{4}).*"([@^][^"]*)$', r'"\1-01-01"\2', date)

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
        num_albums=3,
        # memory architecture
        knowledge_store=SparqlKB(
            SparqlEndpoint('https://dbpedia.org/sparql'),
            augments=[
                SparqlKB.Augment(
                    '<http://xmlns.com/foaf/0.1/name>',
                    '<http://xmlns.com/foaf/0.1/FirstLetter>',
                    (lambda name: re.sub('"(.).*"([@^][^"]*)', r'"\1"\2', name)),
                ),
                SparqlKB.Augment(
                    '<http://wikidata.dbpedia.org/ontology/releaseDate>',
                    '<http://wikidata.dbpedia.org/ontology/releaseYear>',
                    date_to_year,
                ),
            ],
        ),
        # Random Mixin
        random_seed=8675309,
    )
    for trial in range(1000):
        env.start_new_episode()
        step = 0
        total = 0
        while not env.end_of_episode():
            print(step)
            observation = env.get_observation()
            print('   ', observation)
            actions = env.get_actions()
            action = agent.act(observation, actions)
            print('   ', action)
            reward = env.react(action)
            print('   ', reward)
            agent.observe_reward(observation, reward, actions=env.get_actions())
            step += 1
            print()
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


if __name__ == '__main__':
    testing()
