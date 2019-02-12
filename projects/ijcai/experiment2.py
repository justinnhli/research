#!/usr/bin/env python3

import re
import sys
from collections import namedtuple
from datetime import datetime
from itertools import chain
from math import isnan
from pathlib import Path
from textwrap import dedent

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

Schema = namedtuple('Schema', 'name sparql clues categories')


TITLE_YEAR = Schema(
    'title_year',
    dedent('''
        SELECT DISTINCT ?title ?release_date WHERE {
            ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
            ?album <http://xmlns.com/foaf/0.1/name> ?title ;
		   <http://wikidata.dbpedia.org/ontology/artist> ?artist_node ;
                   <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
            FILTER ( lang(?title) = "en" )
        }
    ''').strip(),
    ['title',],
    ['release_date',],
)

TITLE_GENRE_DECADE = Schema(
    'title_genre_decade',
    dedent('''
        SELECT DISTINCT ?title ?genre ?release_date WHERE {
            ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
            ?album <http://xmlns.com/foaf/0.1/name> ?title ;
		   <http://wikidata.dbpedia.org/ontology/artist> ?artist_node ;
                   <http://wikidata.dbpedia.org/ontology/genre> ?genre_node ;
                   <http://wikidata.dbpedia.org/ontology/releaseDate> ?release_date .
            ?genre_node <http://xmlns.com/foaf/0.1/name> ?genre .
            FILTER ( lang(?title) = "en" )
            FILTER ( lang(?genre) = "en" )
        }
    ''').strip(),
    ['title',],
    ['genre', 'release_date',],
)

TITLE_COUNTRY = Schema(
    'title_country',
    dedent('''
        SELECT DISTINCT ?title ?country WHERE {
            ?track <http://wikidata.dbpedia.org/ontology/album> ?album .
            ?album <http://xmlns.com/foaf/0.1/name> ?title ;
                        <http://wikidata.dbpedia.org/ontology/artist> ?artist .
            ?artist <http://wikidata.dbpedia.org/ontology/hometown> ?hometown .
            ?hometown <http://wikidata.dbpedia.org/ontology/country> ?country_node .
            ?country_node <http://xmlns.com/foaf/0.1/name> ?country .
            FILTER ( lang(?title) = "en" )
            FILTER ( lang(?country) = "en" )
        }
    ''').strip(),
    ['title',],
    ['country',],
)


def get_schema_attr(schema, var):
    matches = [
        re.search(r'(?P<attr><[^>]*>) \?' + var + ' [;.]', line)
        for line in schema.sparql.splitlines()
    ]
    matches = [match for match in matches if match is not None]
    assert len(matches) == 1
    return matches[0].group('attr')


class RecordStore(Environment, RandomMixin):

    def __init__(self, schema=None, num_albums=1000, *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        assert schema is not None
        self.schema = schema
        self.num_albums = num_albums
        # database
        self.uris = [
            get_schema_attr(self.schema, var) for var in
            chain(self.schema.clues, self.schema.categories)
        ]
        self.questions = []
        self.answers = {}
        self.actions = set()
        # variables
        self.question = None
        self.location = None
        self.reset()

    def get_state(self):
        return self.get_observation()

    def get_observation(self):
        return State.from_dict({uri: val for uri, val in zip(self.uris, self.question)})

    def get_actions(self):
        if self.location == self.answers[self.question]:
            return []
        actions = []
        for action_str in self.actions:
            assert isinstance(action_str, str), action_str
            actions.append(Action(action_str))
        return actions

    def react(self, action):
        self.location = action.name
        if self.location == self.answers[self.question]:
            return 0
        else:
            return -10

    def reset(self):
        with Path(__file__).parent.joinpath('schemas', self.schema.name).open() as fd:
            for line, _ in zip(fd, range(self.num_albums)):
                vals = []
                for uri, val in zip(self.uris, line.strip().split('\t')):
                    if uri == '<http://wikidata.dbpedia.org/ontology/releaseDate>':
                        vals.append(date_to_year(val))
                    elif uri == '<http://xmlns.com/foaf/0.1/name>':
                        vals.append(date_to_year(val))
                    else:
                        vals.append(val)
                question = tuple(vals[:len(self.schema.clues)])
                answer = tuple(vals[-len(self.schema.categories):])
                self.answers[question] = str(answer)
                self.actions.add(str(answer))
        self.questions = sorted(self.answers.keys())

    def start_new_episode(self):
        self.question = self.rng.choice(self.questions)
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
        schema=TITLE_YEAR,
        num_albums=3,
        # memory architecture
        max_internal_actions=5,
        knowledge_store=SparqlKB(
            SparqlEndpoint('http://162.233.132.179:8890/sparql'),
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
