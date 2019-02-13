#!/usr/bin/env python3

import re
import sys
from datetime import datetime
from itertools import chain
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

from schemas import SCHEMAS
from download_schema import download_schema_data


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
        schema_data_path = Path(__file__).parent.joinpath('schemas', self.schema.name)
        if not schema_data_path.exists():
            schema_data_path.parent.mkdir(parents=True, exist_ok=True)
            download_schema_data(self.schema)
        albums = []
        with Path(__file__).parent.joinpath('schemas', self.schema.name).open() as fd:
            for line in fd:
                vals = []
                for uri, val in zip(self.uris, line.strip().split('\t')):
                    if uri == '<http://wikidata.dbpedia.org/ontology/releaseDate>':
                        if self.schema.name == 'title_year':
                            vals.append(date_to_year(val))
                        elif self.schema.name == 'title_genre_decade':
                            vals.append(date_to_decade(val))
                    elif uri == '<http://xmlns.com/foaf/0.1/name>':
                        vals.append(date_to_year(val))
                    else:
                        vals.append(val)
                albums.append(vals)
        for vals in self.rng.sample(albums, self.num_albums):
            question = tuple(vals[:len(self.schema.clues)])
            answer = tuple(vals[-len(self.schema.categories):])
            self.answers[question] = str(answer)
            self.actions.add(str(answer))
        self.questions = sorted(self.answers.keys())

    def start_new_episode(self):
        self.question = self.rng.choice(self.questions)
        self.location = '__start__'

    def visualize(self):
        raise NotImplementedError()


def date_to_year(date):
    return re.sub('^"([0-9]{4}).*"([@^][^"]*)$', r'"\1-01-01"\2', date)


def date_to_decade(date):
    return re.sub('^"([0-9]{3}).*"([@^][^"]*)$', r'"\g<1>0-01-01"\2', date)


INTERNAL_ACTIONS = set([
    'copy',
    'delete',
    'retrieve',
    'next-retrieval',
    'prev-retrieval',
])


def feature_extractor(state, action=None):
    features = set()
    features.add('_bias')
    internal = action is None or action.name in INTERNAL_ACTIONS
    external = action is None or action.name not in INTERNAL_ACTIONS
    for attribute, value in state.as_dict().items():
        if internal:
            features.add(attribute)
        if external:
            features.add((attribute, value))
    return features

AUGMENTS = [
    SparqlKB.Augment(
        '<http://xmlns.com/foaf/0.1/name>',
        '<http://xmlns.com/foaf/0.1/firstLetter>',
        (lambda name: re.sub('"(.).*"([@^][^"]*)', r'"\1"\2', name)),
    ),
    SparqlKB.Augment(
        '<http://wikidata.dbpedia.org/ontology/releaseDate>',
        '<http://wikidata.dbpedia.org/ontology/releaseYear>',
        date_to_year,
    ),
    SparqlKB.Augment(
        '<http://wikidata.dbpedia.org/ontology/releaseDate>',
        '<http://wikidata.dbpedia.org/ontology/releaseDecade>',
        date_to_decade,
    ),
]

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
        schema=SCHEMAS['title_year'],
        num_albums=3,
        # memory architecture
        max_internal_actions=5,
        knowledge_store=SparqlKB(
            SparqlEndpoint('http://162.233.132.179:8890/sparql'),
            augments=AUGMENTS,
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
        schema=params.schema,
        num_albums=params.num_albums,
        # memory architecture
        max_internal_actions=params.max_internal_actions,
        knowledge_store=SparqlKB(
            SparqlEndpoint('http://162.233.132.179:8890/sparql'),
            augments=AUGMENTS,
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
    filename = '-'.join([
        f'schema{params.schema_name}',
        f'seed{params.random_seed}',
        f'albums{params.num_albums}',
    ])
    episodes = range(0, params.num_episodes, params.eval_frequency)
    results_path = Path(DIRECTORY, 'data', 'exp2')
    results_path.mkdir(parents=True, exist_ok=True)
    for episode, mean_return in zip(episodes, trial_result):
        with results_path.joinpath(filename).open('a') as fd:
            fd.write(f'{datetime.now().isoformat("_")} {episode} {mean_return}\n')
        if (episode + params.eval_frequency) % 1000 == 0:
            with results_path.joinpath(filename).open('a') as fd:
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
                    action = agent.get_best_stored_action(env.get_observation(), actions=env.get_actions())
                    fd.write(f'{action}\n')
                    env.react(action)
                    fd.write('\n')
                fd.write(30 * '-' + '\n')


def main():
    pspace = PermutationSpace(
        ['schema_name', 'random_seed', 'num_albums'],
        random_seed=[
            0.35746869278354254, 0.7368915891545381, 0.03439267552305503, 0.21913569678035283, 0.0664623502695384,
            #0.53305059438797, 0.7405341747379695, 0.29303361447547216, 0.014835598224628765, 0.5731489218909421,
        ],
        num_episodes=10000,
        eval_frequency=100,
        num_albums=[1000,],
        max_internal_actions=5,
        schema_name=['title_year', 'title_genre_decade', 'title_country',],
        schema=(lambda schema_name: SCHEMAS[schema_name]),
    )
    size = len(pspace)
    for i, params in enumerate(pspace, start=1):
        print(f'{datetime.now().isoformat()} {i}/{size} running {params}')
        run_experiment(params)


if __name__ == '__main__':
    testing()
