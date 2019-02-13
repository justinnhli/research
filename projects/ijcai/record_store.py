import re
from itertools import chain
from pathlib import Path

from research.rl_environments import State, Action, Environment
from research.rl_memory import SparqlKB
from research.randommixin import RandomMixin

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


def first_letter(literal):
    if re.fullmatch('"[^a-z]*([a-z]).*"([@^][^"]*)', literal, flags=re.IGNORECASE):
        return re.sub('"[^a-z]*([a-z]).*"([@^][^"]*)', r'"\1"\2', literal)
    else:
        return None


def date_to_year(date):
    if re.fullmatch('"([0-9]{4}).*"([@^][^"]*)', date):
        return re.sub('^"([0-9]{4}).*"([@^][^"]*)$', r'"\1-01-01"\2', date)
    else:
        return None


def date_to_decade(date):
    if re.fullmatch('"([0-9]{3}).*"([@^][^"]*)', date):
        return re.sub('^"([0-9]{3}).*"([@^][^"]*)$', r'"\g<1>0-01-01"\2', date)
    else:
        return None


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

NAME_FIRST_LETTER = SparqlKB.Augment(
    '<http://xmlns.com/foaf/0.1/name>',
    '<http://xmlns.com/foaf/0.1/firstLetter>',
    first_letter,
)

DATE_YEAR = SparqlKB.Augment(
    '<http://wikidata.dbpedia.org/ontology/releaseDate>',
    '<http://wikidata.dbpedia.org/ontology/releaseYear>',
    date_to_year,
)

DATE_DECADE = SparqlKB.Augment(
    '<http://wikidata.dbpedia.org/ontology/releaseDate>',
    '<http://wikidata.dbpedia.org/ontology/releaseDecade>',
    date_to_decade,
)
