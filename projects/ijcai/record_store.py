import re
from ast import literal_eval
from pathlib import Path

from research.rl_environments import State, Action, Environment
from research.rl_memory import SparqlKB
from research.randommixin import RandomMixin


class RecordStore(Environment, RandomMixin):

    def __init__(self, data_file=None, num_albums=1000, *args, **kwargs):
        # pylint: disable = keyword-arg-before-vararg
        super().__init__(*args, **kwargs)
        # parameters
        assert data_file is not None
        self.data_file = Path(data_file).resolve()
        self.num_albums = num_albums
        # database
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
        return State.from_dict(dict(self.question))

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
        with self.data_file.open() as fd:
            for question, answer in literal_eval(fd.read()):
                question = tuple(sorted(question.items()))
                answer = ' ; '.join(answer)
                self.answers[question] = answer
        self.questions = sorted(self.answers.keys())
        self.actions = set(self.answers.values())

    def start_new_episode(self):
        self.question = self.rng.choice(self.questions)
        self.location = '__start__'

    def visualize(self):
        raise NotImplementedError()


def first_letter(literal):
    match = re.fullmatch('"[^a-z]*([a-z]).*"(([@^][^"]*)?)', literal, flags=re.IGNORECASE)
    if match:
        initial = match.group(1).upper()
        metadata = match.group(2)
        return f'"{initial}"{metadata}'
    else:
        return None


def date_to_year(date):
    if re.fullmatch('"([0-9]{4}).*"(([@^][^"]*)?)', date):
        return re.sub('^"([0-9]{4}).*"(([@^][^"]*))?$', r'"\1-01-01"\2', date)
    else:
        return None


def date_to_decade(date):
    if re.fullmatch('"([0-9]{3}).*"(([@^][^"]*)?)', date):
        return re.sub('^"([0-9]{3}).*"(([@^][^"]*)?)$', r'"\g<1>0-01-01"\2', date)
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
