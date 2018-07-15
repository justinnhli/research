
from students.lijia.word2vec import *
from students.lijia.extraction import *    # todo: rename extraction.py

ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)


GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
model = load_model(GOOGLE_NEWS_MODEL_PATH)


def get_action(sentence):
    word2vec_actions = possible_actions(model, sentence)
    probability_actions = get_action_for_text(sentence)