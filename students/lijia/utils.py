import re
import sys
from os import listdir
from os.path import dirname, realpath, join as join_path
import spacy
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from students.lijia.word2vec import wn_is_manipulable_noun, umbel_is_manipulable_noun

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)
GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')


# update and load models
nltk.download('wordnet')
nltk.download('words')

spacy_model = 'en_core_web_sm'
nlp = spacy.load(spacy_model)


# utility functions
def get_filename_from_folder(directory):
    """yield filename from given folder"""
    for filename in listdir(directory):
        if not filename.startswith("."):
            yield filename


def get_nlp_sentence_from_file(directory, filename):
    """yield tokenized individual sentences from given file"""
    for line in open(join_path(directory, filename), encoding='utf-8'):
        sentence_ls = line.replace("\"", "").split(". ")
        for sentence in sentence_ls:
            yield nlp(sentence)


def has_number(string):
    """check if the string contains a number"""
    return bool(re.search(r'\d', string))


def is_good_verb(token):
    """check if the token is an acceptable verb"""
    exclude_verbs = ['have']
    return token.pos_ == "VERB" \
           and wn.morphy(token.text, wn.VERB) \
           and not token.is_stop \
           and token.text not in exclude_verbs \
           and not token.text.startswith('\'')


def is_good_noun(token):
    """check if the token is an acceptable noun"""
    return token.pos_ == "NOUN"\
           and wn.morphy(token.text, wn.NOUN) \
           and not has_number(token.text)


def is_good_adj(token):
    """check if the token is an acceptable adj"""
    return token.pos_ == "ADJ" \
            and wn.morphy(token.text, wn.ADJ) \
            and not has_number(token.text)


def is_person(token):
    """check if the token refers to a person"""
    return token.lemma_ == "-PRON-" or token.ent_type_ == "PERSON" or token.tag_ == "WP"


def is_good_subj(token):
    """check if the token is a subject in the sentence"""
    return (token.pos_ == "NOUN" or token.pos_ == "PRON" or token.pos_ == "PROPN")\
           and token.dep_ == "nsubj"\
           and not has_number(token.text)


def is_good_obj(token):
    """check if the token is a object in the sentence"""
    return is_good_noun(token) and not is_person(token) and token.dep_ == "dobj"


def is_manipulable(token):
    """check if the token is manipubale or not by wordnet and umbel"""
    return is_good_noun(token) \
           and not is_person(token) \
           and (wn_is_manipulable_noun(token.lemma_) or umbel_is_manipulable_noun(token.lemma_))


def replace_wsd(doc, token):
    """return the wsd token of a doc"""
    return lesk(doc.text.split(), str(token))


def get_manipulable_noun(sentence):
    doc = nlp(sentence)
    results = []
    for token in doc:
        if is_manipulable(token):
            results.append(token.text)
    return results
