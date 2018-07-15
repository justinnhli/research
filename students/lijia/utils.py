import re
import sys
import spacy
import nltk
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, words
from PyDictionary import PyDictionary
from os import listdir
from os.path import dirname, realpath, join as join_path

from research.knowledge_base import KnowledgeFile, URI
from research.word_embedding import load_model

from students.lijia.word2vec import wn_is_manipulable_noun, umbel_is_manipulable_noun

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

# setting up static parameters
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)
# STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'data/fanfic_stories')
STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'students/lijia/fantacy')
# OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'students/lijia/output_files/retest_output')
OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'students/lijia/temp_test')
NP_DIR = join_path(OUTPUT_DIR, "np")
VO_DIR = join_path(OUTPUT_DIR, "vpo")

# download wordnet
nltk.download('wordnet')
nltk.download('words')

GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')

UMBEL = KnowledgeFile(UMBEL_KB_PATH)
model = 'en_core_web_sm'

nlp = spacy.load(model)
LEMMATIZER = WordNetLemmatizer()
DICTIONARY = PyDictionary()


# UTILS
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

