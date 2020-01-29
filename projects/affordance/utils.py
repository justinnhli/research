"""utils contains a list of functions that are shared in multiple module"""

import re
import sys
from os import listdir
from os.path import join as join_path, dirname, realpath
from functools import lru_cache as memoize
import spacy
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary

# update and load models
nltk.download('wordnet', download_dir=nltk.data.path[0])
nltk.download('words', download_dir=nltk.data.path[0])
DICTIONARY = PyDictionary()

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from research.knowledge_base import KnowledgeFile, Value

UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')
UMBEL = KnowledgeFile(UMBEL_KB_PATH)

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


def get_sentence_from_file(directory, filename):
    """yield individual sentences from given file"""
    for line in open(join_path(directory, filename), encoding='utf-8'):
        sentence_ls = line.replace("\"", "").split(". ")
        for sentence in sentence_ls:
            yield sentence


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


@memoize(maxsize=None)
def get_synonyms(word, pos=None):
    """Get synonyms for a word using PyDictionary and WordNet.

    Arguments:
        word (str): The word to find synonyms for.
        pos (int): WordNet part-of-speech constant. Defaults to None.

    Returns:
        set[str]: The set of synonyms.
    """
    syn_list = []

    # add WordNet synonyms to the list
    for synset in wn.synsets(word, pos):
        for lemma in synset.lemmas():
            syn = lemma.name()
            if syn != word:
                syn_list.append(syn)
    # add thesaurus synonyms
    dict_syns = DICTIONARY.synonym(word)
    # combine them and return
    if dict_syns:
        return set(syn_list) | set(dict_syns)
    else:
        return set(syn_list)


def umbel_is_manipulable_noun(noun):

    def get_all_superclasses(kb, concept):
        superclasses = set()
        queue = [str(Value.from_namespace_fragment('umbel-rc', concept))]
        query_template = 'SELECT ?parent WHERE {{ {child} {relation} ?parent . }}'
        while queue:
            child = queue.pop(0)
            query = query_template.format(child=child, relation=Value.from_namespace_fragment('rdfs', 'subClassOf'))
            for bindings in kb.query_sparql(query):
                parent = str(bindings['parent'])
                if parent not in superclasses:
                    superclasses.add(parent)
                    queue.append(str(Value.from_uri(parent)))
        return superclasses

    # create superclass to check against
    solid_tangible_thing = Value.from_namespace_fragment('umbel-rc', 'SolidTangibleThing').uri
    for synonym in get_synonyms(noun, wn.NOUN):
        # find the corresponding concept
        variations = [synonym, synonym.lower(), synonym.title()]
        variations = [variation.replace(' ', '') for variation in variations]
        # find all ancestors of all variations
        for variation in variations:
            if solid_tangible_thing in get_all_superclasses(UMBEL, variation):
                return True
    return False


def wn_is_manipulable_noun(noun):

    def get_all_hypernyms(root_synset):
        hypernyms = set()
        queue = [root_synset]
        while queue:
            synset = queue.pop(0)
            new_hypernyms = synset.hypernyms()
            for hypernym in new_hypernyms:
                if hypernym.name() not in hypernyms:
                    hypernyms.add(hypernym.name())
                    queue.append(hypernym)
        return hypernyms

    for synset in wn.synsets(noun, pos=wn.NOUN):
        if 'physical_entity.n.01' in get_all_hypernyms(synset):
            return True
    return False
