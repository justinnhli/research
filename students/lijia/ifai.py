import time
import sys
from collections import defaultdict
from functools import lru_cache as memoize
from os.path import dirname, realpath, join as join_path

import numpy as np
import requests
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, words
from PyDictionary import PyDictionary

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from research.knowledge_base import KnowledgeFile, URI
from research.word_embedding import load_model

# download wordnet
nltk.download('wordnet')
nltk.download('words')

GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')

UMBEL = KnowledgeFile(UMBEL_KB_PATH)

SPACY_NLP = spacy.load('en')
LEMMATIZER = WordNetLemmatizer()
DICTIONARY = PyDictionary()

# Utility Functions


def get_ave_sigma(model, canons):
    """compute the average sigma (vector difference of a word pair) of a list of canonical pairs"""
    sigma = 0
    for pair in canons:
        word1, word2 = pair.split()
        sigma += model.word_vec(word1) - model.word_vec(word2)
    ave_sigma = (1 / len(canons)) * sigma
    return ave_sigma


@memoize(maxsize=None)
def get_word_list_path(word_list_file):
    return join_path(dirname(realpath(__file__)), 'word_lists', word_list_file)


@memoize(maxsize=None)
def prepare_list_from_file(file_name):
    """extract a list of word(s) from a file"""
    with open(file_name) as fd:
        canons = [line.strip() for line in fd.readlines()]
    return canons


def cosine_distance(v1, v2):
    """calculate the cosine distance of two vectors"""
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


@memoize(maxsize=None)
def is_word(word):
    return word.lower() in words.words()


@memoize(maxsize=None)
def to_imperative(verb):
    if not is_word(verb):
        return None
    return LEMMATIZER.lemmatize('running', wn.VERB)


def w2v_get_verbs_for_noun(model, noun):
    """return a list of lemmatized verbs that the noun can afford from a given word2vec model"""
    # load word lists
    canons = prepare_list_from_file(get_word_list_path('verb_noun_pair.txt'))

    # compute average sigma
    sigma = get_ave_sigma(model, canons)

    # extract words from word2vec model & append lemmatized word to list
    model_verbs = model.most_similar([sigma, noun], [], topn=10)
    word2vec_words = [to_imperative(verb[0].lower()) for verb in model_verbs]
    word2vec_words = [verb for verb in word2vec_words if verb]

    return word2vec_words


def w2v_get_adjectives_for_noun(model, noun):
    """return a list of adjectives that describe the given noun"""
    # get average sigma of the adj_noun canonical pairs
    canons = prepare_list_from_file(get_word_list_path('adj_noun_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract adjectives from w2v model with the sigma
    model_adjectives = model.most_similar([sigma, noun], [], topn=10)
    return  [adj[0] for adj in model_adjectives if wn.morphy(adj[0], wn.ADJ)]


def w2v_get_nouns_for_adjective(model, noun):
    """return a list of nouns that can be describes in adjectives way"""
    # get average sigma of the noun_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('noun_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract nouns from w2v model with the sigma
    model_nouns = model.most_similar([sigma, noun], [], topn=10)
    return [noun[0] for noun in model_nouns if wn.morphy(noun[0], wn.NOUN)]


def w2v_get_verbs_for_adjective(model, adj):
    """return a list of verbs that the given adj can be used in such way"""
    # get average sigma of the verb_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('verb_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract verbs from w2v model with the sigma
    model_verbs = model.most_similar([sigma, adj], [], topn=10)
    return [verb[0] for verb in model_verbs]


def w2v_get_tools_for_verb(model, verb):
    """get possible tools to realize the intended action"""
    # get average sigma of the verb_noun canonical pairs
    canons = prepare_list_from_file(get_word_list_path('verb_noun_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract verbs from w2v model with the sigma
    model_tools = model.most_similar([verb], [sigma], topn=10)
    return [tool[0] for tool in model_tools]


def rank_tools_cos(model, verb, tools):
    """Rank tools using cosine distance.

    Specifically, the cosine distance from the verb-tool-pair vector to
    canonical vector.
    """
    canons = prepare_list_from_file(get_word_list_path('verb_tool_list.txt'))
    sigma = get_ave_sigma(model, canons)
    tool_dic = {}

    # Calculate cosine distance of two vectors
    for tool in tools:
        verb_tool_vec = model.word_vec(verb) - model.word_vec(tool)
        tool_dic[tool] = cosine_distance(sigma, verb_tool_vec)

    return sorted(tool_dic.items(), key=(lambda kv: kv[1]), reverse=True)


def rank_tool_l2(model, verb, tools):
    """Rank tools using L2 distance."""
    canons = prepare_list_from_file(get_word_list_path('verb_tool_list.txt'))
    sigma = get_ave_sigma(model, canons)
    tool_dic = {}

    # Calculate cosine distance of two vectors
    for tool in tools:
        verb_tool_vec = model.word_vec(verb) - model.word_vec(tool)
        tool_dic[tool] = np.linalg.norm(sigma - verb_tool_vec)

    return sorted(tool_dic.items(), key=(lambda kv: kv[1]))


def w2v_rank_manipulability(model, nouns):
    """rank inputs nouns from most manipulative to least manipulative"""
    # anchor x_axis by using forest & tree vector difference
    x_axis = model.word_vec("forest") - model.word_vec("tree")
    dic = {}

    # map the noun's vectors to the x_axis and spit out a list from small to big
    for noun in nouns:
        if noun not in dic:
            vec = model.word_vec(noun)
            dic[noun] = np.dot(vec, x_axis)
    sorted_list = sorted(dic.items(), key=(lambda kv: kv[1]))
    return sorted_list


def cn_get_relations_for_concept(concept, relations, limit=None):
    """Get results from ConceptNet in a generic way.

    Arguments:
        concept (str): The concept to look up.
        relations (list[str]): The list of relations to look up.

    Returns:
        list: A list of [word, weight] pairs.
    """
    url_template = 'http://api.conceptnet.io/query?node=/c/en/{concept}&rel=/r/{relation}'
    word = concept.replace(' ', '_')
    results = defaultdict(float)
    for relation in relations:
        # query ConceptNet
        url = url_template.format(concept=word, relation=relation)
        # parse the result
        json = requests.get(url).json()
        for edge in json['edges']:
            language = edge['end']['language']
            if language != 'en':
                continue
            # get the answer from the edge
            answer = edge['end']['label']
            # add to results with weight
            results[answer] += edge['weight']
    sorted_list = sorted(results.items(), key=(lambda kv: kv[1]), reverse=True)
    if limit:
        return sorted_list[:limit]
    else:
        return sorted_list


def cn_get_verbs_for_noun(noun):
    """return a list of possible verbs with weight for the given noun from ConceptNet"""
    raw_results = cn_get_relations_for_concept(noun, ['CapableOf', 'UsedFor'])
    results = []
    for verb, weight in raw_results:
        verb = verb.split()[0]
        verb = to_imperative(verb)
        if verb:
            results.append([verb, weight])
    return results[:10]


def cn_get_adjectives_for_noun(noun):
    """return a list of adj best describe the noun from ConceptNet"""
    raw_results = cn_get_relations_for_concept(noun, ['HasProperty'])
    return [
        [wn.morphy(adjective, wn.ADJ), weight]
        for adjective, weight in raw_results
    ][:10]


def cn_get_materials_for_noun(noun):
    """return a list of possible materials with weight for the given noun from ConceptNet"""
    return cn_get_relations_for_concept(noun, ['HasProperty'])[:10]


def cn_get_locations(noun):
    """return a list of locations that the noun possibly locate in"""
    raw_results = cn_get_relations_for_concept(noun, ['AtLocation', 'LocatedNear', 'PartOf'])
    return [[location, weight] for location, weight in raw_results if location != noun][:10]


@memoize(maxsize=None)
def get_synonyms(word, pos=None):
    """return a list of synonym of the noun from PyDictionary and wordnet

    Arguments:
        word (str): The word to find synonyms for.
        pos (int): WordNet part-of-speech constant.

    Returns:
        set[str]: The set of synonyms.
    """
    syn_list = []

    # add wordnet synonyms to the list
    for synset in wn.synsets(word, pos):
        for lemma in synset.lemmas():
            syn = lemma.name()
            if syn != word:
                syn_list.append(syn)

    # add thesaurus synonyms
    dict_syns = DICTIONARY.synonym(word)
    if dict_syns:
        return set(syn_list) | set(dict_syns)
    else:
        return set(syn_list)


def umbel_is_manipulable_noun(noun):

    def get_all_superclasses(kb, concept):
        superclasses = set()
        queue = [str(URI(concept, 'umbel-rc'))]
        query_template = 'SELECT ?parent WHERE {{ {child} {relation} ?parent . }}'
        while queue:
            child = queue.pop(0)
            query = query_template.format(child=child, relation=URI('subClassOf', 'rdfs'))
            for bindings in kb.query_sparql(query):
                parent = str(bindings['parent'])
                if parent not in superclasses:
                    superclasses.add(parent)
                    queue.append(str(URI(parent)))
        return superclasses

    # create superclass to check against
    solid_tangible_thing = URI('SolidTangibleThing', 'umbel-rc').uri
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


def filter_nouns(nouns):
    return [noun for noun in nouns if wn_is_manipulable_noun(noun) or umbel_is_manipulable_noun(noun)]


# MAIN FUNCTIONS


def get_verbs_for_adj(model, adj):
    """Get verbs that can be caused by the adjective.
    
    For example, sharp -> cut
    """
    return w2v_get_verbs_for_adjective(model, adj)


def get_verbs_for_noun(model, noun):
    """Get list of verb that the noun can afford."""
    w2v_ls = w2v_get_verbs_for_noun(model, noun)
    cn_ls = cn_get_verbs_for_noun(noun)
    return set(w2v_ls) | set(kv[0] for kv in cn_ls)


def get_adjectives_for_noun(model, noun):
    """get list of adj that describe the noun"""
    w2v_ls = w2v_get_adjectives_for_noun(model, noun)
    cn_ls = cn_get_adjectives_for_noun(noun)
    return set(w2v_ls) | set(kv[0] for kv in cn_ls)


def get_nouns_from_text(text):
    """extract noun from given text"""

    # tokenize the given text with SpaCy
    doc = SPACY_NLP(text)
    # collect lemmatized nouns from tokens
    nouns = set([LEMMATIZER.lemmatize(chunk.root.text.lower(), wn.NOUN) for chunk in doc.noun_chunks])

    # filter out non-tangible nouns
    nouns = filter_nouns(nouns)

    return nouns


def possible_actions(model, text):
    """return a list of possible actions that can be done to nouns in the text"""
    nouns = get_nouns_from_text(text)

    # rank nouns in terms of manipulability [most manipulative-----less manipulative]
    sorted_list = w2v_rank_manipulability(model, nouns)

    # for each noun, find relevant verbs and add them to the list of results
    action_pair = []
    for word in sorted_list:
        verbs = get_verbs_for_noun(model, word[0])
        action_pair.extend([(verb + " " + word[0]) for verb in verbs])

    return action_pair


# # todo add in potential tools that can be used for the action (e.g.: cut string with shard)
# def possible_tools(model, verb):
#     tools = w2v_get_tools_for_verb(model, verb)
#
#     action_with_tool = []
#     for tool in tools:
#
#


def main():
    model = load_model(GOOGLE_NEWS_MODEL_PATH)

    # start timing
    tic = time.time()

    # prepare samples
    test_nouns = ["book", "sword", "horse", "key"]
    test_verbs = ["climb", "use", "open", "lift", "kill", "murder", "drive", "ride", "cure", "type", "sing"]
    test_adjectives = ["sharp", "heavy", "hot", "iced", "clean", "long"]
    sentences = [
        "Soon you’ll be able to send and receive money from friends and family right in Messages.",
        "This is an open field west of a white house, with a boarded front door. There is a small mailbox here.",
        "This is a forest, with trees in all directions around you.",
        "This is a dimly lit forest, with large trees all around.  One particularly large tree with some low branches stands here.",
        "You open the mailbox, revealing a small leaflet.",
    ]

    verbs = ["cut", "open", "write", "drink"]
    tools = [
        "knife", "ax", "brain", "neuron", "cup", "computer", "lamp", "pen", "needle", "scissors", "door", "key", "box",
        "building", "life", "glass", "water", "computer"
    ]

    for verb in verbs:
        print(verb, ":")
        print("cosine distance: ", rank_tools_cos(model, verb, tools))
        print("euclidean distance: ", rank_tool_l2(model, verb, tools))

    toc = time.time()
    print("total time spend:", toc - tic, "s")


if __name__ == "__main__":
    main()
