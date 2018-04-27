import time
import sys
from collections import OrderedDict
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

from research.knowledge_base import KnowledgeFile, Query, Node, U, V
from research.word_embedding import load_model

# download wordnet
nltk.download('wordnet')
nltk.download('words')

GOOGLE_NEWS_MODEL_PATH = join_path(ROOT_DIRECTORY, 'data/models/GoogleNews-vectors-negative300.bin')
UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')

UMBEL = KnowledgeFile(UMBEL_KB_PATH)

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
    verb_list = prepare_list_from_file(get_word_list_path('top_1000_verbs.txt'))

    # compute average sigma
    sigma = get_ave_sigma(model, canons)

    # list of common used verbs
    navigation_verbs = [
        "north", "south", "east", "west", "northeast", "southeast", "southwest", "northwest", "up", "down", "enter",
        "exit"
    ]
    essential_manipulation_verbs = ["get", "drop", "push", "pull", "open", "close"]

    # extract words from word2vec model & append lemmatized word to list
    model_verb = model.most_similar([sigma, noun], [], topn=10)
    word2vec_words = []
    wnl = WordNetLemmatizer()
    for verb in model_verb:
        verb = wnl.lemmatize(str(verb[0].lower()))

        # use wordnet to assert verb (can be a verb)
        if wn.morphy(verb, wn.VERB):
            word2vec_words.append(verb)

    # set operations
    affordant_verbs = list(set(verb_list) & set(word2vec_words))

    return affordant_verbs

    # uncomment to include generic verbs
    #return list(set(navigation_verbs) | set(essential_manipulation_verbs) | set(affordant_verbs))


def w2v_get_adjectives_for_noun(model, noun):
    """return a list of adjectives that describe the given noun"""
    # get average sigma of the adj_noun canonical pairs
    canons = prepare_list_from_file(get_word_list_path('adj_noun_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract adjectives from w2v model with the sigma
    model_adjectives = model.most_similar([sigma, noun], [], topn=10)
    adjectives = [adj[0] for adj in model_adjectives if wn.morphy(adj[0], wn.ADJ)]

    return adjectives


def w2v_get_nouns_for_adjective(model, noun):
    """return a list of nouns that can be describes in adjectives way"""
    # get average sigma of the noun_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('noun_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract nouns from w2v model with the sigma
    model_nouns = model.most_similar([sigma, noun], [], topn=10)
    nouns = [noun[0] for noun in model_nouns if wn.morphy(noun[0], wn.NOUN)]

    return nouns


def w2v_get_verbs_for_adjective(model, adj):
    """return a list of verbs that the given adj can be used in such way"""
    # get average sigma of the verb_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('verb_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract verbs from w2v model with the sigma
    model_verbs = model.most_similar([sigma, adj], [], topn=10)
    verbs = [verb[0] for verb in model_verbs]
    return verbs


def w2v_get_tools_for_verb(model, verb):
    """get possible tools to realize the intended action"""
    # get average sigma of the verb_noun canonical pairs
    canons = prepare_list_from_file(get_word_list_path('verb_noun_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract verbs from w2v model with the sigma
    model_tools = model.most_similar([verb], [sigma], topn=10)
    tools = [tool[0] for tool in model_tools]
    return tools


def rank_tools_cos(model, verb, tools):
    """rank tool with regard to verb by measuring the cosine distance from the verb-tool-pair vector to canonical vector"""
    canons = prepare_list_from_file(get_word_list_path('verb_tool_list.txt'))
    sigma = get_ave_sigma(model, canons)
    tool_dic = {}

    # Calculate cosine distance of two vectors
    for tool in tools:
        verb_tool_vec = model.word_vec(verb) - model.word_vec(tool)
        tool_dic[tool] = cosine_distance(sigma, verb_tool_vec)

    sorted_list = sorted(tool_dic.items(), key=(lambda kv: kv[1]), reverse=True)

    return sorted_list


def rank_tool_l2(model, verb, tools):
    """rank tool with regard to verb by measuring the euclidean distance from the verb-tool-pair vector to canonical vector"""
    canons = prepare_list_from_file(get_word_list_path('verb_tool_list.txt'))
    sigma = get_ave_sigma(model, canons)
    tool_dic = {}

    # Calculate cosine distance of two vectors
    for tool in tools:
        verb_tool_vec = model.word_vec(verb) - model.word_vec(tool)
        tool_dic[tool] = np.linalg.norm(sigma - verb_tool_vec)

    sorted_list = sorted(tool_dic.items(), key=(lambda kv: kv[1]))

    return sorted_list


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


def cn_get_verbs_for_noun(noun):
    """return a list of possible verbs with weight for the given noun from conceptNet"""
    v_dic = {}
    wnl = WordNetLemmatizer()

    # query conceptNet
    rel_list = ["CapableOf", "UsedFor"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + noun + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:

            # get the verb from the edge
            verb = edge["end"]["label"].split()[0]

            # use wordnet to assert verb (can be a verb)
            if wn.morphy(verb, wn.VERB):
                verb = wnl.lemmatize(str(verb.lower()))

                # add to dic with weight
                if verb not in v_dic:
                    v_dic[verb] = edge["weight"]
                if verb in v_dic:
                    v_dic[verb] += edge["weight"]

    sorted_list = sorted(v_dic.items(), key=(lambda kv: kv[1]), reverse=True)
    return sorted_list[:10]


def cn_get_adjectives_for_noun(noun):
    """return a list of adj best describe the noun from ConceptNet"""
    adj_dic = {}

    rel_list = ["HasProperty"]
    for rel in rel_list:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/' + noun + '&rel=/r/' + rel).json()
        for edge in obj["edges"]:

            # get the adj version of the word
            word = wn.morphy(edge["end"]["label"], wn.ADJ)

            # add to dic with weight
            if word not in adj_dic:
                adj_dic[word] = edge["weight"]
            if word in adj_dic:
                adj_dic[word] += edge["weight"]

    sorted_list = sorted(adj_dic.items(), key=(lambda kv: kv[1]), reverse=True)
    return sorted_list


def cn_get_locations(noun):
    """return a list of locations that the noun possibly locate in"""
    loca_list = []
    rel_list = ["AtLocation", "LocatedNear", "PartOf"]
    for rel in rel_list:
        url = 'http://api.conceptnet.io/query?node=/c/en/' + noun.replace(" ", "_") + '&rel=/r/' + rel
        obj = requests.get(url).json()
        for edge in obj["edges"]:
            if edge["end"]["language"] == 'en':
                syn = edge["end"]["label"]
                if syn not in loca_list and syn != noun:
                    loca_list.append(syn)
    return loca_list


@memoize(maxsize=None)
def get_synonyms(word, pos=None):
    """return a list of synonym of the noun from PyDictionary and wordnet

    Arguments:
        word (str): the word to find synonyms for
        pos (FIXME): WordNet part-of-speech constant
    """
    syn_list = []

    # add wordnet synonyms to the list
    for synset in wn.synsets(word, pos):
        for lemma in synset.lemmas():
            syn = lemma.name()
            if syn != word:
                syn_list.append(syn)

    # add theraurus synonyms
    dict_syns = DICTIONARY.synonym(word)
    if dict_syns:
        return set(syn_list) | set(dict_syns)
    else:
        return set(syn_list)


def combine_list(w2v_ls, cn_ls):
    """combine word2vec list and knowledge base list"""
    combine_ls = w2v_ls
    for element in cn_ls:
        word = element[0]
        if word not in combine_ls:
            combine_ls.append(word)

    return combine_ls


def filter_nouns(nouns):

    def get_all_ancestors(kb, relation, concept):
        triples = set()
        visited = set()
        queue = [U(concept, 'umbel-rc')]
        kwargs = {('rdfs__' + relation): V('parent')}
        query = Query(V('child', **kwargs))
        while queue:
            child = queue.pop(0)
            if str(child) in visited:
                continue
            visited.add(str(child))
            results = kb.query(query, child=child).splitlines()
            triples.update(results)
            for triple in results:
                parent = triple.split()[2]
                if str(parent) not in visited:
                    queue.append(Node.from_str(parent))
        return triples

    # create superclass to check against
    solid_tangible_thing = U('SolidTangibleThing', 'umbel-rc')
    # open Umbel
    result = []
    for noun in nouns:
        # get all synonyms of the word
        synonyms = get_synonyms(noun, wn.NOUN)
        synonyms.add(noun)
        for synonym in sorted(synonyms):
            # find the corresponding concept
            variations = [synonym, synonym.lower(), synonym.title()]
            variations = [variation.replace(' ', '') for variation in variations]
            for variation in variations:
                # find all ancestors
                triples = get_all_ancestors(UMBEL, 'subClassOf', variation)
                superclasses = set()
                for triple in triples:
                    if str(solid_tangible_thing) == triple.split(' ')[2]:
                        result.append(noun)
                        break
                if result and result[-1] == noun:
                    break
            if result and result[-1] == noun:
                break
    return result


# MAIN FUNCTIONS


def get_verb_for_adj(model, adj):
    """get list of verb that can be caused by the adj(e.g.: sharp -> cut)"""

    return w2v_get_verbs_for_adjective(model, adj)


def get_verbs_for_noun(model, noun):
    """get list of verb that the noun can afford"""
    w2v_ls = w2v_get_verbs_for_noun(model, noun)
    cn_ls = cn_get_verbs_for_noun(noun)
    combine_ls = combine_list(w2v_ls, cn_ls)
    return combine_ls


def get_adjectives_for_noun(model, noun):
    """get list of adj that describe the noun"""
    w2v_ls = w2v_get_adjectives_for_noun(model, noun)
    cn_ls = cn_get_adjectives_for_noun(noun)
    combine_ls = combine_list(w2v_ls, cn_ls)
    return combine_ls


def get_noun_from_text(text):
    """extract noun from given text"""

    # tokenize the given text with SpaCy
    nlp = spacy.load('en')
    doc = nlp(text)
    # collect lemmatized nouns from tokens
    wnl = WordNetLemmatizer()
    nouns = set([wnl.lemmatize(str(chunk.root.text.lower())) for chunk in doc.noun_chunks])
    nouns = list(OrderedDict.fromkeys(nouns))

    # filter out non-tangible nouns
    nouns = filter_nouns(nouns)

    return nouns


def possible_actions(model, text):
    """return a list of possible actions that can be done to nouns in the text"""
    nouns = get_noun_from_text(text)

    # rank nouns in terms of manipulatbility [most manipulative-----less manipulative]
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
    tools = ["knife", "ax", "brain", "neuron", "cup","computer", "lamp", "pen", "needle", "scissors", "door", "key", "box", "building", "life", "glass", "water", "computer"]

    for verb in verbs:
        print(verb, ":")
        print("cosine distance: ", rank_tools_cos(model, verb, tools))
        print("euclidean distance: ", rank_tool_l2(model, verb, tools))

    toc = time.time()
    print("total time spend:", toc - tic, "s")


if __name__ == "__main__":
    main()
