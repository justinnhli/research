"""word2vec contains a list of functions that extract information from word embedding model"""

import sys
from collections import defaultdict, Counter
from functools import lru_cache as memoize
from os.path import dirname, realpath
import numpy as np
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn, words

# make sure research library code is available
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)

from students.lijia.utils import *
from research.knowledge_base import KnowledgeFile


UMBEL_KB_PATH = join_path(ROOT_DIRECTORY, 'data/kbs/umbel-concepts-typology.rdfsqlite')
UMBEL = KnowledgeFile(UMBEL_KB_PATH)
LEMMATIZER = WordNetLemmatizer()
DICTIONARY = PyDictionary()


# Utility Functions
def get_ave_sigma(model, pairs):
    """Calculate the average vector between word pairs.

    Arguments:
        model (VectorModel): Gensim word vector model.
        pairs (list[str]): List of word pairs (as space-separated strings)

    Returns:
        Vector: The mean vector.
    """
    sigma = 0
    for pair in pairs:
        word1, word2 = pair.split()
        sigma += model.word_vec(word1) - model.word_vec(word2)
    ave_sigma = sigma / len(pairs)
    return ave_sigma


@memoize(maxsize=None)
def get_word_list_path(word_list_file):
    return join_path(dirname(realpath(__file__)), 'word_lists', word_list_file)


@memoize(maxsize=None)
def prepare_list_from_file(file_name):
    """Read lines from a file.

    Arguments:
        file_name (str): The file name.

    Returns:
        list[str]: List of lines in the file.
    """
    with open(file_name) as fd:
        lines = [line.strip() for line in fd.readlines()]
    return lines


def cosine_distance(vec1, vec2):
    """Calculate the cosine distance between two vectors.

    Arguments:
        vec1 (Vector): Word vector from Gensim.
        vec2 (Vector): Word vector from Gensim.

    Returns:
        float: The cosine distance between the vectors.
    """
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))


@memoize(maxsize=None)
def is_word(word):
    return word.lower() in words.words()


@memoize(maxsize=None)
def to_imperative(verb):
    if not is_word(verb):
        return None
    return LEMMATIZER.lemmatize(verb, wn.VERB)


def w2v_get_verbs_for_noun(model, noun):
    """Get verbs that count be applied to a noun using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        noun (str): The noun to find verbs for.

    Returns:
        list[str]: List of verbs.
    """
    # have each cannonical pair vote on the appropriate verbs
    verbs = Counter()
    canon = prepare_list_from_file(get_word_list_path('verb_noun_pair.txt'))
    for pair in canon:
        verb, obj = pair.split()
        # calculate the vector for the canonical pair
        affordance_vector = model.word_vec(verb) - model.word_vec(obj)
        # find word analogous to that relationship
        weighted_words = model.most_similar(positive=[model.word_vec(noun) + affordance_vector])
        # ignore the weights
        candidate_verbs = [pair[0] for pair in weighted_words]
        # verify that they are verbs
        verified_verbs = [
            wn.morphy(candidate, wn.VERB) for candidate in candidate_verbs if wn.synsets(candidate, wn.VERB)
        ]
        # filter out None's
        verified_verbs = [verified_verb for verified_verb in verified_verbs if verified_verb is not None]
        # count the them as votes
        verbs.update(verified_verbs)
    # return everything that appears more than once
    return [verb for verb, votes in verbs.most_common() if votes > 1]


def w2v_get_adjectives_for_noun(model, noun):
    """Get adjectives that describe a noun using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        noun (str): The noun to find adjectives for.

    Returns:
        list[str]: List of adjectives.
    """
    # get average sigma of the adj_noun canonical pairs
    canons = prepare_list_from_file(get_word_list_path('adj_noun_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract adjectives from w2v model with the sigma
    model_adjectives = model.most_similar([sigma, noun], [], topn=10)
    return [adj[0] for adj in model_adjectives if wn.morphy(adj[0], wn.ADJ)]


def w2v_get_nouns_for_adjective(model, adjective):
    """Get nouns that are described by an adjective using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        adjective (str): The adjective to find nouns for.

    Returns:
        list[str]: List of nouns.
    """
    # get average sigma of the noun_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('noun_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract nouns from w2v model with the sigma
    model_adjs = model.most_similar([sigma, adjective], [], topn=10)
    return [adj[0] for adj in model_adjs if wn.morphy(adj[0], wn.ADJ)]


def w2v_get_verbs_for_adjective(model, adjective):
    """Get verbs that objects described by an adjective could afford.

    Arguments:
        model (VectorModel): Gensim word vector model.
        adjective (str): The adjective to find verbs for.

    Returns:
        list[str]: List of verbs.
    """
    # get average sigma of the verb_adj canonical pairs
    canons = prepare_list_from_file(get_word_list_path('verb_adj_pair.txt'))
    sigma = get_ave_sigma(model, canons)

    # extract verbs from w2v model with the sigma
    model_verbs = model.most_similar([sigma, adjective], [], topn=10)
    return [verb[0] for verb in model_verbs]


def w2v_get_tools_for_verb(model, verb):
    """Get tools that could be used to perform an action using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        verb (str): The verb to find tools for.

    Returns:
        list[str]: List of tools.
    """
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

    Arguments:
        model (VectorModel): Gensim word vector model.
        verb (str): The verb to find tools for.
        tools (list[str]): The list of tools to rank.

    Returns:
        list[tuple[str, float]]: List of tool-weight pairs.
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
    """Rank tools using L2 distance.

    Arguments:
        model (VectorModel): Gensim word vector model.
        verb (str): The verb to find tools for.
        tools (list[str]): The list of tools to rank.

    Returns:
        list[tuple[str, float]]: List of tool-weight pairs.
    """
    canons = prepare_list_from_file(get_word_list_path('verb_tool_list.txt'))
    sigma = get_ave_sigma(model, canons)
    tool_dic = {}

    # Calculate cosine distance of two vectors
    for tool in tools:
        verb_tool_vec = model.word_vec(verb) - model.word_vec(tool)
        tool_dic[tool] = np.linalg.norm(sigma - verb_tool_vec)

    return sorted(tool_dic.items(), key=(lambda kv: kv[1]))


def w2v_rank_manipulability(model, nouns):
    """Rank nouns from most manipulable to least manipulable.

    Arguments:
        model (VectorModel): Gensim word vector model.
        nouns (list[str]): The nouns to rank.

    Returns:
        list[tuple[str, float]]: The list of nouns and cosine distances.
    """
    # anchor x_axis by using forest & tree vector difference
    manipulable_basis = model.word_vec('forest') - model.word_vec('tree')
    # map the noun's vectors to the x_axis and spit out a list from small to big
    word_cosine_list = [tuple([noun, np.dot(model.word_vec(noun), manipulable_basis)]) for noun in set(nouns)]

    return sorted(word_cosine_list, key=(lambda kv: kv[1]))


def cn_get_relations_for_concept(concept, relations, limit=None):
    """Get results from ConceptNet in a generic way.

    Arguments:
        concept (str): The concept to look up.
        relations (list[str]): The list of relations to look up.
        limit (int): The number of results to return. Defaults to None.

    Returns:
        list[tuple[str, float]]: A list of [word, weight] pairs.
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


def cn_get_verbs_for_noun(noun, min_count=1):
    """Get verbs that count be applied to a noun using ConceptNet.

    Arguments:
        noun (str): The noun to find adjectives for.

    Returns:
        list[str]: List of verbs.
    """
    raw_results = cn_get_relations_for_concept(noun, ['CapableOf', 'UsedFor'])
    verbs = Counter()
    for verb, _ in raw_results:
        verb = verb.split()[0]
        verb = wn.morphy(verb, wn.VERB)
        if verb:
            verbs.update([verb])
    # FIXME filter out tool entries from affordance entries
    # return everything that appears more than once
    return [verb for verb, votes in verbs.most_common() if votes >= min_count]


def cn_get_adjectives_for_noun(noun):
    """Get adjectives that describe a noun using ConceptNet.

    Arguments:
        noun (str): The noun to find adjectives for.

    Returns:
        list[str]: List of adjectives.
    """
    raw_results = cn_get_relations_for_concept(noun, ['HasProperty'])
    return [[wn.morphy(adjective, wn.ADJ), weight] for adjective, weight in raw_results][:10]


def cn_get_materials_for_noun(noun):
    """Get materials that a noun could be made of using ConceptNet.

    Arguments:
        noun (str): The noun to find the composition of.

    Returns:
        list[str]: List of materials.
    """
    return cn_get_relations_for_concept(noun, ['MadeOf'])[:10]


def cn_get_locations(noun):
    """Get locations that a noun could found using ConceptNet.

    Arguments:
        noun (str): The noun to find locations for.

    Returns:
        list[str]: List of locations.
    """
    raw_results = cn_get_relations_for_concept(noun, ['AtLocation', 'LocatedNear', 'PartOf'])
    return [[location, weight] for location, weight in raw_results if location != noun][:10]


def filter_nouns(nouns):
    return [noun for noun in nouns if wn_is_manipulable_noun(noun) or umbel_is_manipulable_noun(noun)]


# MAIN FUNCTIONS


def get_verbs_for_adjective(model, adjective):
    """Get verbs that objects described by an adjective could afford.

    For example, sharp -> cut

    Arguments:
        model (VectorModel): Gensim word vector model.
        adjective (str): The adjective to find verbs for.

    Returns:
        list[str]: List of verbs.
    """
    return w2v_get_verbs_for_adjective(model, adjective)


def get_verbs_for_noun(model, noun):
    """Get verbs that count be applied to a noun using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        noun (str): The noun to find verbs for.

    Returns:
        list[str]: List of verbs.
    """
    # get verbs from word2vec and ConceptNet
    w2v_verbs = w2v_get_verbs_for_noun(model, noun)
    cn_verbs = cn_get_verbs_for_noun(noun)
    verbs = [*w2v_verbs, *cn_verbs]
    # combine the results
    verbs = [*w2v_verbs, *cn_verbs]
    # sort them by decreasing frequency
    verbs = [verb for verb, _ in Counter(verbs).most_common() if verb != noun]
    # get the most common form of the verb
    verbs = [
        Counter([synset.name().split('.')[0] for synset in wn.synsets(verb, wn.VERB)]).most_common(1)[0][0]
        for verb in verbs
    ]
    # remove duplicates
    verbs = sorted(set(verbs), key=(lambda verb: verbs.index(verb)))
    # replace punctuation with spaces
    verbs = [re.sub('[-_]', ' ', verb) for verb in verbs]
    return verbs
    # FIXME things to do here
    # * check whether they are transitive verbs


def get_adjectives_for_noun(model, noun):
    """Get adjectives that describe a noun using word2vec.

    Arguments:
        model (VectorModel): Gensim word vector model.
        noun (str): The noun to find adjectives for.

    Returns:
        list[str]: List of adjectives.
    """
    w2v_ls = w2v_get_adjectives_for_noun(model, noun)
    cn_ls = cn_get_adjectives_for_noun(noun)
    adjectives = set(w2v_ls) | set(kv[0] for kv in cn_ls)
    # adjectives = [adjective for adjective in get_adjectives_for_noun(MODEL, noun) if adjective]
    return adjectives


def get_nouns_from_text(text):
    """Extract nouns from a given text.

    Arguments:
        text (str): The text to extract from.

    Returns:
        list[str]: A list of nouns.
    """
    # tokenize the given text with SpaCy
    doc = SPACY_NLP(text)
    # collect lemmatized nouns from tokens
    nouns = set([LEMMATIZER.lemmatize(chunk.root.text.lower(), wn.NOUN) for chunk in doc.noun_chunks])

    # filter out non-tangible nouns
    nouns = filter_nouns(nouns)

    return nouns


def possible_actions(model, text):
    """Get the possible actions to apply in a given description.

    Arguments:
        model (VectorModel): Gensim word vector model.
        text (str): The description of the situation.

    Returns:
        list[str]: A list of actions to apply.

    """
    nouns = get_nouns_from_text(text)

    # rank nouns in terms of manipulability [most manipulative-----less manipulative]
    sorted_list = w2v_rank_manipulability(model, nouns)

    # for each noun, find relevant verbs and add them to the list of results
    action_pair = []
    for word in sorted_list:
        verbs = get_verbs_for_noun(model, word[0])
        action_pair.extend([(verb + ' ' + word[0]) for verb in verbs])

    return action_pair

