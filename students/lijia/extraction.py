"""extract phrase from folder and calculate p(verb_adj)"""

import re
import sys
import time
import datetime
from collections import Counter, defaultdict, namedtuple
from os import listdir
from os.path import dirname, realpath, join as join_path, exists as file_exists
import spacy
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from ifai import wn_is_manipulable_noun, umbel_is_manipulable_noun


# setting up static parameters
ROOT_DIRECTORY = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, ROOT_DIRECTORY)
# STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'data/fanfic_stories')
STORY_DIRECTORY = join_path(ROOT_DIRECTORY, 'students/lijia/fantacy')
# OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'students/lijia/output_files/retest_output')
OUTPUT_DIR = join_path(ROOT_DIRECTORY, 'students/lijia/temp_test')
NP_DIR = join_path(OUTPUT_DIR, "np")
VO_DIR = join_path(OUTPUT_DIR, "vpo")

# SVO = namedtuple('SVO', ('subject', 'verb', 'preposition', 'object'))
VPO = namedtuple('VPO', ('verb', 'prep', 'object'))
NP = namedtuple('NP', ['noun', 'adjectives'])

# load nlp model
model = 'en_core_web_sm'
nlp = spacy.load(model)


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


def read_to_nested_dict(filename, outer_index, inner_index, value_index):
    """read three arguments from file and store in nested dict

    :param filename
    :param outer_index: outer dictionary key index
    :param inner_index: inner dictionary key index
    :param value_index: value index
    :return: {outer_index: { inner_index: value_index }}
    """
    d = defaultdict(lambda: defaultdict(float))
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split()
            d[ls[outer_index]][ls[inner_index]] = float(ls[value_index])
    return d


def read_to_dict(filename, key_index, value_index):
    """

    :param filename:
    :param key_index:
    :param value_index:
    :return: key: [value1, value2, ..., valuen]
    """
    d = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split()
            if ls[key_index] in d:
                d[ls[key_index]].append(ls[value_index])
            else:
                d[ls[key_index]] = [ls[value_index]]
    return d


def search_file(filename, string):
    """search specific string in file"""
    with open(filename, 'r', encoding='utf-8') as f:
        results = {}
        for line in f.readlines():
            ls = line.split()
            if string == ls[0]:
                results[ls[1]] = float(ls[2])
    return results


def extract_vpo(doc):
    """extract verbs with prep and tangible objects

    :param doc: tokenized sentence
    :return: a list of namedtuple VPO
    """
    results = []
    for token in doc:
        if not is_good_verb(token):
            continue
        # look for direct and indirect objects
        for child in token.children:
            # direct objects
            if is_good_obj(child) and is_manipulable(child):
                results.append(VPO(verb=token.lemma_, prep=None, object=child.lemma_))
            # indirect (proposition) objects
            elif child.dep_ == "prep":
                objects = []
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj" and is_manipulable(grandchild):
                        objects.append(grandchild)
                        objects.extend([ggrandchild for ggrandchild in grandchild.children if ggrandchild.dep_ == "conj"])
                for obj in objects:
                    results.append(VPO(verb=token.lemma_, prep=child.lemma_, object=obj.lemma_))
        # if the verb has neither direct nor indirect objects
        if not results:
            continue
    return results


def extract_np(doc):
    """extract adjectives and tangible nouns

    :param doc: tokenized sentence
    :return: a list of namedtuple NP
    """
    results = []
    for token in doc:
        if not is_good_adj(token):
            continue

        # identify all subjects being described
        if token.dep_ == "amod" and is_manipulable(token.head):
            # eg. "The tall and skinny girl..."
            subjects = [token.head.lemma_]
        # elif token.dep_ == "acomp" and len([child for child in token.children]) == 0:
        elif token.dep_ == "acomp":
            # eg. "The girl is tall and skinny."
            subjects = [
                child.lemma_
                for child in token.head.children
                if child.dep_ == "nsubj" and is_manipulable(child)
            ]
        else:
            subjects = []

        adjectives = [token.lemma_]
        adjectives.extend(child.lemma_ for child in token.children if child.dep_ == "conj")

        for subject in subjects:
            results.append(NP(subject, adjectives))
    return results


def extract_from_directory(directory):
    """extract and write (adj noun) and (verb_prep noun) from given dump"""
    for file in get_filename_from_folder(directory):
        print(file)
        file_name_vo = join_path(VO_DIR, file[:-4] + "_vo.txt")
        file_name_np = join_path(NP_DIR, file[:-4] + "_np.txt")

        # start extraction if the file does not exist yet
        if file_exists(file_name_vo) and file_exists(file_name_np):
            continue

        vo_lines = []
        np_lines = []
        for doc in get_nlp_sentence_from_file(directory, file):
            for vo in extract_vpo(doc):
                try:
                    assert vo.verb != 'takes', "takes output occurs in sentence %s" % doc
                except AssertionError:
                    pass
                if vo.prep is not None:
                    line = "%s_%s %s" % (vo.verb, vo.prep, vo.object)
                else:
                    line = "%s %s" % (vo.verb, vo.object)
                vo_lines.append(line)
            for np in extract_np(doc):
                for adjective in np.adjectives:
                    np_lines.append("%s %s" % (adjective, np.noun))
        with open(file_name_vo, 'w', encoding='utf-8') as vo_file:
            vo_file.write('\n'.join(vo_lines))
        with open(file_name_np, 'w', encoding='utf-8') as np_file:
            np_file.write('\n'.join(np_lines))


def calculate_individual_p(directory, output_filename, outer_index, inner_index):
    """
    a generic function that calculate probability and count
    :param directory: the folder that contains extracted words
    :param output_filename
    :param outer_index: the index for condition (default dictionary's key)
    :param inner_index: the index for marginal (counter's key)
    :return:
    """
    if file_exists(join_path(OUTPUT_DIR, "count_" + output_filename)) \
            and file_exists(join_path(OUTPUT_DIR, "prob_" + output_filename)):
        print("both count and probability for (%s) exist" % output_filename)
        return

    def write_out_count():
        """ write out counts"""
        with open(join_path(OUTPUT_DIR, "count_" + output_filename), 'w', encoding='utf-8') as f:
            for k in d:
                c = d[k]
                f.write("%s (%s)\n" % (k, sum(c.values())))
                for word in d[k]:
                    f.write("---%s (%s)\n" % (word, c[word]))

    def write_out_prob():
        """write out probability for each pair"""
        with open(join_path(OUTPUT_DIR, "prob_" + output_filename), 'w', encoding='utf-8') as f:
            for k in d:
                c = d[k]
                total = sum(c.values())
                for word in d[k]:
                    f.write("%s %s %s\n" % (k, word, float(c[word] / total)))

    # read from a directory to a nested dictionary
    file_gen = get_filename_from_folder(directory)
    d = defaultdict(Counter)
    i = 0
    for file in file_gen:
        i += 1
        lines = [line for line in open(join_path(directory, file), 'r', encoding='utf-8')]
        # add to defaultdict with x as key and a counter as value
        # the counter counts y's appearance
        for line in lines:
            x = line.split()[outer_index]
            y = line.split()[inner_index]
            if d[x]:
                d[x].update([y])
            else:
                d[x] = Counter([y])

    write_out_count()
    print("Finished writing count. Total file processed: ", i)
    write_out_prob()
    print("Finished writing probability")
    return


def calculate_stats():
    """output a file with frequency of extraction from previously stored files"""
    calculate_individual_p(VO_DIR, "verb_noun.txt", 1, 0)
    calculate_individual_p(VO_DIR, "noun_verb.txt", 0, 1)
    calculate_individual_p(NP_DIR, "noun_adj.txt", 0, 1)
    calculate_individual_p(NP_DIR, "adj_noun.txt", 1, 0)
    return


def compute_verb_adj_set():
    """compute and write out verb adjectives set"""
    output_filename = join_path(OUTPUT_DIR, 'verb_adj_pair.txt')
    if file_exists(output_filename):
        print("%s already exist" % output_filename)
        return

        # {object : {verb: prob(v_n)}}
    d_ov = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_verb_noun.txt'), 0, 1, 2)

    # {object: {adj: prob(n_a}}
    d_oa = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_noun_adj.txt'), 1, 0, 2)
    print("finish reading into dictionary")

    verb_adj_set = set()
    for o in d_ov:
        for adj, _ in d_oa[o].items():
            for v, _ in d_ov[o].items():
                verb_adj_set.add((adj, v))
    verb_adj_set = sorted(verb_adj_set)
    with open(output_filename, 'w', encoding='utf-8') as w:
        for adj, v in verb_adj_set:
            w.write("%s %s\n" % (adj, v))


def calculate_verb_given_adj():
    """calculate and write the prob(verb|adj)"""
    output_filename = join_path(OUTPUT_DIR, 'prob_verb_adj.txt')
    if file_exists(output_filename):
        print("%s already exist" % output_filename)
        return

    with open(join_path(OUTPUT_DIR, 'verb_adj_pair.txt'), 'r', encoding='utf-8') as r:
        adj_verb = {}
        for line in r.readlines():
            adj_verb[line] = 0
        # adj_verb = [line.split() for line in r.readlines()]

    # {object : {verb: prob(v_n)}}
    d_ov = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_verb_noun.txt'), 0, 1, 2)

    # {object: {adj: prob(n_a}}
    d_oa = read_to_nested_dict(join_path(OUTPUT_DIR, 'prob_noun_adj.txt'), 1, 0, 2)

    # {verb : [objects]}
    d_vo = read_to_dict(join_path(OUTPUT_DIR, 'prob_verb_noun.txt'), 1, 0)

    for pair in adj_verb:
        adj, verb = pair.split()
        prob = sum(
            # P(V_O) * P(O_A)
            d_ov[obj][verb] * d_oa[obj][adj]
            for obj in d_vo[verb]
        )
        # print(adj, verb, prob)
        if not 0 <= prob <= 1:
            # print(adj, verb, prob)
            continue
        adj_verb[pair] = prob
    with open(output_filename, 'w', encoding='utf-8') as f:
        for key, value in adj_verb.items():
            f.write("%s %s\n" % (key, value))


def overall(noun):
    """calculate overall probability of verb given noun"""

    dict_adj_noun = search_file(join_path(OUTPUT_DIR, "prob_adj_noun.txt"), noun)
    temp_dict = {}
    # p1 is P(adj|noun)
    for adj, p1 in dict_adj_noun.items():
        dict_verb_adj = search_file(join_path(OUTPUT_DIR, "prob_verb_adj_3.txt"), adj)
        # p2 is P(verb|adj)
        for verb, p2 in dict_verb_adj.items():
            if verb in temp_dict:
                temp_dict[verb] += p1 * p2
            else:
                temp_dict[verb] = p1 * p2
    print(sorted(temp_dict.items(), key=(lambda kv: kv[1]), reverse=True)[:100])


def pipe():
    """pipeline for extraction and probability calculation"""
    # extract_from_directory(STORY_DIRECTORY)
    # print("finish extract from directory at %s" % str(datetime.datetime.now()))
    # calculate_stats()
    # print("finish calculating stats at %s" % str(datetime.datetime.now()))
    # compute_verb_adj_set()
    # calculate_verb_given_adj()
    # print("finish calculating prob(v|adj) at %s" % str(datetime.datetime.now()))
    list = ["knife", "lamp", "coffee", "stone", "bottle", "cake"]
    for word in list:
        overall(word)


def main():  # pylint: disable= missing-docstring
    start = time.time()
    pipe()
    end = time.time()
    print("total time cost %s" % datetime.timedelta(seconds=(end - start)))


if __name__ == '__main__':
    main()
